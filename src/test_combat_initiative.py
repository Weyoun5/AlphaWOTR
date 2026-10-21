#!/usr/bin/env python3
"""Combat-card INITIATIVE resolution tests (USER spec, 2026-08).

One deterministic scenario per lettered interaction A-M (B split i/ii,
F into its sub-claims).  Cards are forced onto minimal combat states and
the engine's resolution appliers are driven directly (or via
resolve_combat_round / _resolve_pre_combat / _apply_bb_hd), asserting the
observable outcome: card-cancellation, Leadership counts, hit counts,
which effects fired, and battle-end short-circuits.

Core ordering rules exercised:
  1. lower combat_value resolves first;
  2. tie -> DEFENDER first;
  3. battle-end interrupts resolution (attacker's card never fires);
  4. magnitudes computed AT RESOLUTION TIME;
  5. Leadership can only be forfeited once (White Rider / shine).

Run standalone (no unicode in prints - cp1252 console).
"""

import os
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import combat as cb
import game_state as gs
import game_logic
import board
from cards import get_cards, can_play_combat_card
from game_state import Side, Nation, UnitKind


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(s):
    s = unicodedata.normalize('NFKD', s or '')
    return ''.join(c for c in s if not unicodedata.combining(c)).lower()


_CARD_CACHE = {}


def _cc(combat_name):
    """Return a CombatCard by its printed combat_name (accent-insensitive)."""
    key = _norm(combat_name)
    if key in _CARD_CACHE:
        return _CARD_CACHE[key]
    for c in get_cards().values():
        if _norm(getattr(c, 'combat_name', '') or '') == key:
            cc = c.to_combat_card()
            _CARD_CACHE[key] = cc
            return cc
    raise ValueError('combat card not found: ' + combat_name)


def _fresh():
    b = board.load_board()
    st = gs.setup_initial_game(b)
    game_logic.initialize_decks(st)
    st.armies.clear()
    return st


def _place(st, region, units, leaders=None):
    a = game_logic.get_or_create_army(st, region)
    a.units = list(units)
    a.leaders = list(leaders or [])
    return a


def _u(nation, kind=UnitKind.REGULAR, n=1):
    return [gs.Unit(nation, kind) for _ in range(n)]


def _naz(n=1):
    return [gs.Nazgul(nation=Nation.SAURON) for _ in range(n)]


def _comp(name, level, leadership):
    return gs.Companion(name=name, level=level, leadership=leadership)


def _setup(st, atk_region, def_region, shine=False):
    """setup_combat with NO cards passed (cards set manually afterwards so
    the FP-legality nulling in setup_combat cannot drop them)."""
    if shine:
        st.pending_combat = {'gtw_shine': True}
    else:
        st.pending_combat = None
    return cb.setup_combat(st, atk_region, def_region)


class RollPatch:
    """Context manager patching combat.roll_d6_batch.

    scripts : list of result-lists popped in call order (then falls back).
    fixed   : after scripts exhausted, every die = this face (default 1 =
              guaranteed miss; use 6 for guaranteed hit).
    record  : list appended with (n, ctx) for every batch.
    """

    def __init__(self, scripts=None, fixed=1, record=None):
        self.scripts = list(scripts or [])
        self.fixed = fixed
        self.record = record
        self._orig = None

    def __enter__(self):
        self._orig = cb.roll_d6_batch

        def fake(n, ctx='', side=None):
            # `side` mirrors combat.roll_d6_batch's signature (whose
            # dice these are); the fake ignores it like the default body
            if self.record is not None:
                self.record.append((n, ctx))
            if self.scripts:
                return list(self.scripts.pop(0))
            return [self.fixed] * n

        cb.roll_d6_batch = fake
        return self

    def __exit__(self, *a):
        cb.roll_d6_batch = self._orig
        return False


# ---------------------------------------------------------------------------
# A. Daring Defiance (0) cancels Words of Power (1); White Rider exception
# ---------------------------------------------------------------------------

def test_A_daring_defiance_vs_words_of_power():
    # A-normal: DD (FP defender) cancels WoP (Shadow attacker).
    st = _fresh()
    _place(st, 'Morannon', _u(Nation.SAURON, n=4), _naz(1))     # SH attacker + Nazgul
    _place(st, 'Dagorlad', _u(Nation.GONDOR, n=3),
           [_comp('Legolas', 2, 1), _comp('Aragorn', 3, 2)])     # FP defender
    c = _setup(st, 'Morannon', 'Dagorlad')
    c.attacker_card = _cc('Words of Power')
    c.defender_card = _cc('Daring Defiance')
    cb.apply_cancel_effects(c)
    assert c.attacker_card_cancelled is True, \
        'A: DD should cancel Words of Power normally'

    # A-White-Rider: Gandalf the White has shined (forfeited Leadership).
    # DD is legally played but CANNOT pay its all-Companion forfeit ->
    # NO effect -> Words of Power applies to one Companion; the others
    # retain their Leadership.
    st2 = _fresh()
    _place(st2, 'Morannon', _u(Nation.SAURON, n=4), _naz(1))
    _place(st2, 'Dagorlad', _u(Nation.GONDOR, n=3),
           [_comp('Gandalf the White', 3, 1),
            _comp('Legolas', 2, 1), _comp('Aragorn', 3, 2)])
    c2 = _setup(st2, 'Morannon', 'Dagorlad', shine=True)
    # shine forfeited Gandalf's own Leadership: 1+1+2 - 1 = 3
    assert c2.defender_leadership == 3, c2.defender_leadership
    assert c2._shined_companion is not None
    c2.attacker_card = _cc('Words of Power')
    c2.defender_card = _cc('Daring Defiance')
    cb.apply_cancel_effects(c2)
    assert c2.attacker_card_cancelled is False, \
        'A: shined DD must NOT cancel Words of Power (no forfeit possible)'
    cb.apply_words_of_power(c2)
    # WoP cancels the highest-Leadership companion (Aragorn, 2). Legolas
    # (not targeted) retains 1; Gandalf already 0 from shine -> total 1.
    assert c2.defender_leadership == 1, \
        ('A: after WoP only the untargeted companion retains Leadership; '
         'got %d' % c2.defender_leadership)
    print('  PASS A: DD cancels WoP; White Rider -> DD inert, WoP applies')


# ---------------------------------------------------------------------------
# B. Swarm of Bats (0) vs Daring Defiance (0) - defender first
# ---------------------------------------------------------------------------

def test_B_i_shadow_attacking():
    # (i) Shadow attacking (FP defends): DD (defender) resolves first,
    # forfeits all Companion Leadership, cancels SoB.
    st = _fresh()
    _place(st, 'Morannon', _u(Nation.SAURON, n=5), _naz(1))
    _place(st, 'Dagorlad', _u(Nation.GONDOR, n=3),
           [_comp('Legolas', 2, 1), _comp('Gimli', 2, 1)])
    c = _setup(st, 'Morannon', 'Dagorlad')
    c.attacker_card = _cc('Swarm of Bats')
    c.defender_card = _cc('Daring Defiance')
    assert c.defender_leadership == 2, c.defender_leadership
    cb.apply_cancel_effects(c)
    assert c.attacker_card_cancelled is True, 'B(i): SoB cancelled by DD'
    assert c.defender_leadership == 0, \
        'B(i): all Companions forfeited Leadership'

    # (i)+White Rider: DD inert -> SoB activates, cancels DD, Companions
    # RETAIN Leadership, and Shadow gets NO re-roll bonus (FP played DD).
    st2 = _fresh()
    _place(st2, 'Morannon', _u(Nation.SAURON, n=5), _naz(1))
    _place(st2, 'Dagorlad', _u(Nation.GONDOR, n=3),
           [_comp('Gandalf the White', 3, 1), _comp('Legolas', 2, 1)])
    c2 = _setup(st2, 'Morannon', 'Dagorlad', shine=True)
    assert c2.defender_leadership == 1, c2.defender_leadership  # 1+1-1(shine)
    c2.attacker_card = _cc('Swarm of Bats')
    c2.defender_card = _cc('Daring Defiance')
    cb.apply_cancel_effects(c2)
    assert c2.defender_card_cancelled is True, 'B(i)WR: SoB cancels DD'
    assert c2.attacker_card_cancelled is False, 'B(i)WR: SoB not cancelled'
    assert c2.defender_leadership == 1, 'B(i)WR: Companions retain Leadership'
    cb.apply_dice_modifiers(c2)
    assert c2.attacker_reroll_mod == 0, \
        'B(i)WR: SoB gives NO re-roll bonus (FP did play a card)'
    print('  PASS B(i): Shadow attacking - DD first (or inert under WR)')


def test_B_ii_fp_attacking():
    # (ii) FP attacking (Shadow defends): SoB (defender) resolves first,
    # cancels DD -> Companions RETAIN Leadership. (Old attacker-first code
    # would wrongly let DD forfeit + cancel first.)
    st = _fresh()
    _place(st, 'Dagorlad', _u(Nation.GONDOR, n=3),
           [_comp('Legolas', 2, 1), _comp('Gimli', 2, 1)])       # FP attacker
    _place(st, 'Morannon', _u(Nation.SAURON, n=5), _naz(1))       # SH defender
    c = _setup(st, 'Dagorlad', 'Morannon')
    c.attacker_card = _cc('Daring Defiance')
    c.defender_card = _cc('Swarm of Bats')
    assert c.attacker_leadership == 2, c.attacker_leadership
    cb.apply_cancel_effects(c)
    assert c.attacker_card_cancelled is True, 'B(ii): SoB cancels DD first'
    assert c.attacker_leadership == 2, \
        'B(ii): Companions retain Leadership (DD never forfeited)'
    print('  PASS B(ii): FP attacking - SoB (defender) first, companions keep '
          'leadership')


# ---------------------------------------------------------------------------
# C. Swarm of Bats always negates Scouts (0 < Scouts)
# ---------------------------------------------------------------------------

def test_C_swarm_negates_scouts():
    st = _fresh()
    _place(st, 'Morannon', _u(Nation.SAURON, n=5), _naz(1))
    _place(st, 'Dagorlad', _u(Nation.GONDOR, n=4))
    c = _setup(st, 'Morannon', 'Dagorlad')
    c.attacker_card = _cc('Swarm of Bats')
    c.defender_card = _cc('Scouts')
    cb.apply_cancel_effects(c)
    assert c.defender_card_cancelled is True, 'C: SoB must negate Scouts'
    print('  PASS C: Swarm of Bats (0) always negates Scouts')


# ---------------------------------------------------------------------------
# D. Confusion: no combat roll if FP eliminated before it (Durin's Bane)
# ---------------------------------------------------------------------------

def test_D_confusion_no_roll_when_army_gone():
    st = _fresh()
    _place(st, 'Morannon', _u(Nation.SAURON, n=5))               # SH attacker
    _place(st, 'Dagorlad', _u(Nation.GONDOR, n=3))               # FP: 3 units
    c = _setup(st, 'Morannon', 'Dagorlad')
    c.attacker_card = _cc("Durin's Bane")
    c.defender_card = _cc('Confusion')
    with RollPatch(fixed=6):    # Durin's Bane [6,6,6] = 3 hits -> wipes FP
        cb.resolve_combat_round(c)
    assert c.defender_destroyed is True and c.battle_over is True, \
        'D: FP army eliminated by pre-combat Durin\'s Bane'
    assert c.attacker_combat_rolls == [], 'D: no Combat roll happened'
    assert c.attacker_confusion is False and c.defender_hits == 0, \
        'D: Confusion never resolved (battle already over)'
    print('  PASS D: no Combat roll / Confusion after pre-combat wipe')


# ---------------------------------------------------------------------------
# E. Daylight-class dice caps do NOT affect card-granted pre/post attacks
# ---------------------------------------------------------------------------

def test_E_daylight_ignores_card_attacks():
    # E1 (post-combat): Daylight caps the Shadow Combat roll to 3, but
    # We Come to Kill's post-casualty attack still rolls 5 (its elites).
    st = _fresh()
    _place(st, 'Morannon', _u(Nation.SAURON, UnitKind.ELITE, 5))  # SH 5 elites
    _place(st, 'Dagorlad', _u(Nation.GONDOR, n=3))               # FP 3 units
    c = _setup(st, 'Morannon', 'Dagorlad')
    c.attacker_card = _cc('We Come to Kill')
    c.defender_card = _cc('Daylight')
    rec = []
    with RollPatch(fixed=1, record=rec):   # all misses -> FP survives -> WCtK
        cb.resolve_combat_round(c)
    assert c.attacker_dice_cap == 3, 'E1: Daylight caps Shadow Combat roll'
    sizes = [n for n, _ in rec]
    assert 5 in sizes, \
        'E1: We Come to Kill rolled 5 dice (uncapped by Daylight); got %r' % sizes

    # E2 (pre-combat): a "maximum of two dice" cap (Huorn-dark) does NOT
    # shrink Durin's Bane's own 3-die pre-combat attack.
    st2 = _fresh()
    _place(st2, 'Morannon', _u(Nation.SAURON, n=5))
    _place(st2, 'Dagorlad', _u(Nation.GONDOR, n=6))
    c2 = _setup(st2, 'Morannon', 'Dagorlad')
    c2.attacker_card = _cc("Durin's Bane")
    c2.defender_card = _cc('Huorn-dark')
    rec2 = []
    with RollPatch(fixed=1, record=rec2):
        cb.resolve_combat_round(c2)
    assert c2.attacker_dice_cap == 2, 'E2: Huorn-dark caps Shadow Combat roll to 2'
    assert 3 in [n for n, _ in rec2], \
        'E2: Durin\'s Bane still rolled its 3 pre-combat dice; got %r' % rec2
    print('  PASS E: dice caps ignore Durin\'s Bane / We Come to Kill attacks')


# ---------------------------------------------------------------------------
# F. Mumakil "more total hits" test (at initiative 5)
# ---------------------------------------------------------------------------

def _bare_combat(atk_side=Side.SHADOW, def_side=Side.FREE_PEOPLES,
                 atk_leaders=None, def_leaders=None):
    return gs.CombatState(
        attacker_region='a', defender_region='d',
        attacker_army=gs.Army(region='a', leaders=list(atk_leaders or [])),
        defender_army=gs.Army(region='d', leaders=list(def_leaders or [])),
        attacker_side=atk_side, defender_side=def_side)


def test_F_mumakil_hit_test():
    mum = _cc('Mumakil')

    # F1: Shield-Wall (6, casualty prevention) does NOT influence the
    # Mumakil determination (init 5, computed from raw hits before it).
    c = _bare_combat()
    c.attacker_card = mum
    c.defender_card = _cc('Shield-Wall')
    c.attacker_hits, c.defender_hits = 2, 1
    cb.apply_bonus_hits(c)
    assert c.attacker_bonus_hits == 1, 'F1: Mumakil fires on 2 > 1'
    cb.apply_casualty_prevention(c)             # Shield-Wall now
    assert c.attacker_bonus_hits == 1, \
        'F1: Shield-Wall must not retroactively change Mumakil'

    # F2: Sudden Strike (2) pre-combat hits DO count toward the FP total.
    c = _bare_combat()
    c.attacker_card = mum
    c.defender_card = _cc('Sudden Strike')
    c.attacker_hits = 2
    c.defender_hits = 1
    c.defender_pre_combat_hits = 2              # Sudden Strike scored 2
    cb.apply_bonus_hits(c)
    assert c.attacker_bonus_hits == 0, \
        'F2: Sudden Strike hits raise FP total (3) -> Mumakil suppressed'

    # F3: Blade of Westernesse redirects a re-roll hit to kill a Minion but
    # the hit is NOT consumed from the count -> STILL counts for Mumakil.
    minion = game_logic.Minion(nation=Nation.SAURON, name='Mouth of Sauron')
    c = _bare_combat(atk_leaders=[minion])
    c.attacker_card = mum
    c.defender_card = _cc('Blade of Westernesse')
    c.minions_in_battle = ['Mouth of Sauron']
    c.defender_hits = 2                          # 2 re-roll hits scored
    cb.apply_blade_of_westernesse(c, reroll_hits=2, is_attacker=False)
    assert all(getattr(l, 'name', '') != 'Mouth of Sauron'
               for l in c.attacker_army.leaders), 'F3: Blade killed the Minion'
    assert c.defender_hits == 2, 'F3: the redirected hit is NOT consumed'
    c.attacker_hits = 2
    cb.apply_bonus_hits(c)
    assert c.attacker_bonus_hits == 0, \
        'F3: Blade hit still counts (2 vs 2 tie) -> Mumakil does not fire'

    # F4: Confusion 1s become FP hits (defender_hits) -> count for Mumakil.
    c = _bare_combat()
    c.attacker_card = mum
    c.defender_card = _cc('Confusion')
    cb.apply_confusion_effect(c)
    assert c.attacker_confusion is True, 'F4: Confusion armed on Shadow roll'
    c.attacker_combat_rolls = [1, 1, 5]
    c.attacker_hits = 1                          # the natural 5
    # engine Phase-4 confusion rule: defender_hits += count of Shadow 1s
    c.defender_hits += sum(1 for r in c.attacker_combat_rolls if r == 1)
    cb.apply_bonus_hits(c)
    assert c.attacker_bonus_hits == 0, \
        'F4: Confusion hits (2) count for FP total -> Mumakil suppressed (1<2)'

    # F5: No Quarter / Nameless Wood (5) hits count for Mumakil ONLY when
    # the FP Army is DEFENDING (tie at 5 -> defender first).
    #   FP defending: No Quarter resolves first -> counts -> Mumakil off.
    c = _bare_combat()
    c.attacker_card = mum                        # Shadow attacker
    c.defender_card = _cc('No Quarter')          # FP defender
    c.attacker_hits, c.defender_hits = 2, 1
    cb.apply_bonus_hits(c)
    assert c.defender_bonus_hits == 1, 'F5a: No Quarter fired'
    assert c.attacker_bonus_hits == 0, \
        'F5a: FP defending -> No Quarter hit counted -> Mumakil suppressed'
    #   FP attacking: Mumakil (defender) resolves first -> No Quarter NOT
    #   counted -> Mumakil fires.
    c = _bare_combat(atk_side=Side.FREE_PEOPLES, def_side=Side.SHADOW)
    c.attacker_card = _cc('No Quarter')          # FP attacker
    c.defender_card = mum                        # Shadow defender
    c.attacker_hits, c.defender_hits = 1, 2
    cb.apply_bonus_hits(c)
    assert c.defender_bonus_hits == 1, \
        'F5b: FP attacking -> Mumakil (defender) fires (No Quarter not counted)'
    assert c.attacker_bonus_hits == 1, 'F5b: No Quarter still fired for FP'
    print('  PASS F: Mumakil hit-test (Shield-Wall/Sudden Strike/Blade/'
          'Confusion/No Quarter ordering)')


# ---------------------------------------------------------------------------
# G. Valour: effect stands even if FP Elites are gone at resolution
# ---------------------------------------------------------------------------

def test_G_valour_survives_elite_loss():
    st = _fresh()
    _place(st, 'Morannon', _u(Nation.SAURON, n=5))              # SH attacker
    _place(st, 'Dagorlad', _u(Nation.GONDOR, UnitKind.ELITE, 3))  # FP 3 elites
    c = _setup(st, 'Morannon', 'Dagorlad')
    c.attacker_card = _cc("Durin's Bane")
    c.defender_card = _cc('Valour')
    # Durin's Bane kills FP Elites (2 hits) BEFORE the Combat roll; Valour's
    # +1 was set at Phase 2 and must persist.
    with RollPatch(scripts=[[6, 6, 1]], fixed=1):
        cb.resolve_combat_round(c)
    assert c.defender_dice_mod == 1, \
        'G: Valour +1 stands despite pre-combat Elite loss'
    assert c.defender_destroyed is False, 'G: FP survived to roll'
    assert len(c.defender_casualties) >= 1, 'G: Durin\'s Bane reduced the Elites'
    print('  PASS G: Valour still applies after Elites reduced pre-roll')


# ---------------------------------------------------------------------------
# H. Black Breath (6) vs Heroic Death (6) - defender first
# ---------------------------------------------------------------------------

def test_H_black_breath_vs_heroic_death():
    # H1: Shadow DEFENDING -> Black Breath first; eliminating the sole FP
    # leader leaves Heroic Death (FP attacker) nothing to sacrifice.
    st = _fresh()
    leg = _comp('Legolas', 2, 1)
    _place(st, 'Dagorlad', _u(Nation.GONDOR, n=3), [leg])        # FP attacker
    _place(st, 'Morannon', _u(Nation.SAURON, n=3), _naz(1))      # SH defender
    c = _setup(st, 'Dagorlad', 'Morannon')
    c.attacker_card = _cc('Heroic Death')       # FP
    c.defender_card = _cc('Black Breath')        # SH
    c.companions_in_battle = [leg]
    c.nazgul_in_battle = 1
    c.defender_hits = 3                          # hits Heroic Death could cancel
    cb._apply_bb_hd(c, atk_reroll_hits=0, def_reroll_hits=2,
                    choices={'atk_effect_count': 2})
    assert leg not in c.companions_in_battle, 'H1: Black Breath eliminated Legolas'
    assert c.attacker_hits_cancelled == 0, \
        'H1: Heroic Death could not fire (no FP leader left)'

    # H2: FP DEFENDING -> Heroic Death first; sacrificing the sole FP leader
    # leaves Black Breath (SH attacker) no target.
    st2 = _fresh()
    leg2 = _comp('Legolas', 2, 1)
    _place(st2, 'Morannon', _u(Nation.SAURON, n=3), _naz(1))     # SH attacker
    _place(st2, 'Dagorlad', _u(Nation.GONDOR, n=3), [leg2])      # FP defender
    c2 = _setup(st2, 'Morannon', 'Dagorlad')
    c2.attacker_card = _cc('Black Breath')       # SH
    c2.defender_card = _cc('Heroic Death')       # FP
    c2.companions_in_battle = [leg2]
    c2.nazgul_in_battle = 1
    c2.attacker_hits = 3                          # hits Heroic Death cancels
    cb._apply_bb_hd(c2, atk_reroll_hits=2, def_reroll_hits=0,
                    choices={'def_effect_count': 2})
    assert c2.defender_hits_cancelled == 2, \
        'H2: Heroic Death (defender) fired first'
    assert len(c2.defender_army.leaders) == 0, 'H2: Legolas sacrificed'
    print('  PASS H: Black Breath vs Heroic Death resolved defender-first')


# ---------------------------------------------------------------------------
# I. Cruel as Death / They are Terrible / Dread and Despair unplayable
#    (no effect) when Gandalf the White is shining
# ---------------------------------------------------------------------------

def _shine_pair(card_name, shine, naz=2, fp_units=3):
    """Build (with/without shine) a SH-attacker vs FP-defender combat and
    return the combat with the given Shadow combat card assigned."""
    st = _fresh()
    leaders = _naz(naz)
    _place(st, 'Morannon', _u(Nation.SAURON, n=5), leaders)
    fp_leaders = [_comp('Gandalf the White', 3, 1)] if shine else []
    _place(st, 'Dagorlad', _u(Nation.GONDOR, n=fp_units), fp_leaders)
    c = _setup(st, 'Morannon', 'Dagorlad', shine=shine)
    c.attacker_card = _cc(card_name)
    return c


def test_I_nazgul_leadership_cards_negated_by_white_rider():
    # Cruel as Death (+1 combat by forfeiting 2 Nazgul Leadership)
    c_off = _shine_pair('Cruel as Death', shine=False)
    cb.apply_dice_modifiers(c_off)
    assert c_off.attacker_dice_mod == 1, 'I: Cruel as Death works without shine'
    c_on = _shine_pair('Cruel as Death', shine=True)
    cb.apply_dice_modifiers(c_on)
    assert c_on.attacker_dice_mod == 0, \
        'I: Cruel as Death negated by shining Gandalf the White'

    # They are Terrible (+1 re-roll by forfeiting 1 Nazgul Leadership)
    c_off = _shine_pair('They are Terrible', shine=False)
    cb.apply_they_are_terrible(c_off)
    assert c_off.attacker_reroll_mod == 1, 'I: They are Terrible works w/o shine'
    c_on = _shine_pair('They are Terrible', shine=True)
    cb.apply_they_are_terrible(c_on)
    assert c_on.attacker_reroll_mod == 0, \
        'I: They are Terrible negated by shine'

    # Dread and Despair (forfeit Nazgul Leadership -> FP rolls fewer dice)
    c_off = _shine_pair('Dread and Despair', shine=False)
    cb.apply_dread_and_despair(c_off)
    assert c_off.defender_dice_cap is not None, \
        'I: Dread and Despair reduces FP dice without shine'
    c_on = _shine_pair('Dread and Despair', shine=True)
    cb.apply_dread_and_despair(c_on)
    assert c_on.defender_dice_cap is None, \
        'I: Dread and Despair negated by shine'

    # Documented gap: the play-LEGALITY gate lives in cards.can_play_combat_card
    # (not owned here); it still keys off the raw nazgul_in_battle count, so it
    # reports the card as playable even under shine. The resolution self-gate
    # above nullifies the effect. Record the current behavior for the report.
    global _I_LEGALITY_PLAYABLE
    cruel = None
    for cobj in get_cards().values():
        if _norm(getattr(cobj, 'combat_name', '') or '') == _norm('Cruel as Death'):
            cruel = cobj
            break
    c_on = _shine_pair('Cruel as Death', shine=True)
    ok, _why = can_play_combat_card(cruel, c_on, is_attacker=True)
    _I_LEGALITY_PLAYABLE = bool(ok)
    print('  PASS I: Nazgul-Leadership-forfeit cards inert under White Rider '
          '(legality gate still playable=%s - see report)' % _I_LEGALITY_PLAYABLE)


# ---------------------------------------------------------------------------
# J. Durin's Bane (2) vs Charge (2) - defender first, interrupt, magnitude
# ---------------------------------------------------------------------------

def test_J_pre_combat_defender_first_and_magnitude():
    # J1 (interrupt): FP DEFENDING with Charge resolves first; if it wipes
    # the Shadow attacker, Durin's Bane (attacker) never fires.
    st = _fresh()
    _place(st, 'Morannon', _u(Nation.SAURON, n=2))              # SH attacker (2)
    _place(st, 'Dagorlad', _u(Nation.GONDOR, UnitKind.ELITE, 2))  # FP 2 elites
    c = _setup(st, 'Morannon', 'Dagorlad')
    c.attacker_card = _cc("Durin's Bane")
    c.defender_card = _cc('Charge')
    with RollPatch(fixed=6):    # Charge [6,6] = 2 hits -> wipes SH attacker
        cb.resolve_combat_round(c)
    assert c.attacker_destroyed is True and c.battle_over is True, \
        'J1: Charge (defender) eliminated the Shadow Army first'
    assert c.attacker_pre_combat_hits == 0, \
        'J1: Durin\'s Bane never fired (attacker card interrupted)'

    # J2 (resolution-time magnitude): FP ATTACKING with Charge; Shadow
    # DEFENDING with Durin's Bane resolves first (defender on tie) and kills
    # FP Elites, so Charge's dice count reflects the Elites still present.
    st2 = _fresh()
    _place(st2, 'Dagorlad', _u(Nation.GONDOR, UnitKind.ELITE, 3))  # FP 3 elites
    _place(st2, 'Morannon', _u(Nation.SAURON, n=5))               # SH defender
    c2 = _setup(st2, 'Dagorlad', 'Morannon')
    c2.attacker_card = _cc('Charge')             # FP
    c2.defender_card = _cc("Durin's Bane")       # SH
    rec = []
    with RollPatch(scripts=[[6, 6, 6]], fixed=1, record=rec):
        cb._resolve_pre_combat(c2)
    sizes = [n for n, _ in rec]
    # Durin's Bane batch (3) then Charge batch (1 - only 1 Elite survives the
    # 3 hits: 1 killed, 1 downgraded, 1 left).
    assert sizes == [3, 1], \
        ('J2: Durin\'s Bane (3) resolves first and Charge rolls the reduced '
         'Elite count; got %r' % sizes)
    assert c2.attacker_destroyed is False, 'J2: FP survived'
    print('  PASS J: pre-combat defender-first, interrupt, resolution magnitude')


# ---------------------------------------------------------------------------
# K. Foul Stench (3) negates re-roll cards; not Sudden Strike (2)
# ---------------------------------------------------------------------------

def test_K_foul_stench():
    # K1: Foul Stench cancels the FP re-roll and blocks Mighty Attack.
    st = _fresh()
    leg = _comp('Legolas', 2, 1)
    _place(st, 'Morannon', _u(Nation.SAURON, n=4), _naz(3))       # SH + 3 Nazgul
    _place(st, 'Dagorlad', _u(Nation.GONDOR, n=3), [leg])         # FP defender
    c = _setup(st, 'Morannon', 'Dagorlad')
    c.attacker_card = _cc('Foul Stench')
    c.defender_card = _cc('Mighty Attack')
    c.companions_in_battle = [leg]
    # Nazgul Leadership (3) >= FP Leadership -> Foul Stench cancels re-roll.
    cb.apply_reroll_cancellation(c)
    assert c.defender_reroll_cancelled is True, 'K1: Foul Stench cancelled re-roll'
    conv = cb.apply_mighty_attack(c, rolls=[1, 1], threshold=5, modifier=0,
                                  is_attacker=False)
    assert conv == 0, 'K1: Mighty Attack / Anduril negated by Foul Stench'
    # Blade of Westernesse / Fateful Strike need re-roll hits; a cancelled
    # re-roll yields 0 -> they cannot fire.
    minion = game_logic.Minion(nation=Nation.SAURON, name='Mouth of Sauron')
    c.attacker_army.leaders.append(minion)
    c.defender_card = _cc('Blade of Westernesse')
    cb.apply_blade_of_westernesse(c, reroll_hits=0, is_attacker=False)
    assert any(getattr(l, 'name', '') == 'Mouth of Sauron'
               for l in c.attacker_army.leaders), \
        'K1: Blade cannot fire with 0 re-roll hits (re-roll cancelled)'

    # K2: Sudden Strike (2) is a PRE-combat attack (before the re-roll), so
    # Foul Stench does NOT negate it.
    st2 = _fresh()
    _place(st2, 'Morannon', _u(Nation.SAURON, n=4), _naz(3))
    _place(st2, 'Dagorlad', _u(Nation.GONDOR, n=3),
           [_comp('Aragorn', 3, 2), _comp('Legolas', 2, 1)])
    c2 = _setup(st2, 'Morannon', 'Dagorlad')
    c2.attacker_card = _cc('Foul Stench')
    c2.defender_card = _cc('Sudden Strike')
    with RollPatch(fixed=6):     # Sudden Strike rolls its Leadership dice, all hit
        cb._apply_pre_combat_for_side(c2, is_attacker=False)
    assert c2.defender_pre_combat_hits > 0, \
        'K2: Sudden Strike fired despite Foul Stench (resolves earlier)'
    print('  PASS K: Foul Stench negates re-roll cards, not Sudden Strike')


# ---------------------------------------------------------------------------
# L. We Come to Kill applies AFTER Heroic Death / Shield-wall / all others
# ---------------------------------------------------------------------------

def test_L_we_come_to_kill_last():
    wctk = _cc('We Come to Kill')
    # LAST by initiative: strictly greater than the init-6 cluster.
    assert wctk.combat_value == 7, 'L: We Come to Kill initiative should be 7'
    for name in ('Heroic Death', 'Shield-Wall', 'Black Breath',
                 'Blade of Westernesse'):
        assert wctk.combat_value > _cc(name).combat_value, \
            'L: We Come to Kill resolves after ' + name

    # Functional: it fires in the post-casualty window on the survivors.
    c = _bare_combat()
    c.attacker_army = gs.Army(region='a',
                              units=_u(Nation.SAURON, UnitKind.ELITE, 5))
    c.defender_army = gs.Army(region='d', units=_u(Nation.GONDOR, n=4))
    c.attacker_card = wctk
    with RollPatch(fixed=6):     # 5 elites at 5+ = 5 hits
        cb._apply_post_casualty_for_side(c, is_attacker=True)
    assert c.attacker_bonus_hits == 5, 'L: We Come to Kill post-casualty attack fired'
    assert len(c.defender_casualties) >= 1, 'L: survivors took We Come to Kill hits'
    print('  PASS L: We Come to Kill is last (init 7, post-casualty)')


# ---------------------------------------------------------------------------
# M. Sudden Strike (2) after Words of Power (1): WoP-cancelled Leadership
#    is not counted in the Sudden Strike roll
# ---------------------------------------------------------------------------

def test_M_words_of_power_shrinks_sudden_strike():
    st = _fresh()
    _place(st, 'Morannon', _u(Nation.SAURON, n=4), _naz(1))       # SH + Nazgul
    _place(st, 'Dagorlad', _u(Nation.GONDOR, n=3),
           [_comp('Aragorn', 3, 2), _comp('Legolas', 2, 1)])      # FP ldr 3
    c = _setup(st, 'Morannon', 'Dagorlad')
    c.attacker_card = _cc('Words of Power')      # SH, init 1
    c.defender_card = _cc('Sudden Strike')       # FP, init 2
    assert c.defender_leadership == 3, c.defender_leadership
    cb.apply_words_of_power(c)                    # cancels Aragorn (2)
    assert c.defender_leadership == 1, \
        'M: Words of Power cancelled the highest Companion Leadership'
    rec = []
    with RollPatch(fixed=1, record=rec):
        cb._apply_pre_combat_for_side(c, is_attacker=False)   # Sudden Strike
    assert rec and rec[0][0] == 1, \
        ('M: Sudden Strike rolls min(5, remaining Leadership)=1, not the '
         'pre-cancel 3; got %r' % rec)
    print('  PASS M: Sudden Strike uses WoP-reduced Leadership')


# ---------------------------------------------------------------------------
# N. RULING 1.228 (USER 2026-08-03): Fateful Strike vs Black Breath, both
#    initiative 6 - the defender-first tie rule is NOT relevant.  FS
#    activates when its play condition (the Leader re-roll) is met, so it
#    resolves at Phase 5d; BB remains playable and resolves at Phase 7.
#    Locked-in edge case (CL-216, certified): FS killing the SOLE Nazgul
#    at 5d blocks BB via its resolver gate (nazgul_in_battle < 1).
# ---------------------------------------------------------------------------

def test_N_fs_bb_both_resolve_1228():
    # N1 (appliers in engine phase order): TWO Nazgul present -> FS kills
    # one at 5d, BB still fires at 7 and eliminates the Companion.
    st = _fresh()
    leg = _comp('Legolas', 2, 1)
    _place(st, 'Dagorlad', _u(Nation.GONDOR, n=3), [leg])        # FP attacker
    _place(st, 'Morannon', _u(Nation.SAURON, n=3), _naz(2))      # SH defender
    c = _setup(st, 'Dagorlad', 'Morannon')
    c.attacker_card = _cc('Fateful Strike')      # FP, init 6
    c.defender_card = _cc('Black Breath')        # SH, init 6
    assert c.nazgul_in_battle == 2, c.nazgul_in_battle
    cb.apply_fateful_strike(c, 1, is_attacker=True)      # Phase 5d
    assert c.nazgul_in_battle == 1, \
        'N1: Fateful Strike eliminated a Nazgul at Phase 5d'
    cb._apply_bb_hd(c, atk_reroll_hits=0, def_reroll_hits=2, choices={})
    assert leg not in c.companions_in_battle, \
        'N1: Black Breath still eliminated the Companion at Phase 7'

    # N2 (single-Nazgul edge, CL-216): FS kills the SOLE Nazgul at 5d ->
    # BB's resolver gate (nazgul_in_battle < 1) blocks it at Phase 7 ->
    # the Companion SURVIVES.  Certified behavior - must not regress.
    st2 = _fresh()
    leg2 = _comp('Legolas', 2, 1)
    _place(st2, 'Dagorlad', _u(Nation.GONDOR, n=3), [leg2])
    _place(st2, 'Morannon', _u(Nation.SAURON, n=3), _naz(1))
    c2 = _setup(st2, 'Dagorlad', 'Morannon')
    c2.attacker_card = _cc('Fateful Strike')
    c2.defender_card = _cc('Black Breath')
    cb.apply_fateful_strike(c2, 1, is_attacker=True)
    assert c2.nazgul_in_battle == 0, 'N2: sole Nazgul eliminated'
    cb._apply_bb_hd(c2, atk_reroll_hits=0, def_reroll_hits=2, choices={})
    assert leg2 in c2.companions_in_battle, \
        'N2: Black Breath blocked by its Nazgul resolver gate (CL-216)'

    # N3 (end-to-end round path): resolve_combat_round runs FS at 5d and
    # BB at 7 off the same scripted rolls - both fire in one round.
    st3 = _fresh()
    merry = _comp('Meriadoc', 1, 1)
    _place(st3, 'Dagorlad', _u(Nation.GONDOR, n=5), [merry])     # FP attacker
    _place(st3, 'Morannon', _u(Nation.SAURON, n=3), _naz(2))     # SH defender
    c3 = _setup(st3, 'Dagorlad', 'Morannon')
    c3.attacker_card = _cc('Fateful Strike')
    c3.defender_card = _cc('Black Breath')
    # batches: atk combat [5 dice], def combat [3], atk re-roll, def re-roll
    with RollPatch(scripts=[[6, 1, 1, 1, 1], [6, 6, 1], [6], [6]], fixed=1):
        cb.resolve_combat_round(c3)
    assert c3.nazgul_in_battle == 1, \
        'N3: FS (1 re-roll hit) removed one of two Nazgul'
    assert sum(1 for l in c3.defender_army.leaders
               if isinstance(l, gs.Nazgul)) == 1, 'N3: one Nazgul left'
    assert merry not in c3.companions_in_battle, \
        'N3: BB (1 re-roll hit >= level 1) eliminated the hobbit at Phase 7'
    assert c3.battle_over is False, 'N3: both armies survived'
    print('  PASS N: 1.228 - FS (5d) and BB (7) both resolve; CL-216 '
          'sole-Nazgul gate kept')


# ---------------------------------------------------------------------------
# O. RULING 1.228c (USER 2026-08-03): combat-card Minion kills do FULL
#    character-sheet bookkeeping - state.witch_king/.saruman/.mouth
#    in_play flips (feeds the ai.py 1.228b Sorcerer gate) and Saruman
#    kills discard The Palantir of Orthanc / Wormtongue from the table.
# ---------------------------------------------------------------------------

def test_O_card_kill_minion_bookkeeping_1228c():
    # O1: Fateful Strike (2+ re-roll hits -> Minion branch) kills the
    # Witch-king (modelled as a Minion figure) -> sheet flips in_play.
    st = _fresh()
    st.witch_king.in_play = True
    wk = game_logic.Minion(nation=Nation.SAURON, name='Witch King')
    _place(st, 'Dagorlad', _u(Nation.GONDOR, n=3), [_comp('Meriadoc', 1, 1)])
    _place(st, 'Morannon', _u(Nation.SAURON, n=3), [wk] + _naz(1))
    c = _setup(st, 'Dagorlad', 'Morannon')
    c.attacker_card = _cc('Fateful Strike')
    cb.apply_fateful_strike(c, 2, is_attacker=True)
    assert all('witch' not in str(getattr(l, 'name', '')).lower()
               for l in c.defender_army.leaders), 'O1: WK popped from army'
    assert 'Witch King' not in c.minions_in_battle, 'O1: minion list updated'
    assert st.witch_king.in_play is False, \
        'O1: card-killed Witch-king must flip the character sheet (1.228c)'

    # O2: Blade of Westernesse kills Saruman -> in_play False AND the
    # Palantir of Orthanc + Wormtongue table cards are DISCARDED (so the
    # Palantir cannot activate at end of battle).
    st2 = _fresh()
    st2.saruman.in_play = True
    st2.persistent_cards.add_card(gs.PersistentCard(
        card_id='Character-93-test', card_name='The Palantir of Orthanc',
        side=Side.SHADOW,
        card_type=gs.PersistentCardType.PALANTIR_OF_ORTHANC))
    st2.persistent_cards.add_card(gs.PersistentCard(
        card_id='Character-94-test', card_name='Wormtongue',
        side=Side.SHADOW, card_type=gs.PersistentCardType.WORMTONGUE))
    sar = game_logic.Minion(nation=Nation.ISENGARD, name='Saruman')
    _place(st2, 'Dagorlad', _u(Nation.GONDOR, n=3), [_comp('Meriadoc', 1, 1)])
    _place(st2, 'Morannon', _u(Nation.ISENGARD, n=3), [sar])
    c2 = _setup(st2, 'Dagorlad', 'Morannon')
    c2.attacker_card = _cc('Blade of Westernesse')
    cb.apply_blade_of_westernesse(c2, 1, is_attacker=True)
    assert st2.saruman.in_play is False, \
        'O2: card-killed Saruman must flip the character sheet (1.228c)'
    assert not st2.persistent_cards.has_card_type(
        gs.PersistentCardType.PALANTIR_OF_ORTHANC), \
        'O2: Palantir of Orthanc discarded on Saruman elimination'
    assert not st2.persistent_cards.has_card_type(
        gs.PersistentCardType.WORMTONGUE), \
        'O2: Wormtongue discarded on Saruman elimination'
    assert len(st2.discard_face_up.get('sh_character', [])) == 2, \
        'O2: both table cards went to the sh_character face-up discard'

    # O3: a battle with NO minions leaves the sheet untouched (FS falls
    # through to its plain-Nazgul branch; plain Nazgul have no sheet
    # entry).
    st3 = _fresh()
    st3.witch_king.in_play = True
    st3.saruman.in_play = True
    _place(st3, 'Dagorlad', _u(Nation.GONDOR, n=3), [_comp('Meriadoc', 1, 1)])
    _place(st3, 'Morannon', _u(Nation.SAURON, n=3), _naz(2))
    c3 = _setup(st3, 'Dagorlad', 'Morannon')
    c3.attacker_card = _cc('Fateful Strike')
    cb.apply_fateful_strike(c3, 2, is_attacker=True)
    assert c3.nazgul_in_battle == 1, 'O3: a plain Nazgul was eliminated'
    assert st3.witch_king.in_play is True and st3.saruman.in_play is True, \
        'O3: non-minion kill must not touch the character sheet'
    print('  PASS O: 1.228c - FS/BoW minion kills flip in_play; Saruman '
          'kill discards Palantir + Wormtongue; non-minion battles inert')


# ---------------------------------------------------------------------------
# P. RULING 1.228 wave (USER 2026-08-03): Black Breath / Heroic Death
#    companion eliminations decrement the Leadership POOL by the
#    Companion's LEADERSHIP rating (Gandalf: level 3, leadership 1), not
#    his level.  Hit-cancel (HD) and viability (BB) magnitudes stay
#    LEVEL-based per the card texts.  NOTE: the pool bug only manifests
#    in the engine-internal multi-round loop (_resolve_combat_body's
#    while loop carries combat.*_leadership across rounds); the
#    per-round ai.py flow rebuilds leadership via setup_combat each
#    round, which masked it.
# ---------------------------------------------------------------------------

def test_P_bb_hd_leadership_rating_decrement():
    # P1: Black Breath eliminates Gandalf the Grey (level 3, leadership
    # 1) -> FP pool drops 2 -> 1 (by leadership), NOT 2 -> 0-clamped -1
    # (by level).
    st = _fresh()
    gan = _comp('Gandalf the Grey', 3, 1)
    leg = _comp('Legolas', 2, 1)
    _place(st, 'Morannon', _u(Nation.SAURON, n=3), _naz(1))      # SH attacker
    _place(st, 'Dagorlad', _u(Nation.GONDOR, n=4), [gan, leg])   # FP defender
    c = _setup(st, 'Morannon', 'Dagorlad')
    c.attacker_card = _cc('Black Breath')
    assert c.defender_leadership == 2, c.defender_leadership
    cb.apply_black_breath(c, 3, is_attacker=True)   # 3 hits >= level 3
    assert gan not in c.companions_in_battle, 'P1: BB eliminated Gandalf'
    assert c.defender_leadership == 1, \
        ('P1: pool must drop by LEADERSHIP rating (1), not level (3); '
         'got %d' % c.defender_leadership)

    # P2: Heroic Death learned-choice sacrifice of a level-2 /
    # leadership-1 Companion: cancels 2 hits (LEVEL-based, card text)
    # but the pool drops only 1 (leadership).
    st2 = _fresh()
    bor = _comp('Boromir', 2, 1)
    gim = _comp('Gimli', 2, 1)
    _place(st2, 'Dagorlad', _u(Nation.GONDOR, n=4), [bor, gim])  # FP attacker
    _place(st2, 'Morannon', _u(Nation.SAURON, n=4))              # SH defender
    c2 = _setup(st2, 'Dagorlad', 'Morannon')
    c2.attacker_card = _cc('Heroic Death')
    c2.defender_hits = 3
    assert c2.attacker_leadership == 2, c2.attacker_leadership
    cb._heroic_death_for_side(c2, True, {'atk_effect_count': 2})
    assert c2.attacker_hits_cancelled == 2, \
        'P2: hit-cancel stays LEVEL-based (2) per the card text'
    assert c2.attacker_leadership == 1, \
        ('P2: pool must drop by LEADERSHIP rating (1), not level (2); '
         'got %d' % c2.attacker_leadership)

    # P3: Heroic Death heuristic fallback path (no learned choice) uses
    # the same leadership-rating decrement.
    st3 = _fresh()
    bor3 = _comp('Boromir', 2, 1)
    gim3 = _comp('Gimli', 2, 1)
    _place(st3, 'Dagorlad', _u(Nation.GONDOR, n=2), [bor3, gim3])
    _place(st3, 'Morannon', _u(Nation.SAURON, n=4))
    c3 = _setup(st3, 'Dagorlad', 'Morannon')
    c3.attacker_card = _cc('Heroic Death')
    c3.defender_hits = 2      # > half of the 2-unit army -> heuristic fires
    cb._heroic_death_for_side(c3, True, None)
    assert c3.attacker_hits_cancelled == 2, \
        'P3: heuristic sacrifice cancels level-based hits'
    assert c3.attacker_leadership == 1, \
        ('P3: heuristic path pool drop = leadership rating (1); got %d'
         % c3.attacker_leadership)
    print('  PASS P: BB/HD eliminations decrement pool by LEADERSHIP '
          'rating; cancel/viability magnitudes stay level-based')


# ---------------------------------------------------------------------------

_I_LEGALITY_PLAYABLE = None

ALL_TESTS = [
    test_A_daring_defiance_vs_words_of_power,
    test_B_i_shadow_attacking,
    test_B_ii_fp_attacking,
    test_C_swarm_negates_scouts,
    test_D_confusion_no_roll_when_army_gone,
    test_E_daylight_ignores_card_attacks,
    test_F_mumakil_hit_test,
    test_G_valour_survives_elite_loss,
    test_H_black_breath_vs_heroic_death,
    test_I_nazgul_leadership_cards_negated_by_white_rider,
    test_J_pre_combat_defender_first_and_magnitude,
    test_K_foul_stench,
    test_L_we_come_to_kill_last,
    test_M_words_of_power_shrinks_sudden_strike,
    test_N_fs_bb_both_resolve_1228,
    test_O_card_kill_minion_bookkeeping_1228c,
    test_P_bb_hd_leadership_rating_decrement,
]


def main():
    failures = 0
    for t in ALL_TESTS:
        try:
            t()
        except AssertionError as e:
            failures += 1
            print('  FAIL %s: %s' % (t.__name__, e))
        except Exception as e:
            failures += 1
            print('  ERROR %s: %r' % (t.__name__, e))
    print('')
    if failures:
        print('=== %d/%d initiative tests FAILED ===' % (failures, len(ALL_TESTS)))
        sys.exit(1)
    print('=== all %d combat-initiative tests passed! ===' % len(ALL_TESTS))


if __name__ == '__main__':
    main()
