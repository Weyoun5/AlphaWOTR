# -*- coding: ascii -*-
"""Regression pins for the 2026-08-29 audit fix wave, owner AI2
(src/ai.py + src/engine_play.py; second AI owner).

One plain test_* function per finding; every pin was run RED against
the unfixed tree before the fix landed (AI2_FIX_REPORT.md quotes the RED
line).  Findings: 1.254 / CB-7 remainder (Words of Power, Black Breath,
Blade of Westernesse, Heroic Death target picks as learned decisions),
EX-8 (Shadow table-card kill payment pair as a learned pending), MC4
(kernel-exact chance nodes under a blinded root follow each simulation's
sampled commit), MC7 (_state_sig distinguishes WHICH companion / WHICH
nation's unit), 1.230 principle (the in-place zero-effect declare at
progress 0 is offered).
"""
import os
import sys
import traceback

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('PYTHONHASHSEED', '0')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('WOTR_ACT_GEN', 'v41')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import board
import game_state as gs
import game_logic
import ai
import cards
import combat as cb
import combat_dist as cd
from board import sanitize
from game_logic import Minion
from game_state import (Side, Nation, UnitKind, Unit, Leader, Nazgul,
                        ActionDieFace as Face, PersistentCard,
                        PersistentCardType)

FP = Side.FREE_PEOPLES
SH = Side.SHADOW
GON, SAU = Nation.GONDOR, Nation.SAURON


# ---------------------------------------------------------------- helpers
def setup_game():
    b = board.load_board()
    st = gs.setup_initial_game(b)
    game_logic.initialize_decks(st)
    for attr in ('_declare_phase', '_eye_allocation_phase'):
        setattr(st, attr, False)
    st.pending_discard = None
    return st


def set_dice(st, side, faces):
    st.dice[side] = []
    for f in faces:
        d = gs.ActionDie(faces=[f], side=side)
        d.current = f
        st.dice[side].append(d)


def place(st, rk, nation, regulars=0, elites=0, leaders=0, nazgul=0,
          clear=False):
    a = game_logic.get_or_create_army(st, rk)
    if clear:
        a.units.clear()
        a.leaders.clear()
    for _ in range(regulars):
        a.units.append(Unit(nation=nation, kind=UnitKind.REGULAR))
    for _ in range(elites):
        a.units.append(Unit(nation=nation, kind=UnitKind.ELITE))
    for _ in range(leaders):
        a.leaders.append(Leader(nation=nation))
    for _ in range(nazgul):
        a.leaders.append(Nazgul(nation=Nation.SAURON))
    return a


def take_companion(st, name):
    for c in list(st.fellowship.companions):
        if c.name == name:
            st.fellowship.companions.remove(c)
            return c
    raise KeyError(name)


def army_at(st, rk):
    return next((v for k, v in st.armies.items() if sanitize(k) == rk),
                None)


def leader_names(army):
    return [getattr(l, 'name', type(l).__name__)
            for l in (army.leaders if army is not None else [])]


class Rec:
    """scripted roller (mirrors test_audit0829_CB): pops scripted
    batches in engine call order, else fills with `fill`."""

    def __init__(self, scripts=(), fill=1):
        self.scripts = list(scripts)
        self.fill = fill
        self.calls = []

    def __call__(self, n, ctx='', side=None):
        self.calls.append((n, ctx))
        if self.scripts:
            s = self.scripts.pop(0)
            return list(s)[:n] + [self.fill] * max(0, n - len(s))
        return [self.fill] * n


def with_roller(rec, fn):
    o = cb.roll_d6_batch
    cb.roll_d6_batch = rec
    try:
        return fn()
    finally:
        cb.roll_d6_batch = o


def battle(st, atk_region, def_region, atk_side, atk_card=None,
           def_card=None, phase='defender_combat_card'):
    """Open a pending field battle the way the round machinery leaves it
    after the attacker's commit: phase 'defender_combat_card' (the
    defender decides), atk card recorded (None = declined)."""
    def_side = SH if atk_side == FP else FP
    pc = {'attacker_region': atk_region, 'defender_region': def_region,
          'attacker_side': atk_side, 'defender_side': def_side,
          'phase': phase,
          'deciding_side': (def_side if phase == 'defender_combat_card'
                            else atk_side),
          'is_siege': False, 'rounds_fought': 0,
          'atk_combat_card': atk_card}
    if def_card is not None:
        pc['def_combat_card'] = def_card
    st.pending_combat = pc
    return pc


def decline(st, side):
    ai._execute_action(st, ai.Action(
        die_idx=-1, action_type='combat_card_none',
        params={'side': 'fp' if side == FP else 'sh'}))


def effect_actions(st, side):
    return [a for a in ai.get_legal_actions(st, side)
            if a.action_type == 'combat_effect_choice']


def cname(cid):
    c = cards.get_card(cid)
    return (getattr(c, 'combat_name', '') or '') if c else ''


# ------------------------------------------------- 1.254 target picks
def test_1254_words_of_power_target_is_learned():
    """Words of Power: WHICH Companion is cancelled is the Shadow owner's
    learned decision - the round opens a combat_effect_choice pending
    for the SHADOW with one action per Companion, and the pick reaches
    combat.apply_words_of_power."""
    st = setup_game()
    place(st, 'gorgoroth', SAU, regulars=5, nazgul=1, clear=True)
    d = place(st, 'dagorlad', GON, regulars=3, clear=True)
    d.leaders += [take_companion(st, 'Legolas'),
                  take_companion(st, 'Boromir')]
    set_dice(st, FP, [Face.CHARACTER])
    set_dice(st, SH, [Face.ARMY])
    wop = cd.combat_card_id('Words of Power')
    battle(st, 'gorgoroth', 'dagorlad', SH, atk_card=wop)
    decline(st, FP)
    pc = st.pending_combat or {}
    assert pc.get('phase') == 'combat_effect_choice' \
        and pc.get('deciding_side') == SH, (
            '1.254 WoP: no SHADOW target pending opened (phase=%s '
            'deciding=%s)' % (pc.get('phase'), pc.get('deciding_side')))
    ec = pc.get('effect_choice') or {}
    assert ec.get('effect') == 'words_of_power' \
        and sorted(ec.get('targets') or []) == ['Boromir', 'Legolas'], ec
    menu = effect_actions(st, SH)
    assert sorted(a.params.get('target') for a in menu) \
        == ['Boromir', 'Legolas'], [a.params for a in menu]
    seen = {}
    orig = cb.apply_words_of_power

    def spy(combat, choices=None):
        orig(combat, choices)
        seen['choice'] = (choices or {}).get('atk_effect_target')
        seen['target'] = getattr(getattr(combat, '_wop_cancelled', None),
                                 'name', None)
    cb.apply_words_of_power = spy
    try:
        pick = next(a for a in menu if a.params.get('target') == 'Boromir')
        with_roller(Rec(fill=1), lambda: ai._execute_action(st, pick))
    finally:
        cb.apply_words_of_power = orig
    assert seen.get('choice') == 'Boromir' and seen.get('target') == 'Boromir', (
        '1.254 WoP: the pick did not reach the engine: %r' % (seen,))
    lrc = getattr(st, '_last_round_call', None) or {}
    assert (lrc.get('choices') or {}).get('atk_effect_target') == 'Boromir', lrc


def test_1254_black_breath_target_is_learned():
    """Black Breath: the Shadow owner picks the figure (pre-roll
    conditional pick: Companion by name or 'leader'); with a scripted
    2-hit re-roll the picked Meriadoc dies and Legolas (the heuristic's
    max-level victim) lives."""
    st = setup_game()
    place(st, 'gorgoroth', SAU, regulars=5, nazgul=2, clear=True)
    d = place(st, 'dagorlad', GON, regulars=3, leaders=1, clear=True)
    d.leaders += [take_companion(st, 'Meriadoc'),
                  take_companion(st, 'Legolas')]
    set_dice(st, FP, [Face.CHARACTER])
    set_dice(st, SH, [Face.ARMY])
    bb = cd.combat_card_id('Black Breath')
    battle(st, 'gorgoroth', 'dagorlad', SH, atk_card=bb)
    decline(st, FP)
    pc = st.pending_combat or {}
    assert pc.get('phase') == 'combat_effect_choice' \
        and pc.get('deciding_side') == SH, (
            '1.254 BB: no SHADOW target pending opened (phase=%s '
            'deciding=%s)' % (pc.get('phase'), pc.get('deciding_side')))
    ec = pc.get('effect_choice') or {}
    assert ec.get('effect') == 'black_breath' \
        and sorted(ec.get('targets') or []) == ['Legolas', 'Meriadoc',
                                                'leader'], ec
    menu = effect_actions(st, SH)
    assert sorted(a.params.get('target') for a in menu) \
        == ['Legolas', 'Meriadoc', 'leader'], [a.params for a in menu]
    pick = next(a for a in menu if a.params.get('target') == 'Meriadoc')
    # engine batch order: attacker roll (5 dice, all miss), defender
    # roll (3, all miss), attacker re-roll (leadership 2 -> [6, 6] = 2
    # re-roll hits), defender re-roll (fill)
    rec = Rec(scripts=[[1] * 5, [1] * 3, [6, 6]], fill=1)
    with_roller(rec, lambda: ai._execute_action(st, pick))
    names = leader_names(army_at(st, 'dagorlad'))
    assert 'Meriadoc' not in names and 'Legolas' in names, (
        '1.254 BB: the owner picked Meriadoc but the engine left %r '
        '(batches %r)' % (names, rec.calls))


def test_1254_blade_of_westernesse_target_is_learned():
    """Blade of Westernesse: the Free Peoples owner picks the Minion;
    the picked Mouth of Sauron dies, the roster-first Witch King
    lives."""
    st = setup_game()
    place(st, 'dagorlad', GON, regulars=4, leaders=1, clear=True)
    d = place(st, 'gorgoroth', SAU, regulars=3, nazgul=1, clear=True)
    d.leaders += [Minion(nation=SAU, name='Witch King'),
                  Minion(nation=SAU, name='Mouth of Sauron')]
    set_dice(st, FP, [Face.CHARACTER])
    set_dice(st, SH, [Face.ARMY])
    bow = cd.combat_card_id('Blade of Westernesse')
    battle(st, 'dagorlad', 'gorgoroth', FP, atk_card=bow)
    decline(st, SH)
    pc = st.pending_combat or {}
    assert pc.get('phase') == 'combat_effect_choice' \
        and pc.get('deciding_side') == FP, (
            '1.254 BoW: no FREE PEOPLES target pending opened (phase=%s '
            'deciding=%s)' % (pc.get('phase'), pc.get('deciding_side')))
    ec = pc.get('effect_choice') or {}
    assert ec.get('effect') == 'blade_of_westernesse' \
        and sorted(ec.get('targets') or []) == ['Mouth of Sauron',
                                                'Witch King'], ec
    menu = effect_actions(st, FP)
    assert sorted(a.params.get('target') for a in menu) \
        == ['Mouth of Sauron', 'Witch King'], [a.params for a in menu]
    pick = next(a for a in menu
                if a.params.get('target') == 'Mouth of Sauron')
    # attacker (FP) roll 4 dice miss, defender 3 miss, FP re-roll
    # (leadership 1) -> [6] = 1 re-roll hit -> Blade fires
    rec = Rec(scripts=[[1] * 4, [1] * 3, [6]], fill=1)
    with_roller(rec, lambda: ai._execute_action(st, pick))
    names = leader_names(army_at(st, 'gorgoroth'))
    assert 'Mouth of Sauron' not in names and 'Witch King' in names, (
        '1.254 BoW: the owner picked the Mouth but the engine left %r '
        '(batches %r)' % (names, rec.calls))


def test_1254_heroic_death_pick_is_learned():
    """Heroic Death: (count, target) pairs - the FP owner picks WHICH
    same-level Companion dies; (2, Gimli) eliminates Gimli, not the
    list-first Boromir."""
    st = setup_game()
    place(st, 'gorgoroth', SAU, regulars=5, nazgul=1, clear=True)
    d = place(st, 'dagorlad', GON, regulars=4, leaders=1, clear=True)
    d.leaders += [take_companion(st, 'Boromir'),
                  take_companion(st, 'Gimli')]
    set_dice(st, FP, [Face.CHARACTER])
    set_dice(st, SH, [Face.ARMY])
    hd = cd.combat_card_id('Heroic Death')
    pc = battle(st, 'gorgoroth', 'dagorlad', SH, atk_card=None,
                def_card=hd)
    ai._run_combat_round(st, pc)
    pc = st.pending_combat or {}
    assert pc.get('phase') == 'combat_effect_choice' \
        and pc.get('deciding_side') == FP, pc
    menu = effect_actions(st, FP)
    pairs = {(int(a.params.get('count', 0) or 0), a.params.get('target'))
             for a in menu}
    assert pairs == {(0, None), (1, 'leader'), (2, 'Boromir'),
                     (2, 'Gimli')}, (
        '1.254 HD: expected (count, target) pairs, menu=%r'
        % sorted(pairs, key=str))
    pick = next(a for a in menu if a.params.get('target') == 'Gimli')
    # attacker 5 dice [6,6,6,1,1] = 3 hits; defender 4 miss; attacker
    # re-roll (nazgul 1) miss; defender re-roll (3) miss
    rec = Rec(scripts=[[6, 6, 6, 1, 1], [1] * 4, [1], [1] * 3], fill=1)
    with_roller(rec, lambda: ai._execute_action(st, pick))
    names = leader_names(army_at(st, 'dagorlad'))
    assert 'Gimli' not in names and 'Boromir' in names \
        and 'Leader' in names, (
            '1.254 HD: the owner picked Gimli but the engine left %r '
            '(batches %r)' % (names, rec.calls))


def test_1254_single_target_auto_resolves():
    """Convention guard (never RED by design): a single legal target is
    no decision - Words of Power with ONE Companion in the battle opens
    no pending, the round runs (the engine's default resolves to the
    sole figure).  Mirrors the D&D / RA / HD 'open only with a real
    choice' convention."""
    st = setup_game()
    place(st, 'gorgoroth', SAU, regulars=5, nazgul=1, clear=True)
    d = place(st, 'dagorlad', GON, regulars=3, clear=True)
    d.leaders.append(take_companion(st, 'Legolas'))
    set_dice(st, FP, [Face.CHARACTER])
    set_dice(st, SH, [Face.ARMY])
    wop = cd.combat_card_id('Words of Power')
    battle(st, 'gorgoroth', 'dagorlad', SH, atk_card=wop)
    with_roller(Rec(fill=1), lambda: decline(st, FP))
    pc = st.pending_combat or {}
    assert pc.get('phase') != 'combat_effect_choice', pc
    assert isinstance(getattr(st, '_last_round_call', None), dict), \
        'the round did not run'


def test_1254_target_variants_encode_distinctly():
    """Target variants must not collapse (v11 victor_advance class):
    distinct encodings + distinct cache keys, with NO growth of
    ACTION_ENCODING_DIM (v41 = 93; the shipped nets keep loading)."""
    st = setup_game()
    bb = cd.combat_card_id('Black Breath')
    acts = [ai.Action(die_idx=-1, action_type='combat_effect_choice',
                      params={'card_id': bb, 'target': t})
            for t in ('Meriadoc', 'Legolas', 'leader')]
    encs = [ai.encode_action(a, st, SH) for a in acts]
    assert encs[0].shape[0] == ai.ACTION_ENCODING_DIM == 93, \
        ai.ACTION_ENCODING_DIM
    assert not np.array_equal(encs[0], encs[1]), \
        '1.254: Meriadoc / Legolas target picks encode identically'
    assert not np.array_equal(encs[0], encs[2]), \
        '1.254: companion / leader target picks encode identically'
    keys = {ai._action_cache_key(a, st, SH) for a in acts}
    assert len(keys) == 3, '1.254: target picks share a cache key'
    # Heroic Death pairs: same count, different companion
    hd = cd.combat_card_id('Heroic Death')
    p1 = ai.Action(die_idx=-1, action_type='combat_effect_choice',
                   params={'card_id': hd, 'count': 2, 'target': 'Boromir'})
    p2 = ai.Action(die_idx=-1, action_type='combat_effect_choice',
                   params={'card_id': hd, 'count': 2, 'target': 'Gimli'})
    assert not np.array_equal(ai.encode_action(p1, st, FP),
                              ai.encode_action(p2, st, FP))
    assert ai._action_cache_key(p1, st, FP) != ai._action_cache_key(p2, st, FP)


def test_1254_teacher_path_keeps_heuristic():
    """Guard (never RED by design): under the teacher (FINAL_HITS_HOOK
    set, log = truth) no target pending opens - the logged round
    replays with the engine heuristic; the count matcher is untouched."""
    st = setup_game()
    place(st, 'gorgoroth', SAU, regulars=5, nazgul=1, clear=True)
    d = place(st, 'dagorlad', GON, regulars=3, clear=True)
    d.leaders += [take_companion(st, 'Legolas'),
                  take_companion(st, 'Boromir')]
    set_dice(st, FP, [Face.CHARACTER])
    set_dice(st, SH, [Face.ARMY])
    wop = cd.combat_card_id('Words of Power')
    battle(st, 'gorgoroth', 'dagorlad', SH, atk_card=wop)
    prev = cb.FINAL_HITS_HOOK
    cb.FINAL_HITS_HOOK = lambda combat, a, d: (a, d)
    try:
        with_roller(Rec(fill=1), lambda: decline(st, FP))
    finally:
        cb.FINAL_HITS_HOOK = prev
    pc = st.pending_combat or {}
    assert pc.get('phase') != 'combat_effect_choice', pc


# ------------------------------------------------------------------ EX-8
def _table_kill_fixture():
    st = setup_game()
    hand = st.hands[SH].hand
    del hand[:]
    army = [c for c in st.sh_strat_deck
            if getattr(cards.get_card(c), 'card_type', None) == 'Army'][:2]
    chars = [c for c in st.sh_char_deck if '_Character_' in c][:2]
    for c in army:
        st.sh_strat_deck.remove(c)
    for c in chars:
        st.sh_char_deck.remove(c)
    hand[:] = [army[0], chars[0], army[1], chars[1]]
    st.persistent_cards.add_card(PersistentCard(
        card_id='FREE_PEOPLES_Strategy_A_Power_too_Great',
        card_name='A Power too Great', side=FP,
        card_type=PersistentCardType.POWER_TOO_GREAT))
    set_dice(st, SH, [Face.ARMY])
    set_dice(st, FP, [Face.CHARACTER])
    return st, army, chars


def _in_discards(st, cid):
    # 1.261: payment cards file FACE-DOWN to the payer's tracked pile
    return cid in st.hands[SH].discarded_face_down


def test_ex8_table_kill_payment_is_learned():
    """EX-8: the Shadow's forced discard of one Army-type + one
    Character card is a learned pending (every legal pair), resolved by
    card_option; the bridge answers it through
    ai.table_kill_payment_action."""
    st, army, chars = _table_kill_fixture()
    hand = st.hands[SH].hand
    kills = [a for a in ai.get_legal_actions(st, SH)
             if a.action_type == 'discard_table_card']
    assert kills, 'fixture: table kill not offered'
    ai._execute_action(st, kills[0])
    pco = getattr(st, 'pending_card_option', None)
    assert pco is not None and pco.get('side') == 'sh' \
        and pco.get('table_kill_payment'), (
            'EX-8: the payment pair was auto-picked (pending=%r hand=%r)'
            % (pco, hand))
    assert sorted(hand) == sorted(army + chars), \
        'EX-8: cards left the hand before the pair was chosen: %r' % hand
    assert st.persistent_cards.get_by_type(
        PersistentCardType.POWER_TOO_GREAT) is None, 'table card not killed'
    assert st.dice[SH][0].used, 'die not spent'
    opts = pco.get('options') or []
    assert len(opts) == 4 and all(
        o.get('army_card') in army and o.get('character_card') in chars
        for o in opts), opts
    menu = ai.get_legal_actions(st, SH)
    assert len(menu) == 4 and all(a.action_type == 'card_option'
                                  for a in menu), [a.action_type for a in menu]
    pick = ai.table_kill_payment_action(st, army[1], chars[1])
    assert any(a.params.get('choices') == pick.params.get('choices')
               for a in menu), (pick.params, [a.params for a in menu])
    # two distinct pairs encode distinctly (the pair IS the decision)
    e1 = ai.encode_action(menu[0], st, SH)
    e2 = ai.encode_action(pick, st, SH)
    assert not np.array_equal(e1, e2), 'EX-8: payment pairs encode identically'
    ai._execute_action(st, pick)
    assert getattr(st, 'pending_card_option', None) is None
    assert sorted(hand) == sorted([army[0], chars[0]]), hand
    assert _in_discards(st, army[1]) and _in_discards(st, chars[1]), \
        'EX-8/1.261: paid cards not in the Shadow face-down pile ' \
        '(conservation)'
    assert not any(army[1] in lst or chars[1] in lst
                   for lst in (st.discard_face_up or {}).values()), \
        '1.261: payment cards leaked to a face-up pile'
    # a pair outside the hand is refused loudly
    try:
        ai.table_kill_payment_action(st, army[1], chars[1])
    except Exception:
        pass
    else:
        raise AssertionError('EX-8: a stale pair built an action')


def test_ex8_teacher_path_auto_pays():
    """Guard (never RED by design): the teacher (log = truth) keeps the
    historical first-in-hand payment."""
    st, army, chars = _table_kill_fixture()
    hand = st.hands[SH].hand
    kills = [a for a in ai.get_legal_actions(st, SH)
             if a.action_type == 'discard_table_card']
    prev = cb.FINAL_HITS_HOOK
    cb.FINAL_HITS_HOOK = lambda combat, a, d: (a, d)
    try:
        ai._execute_action(st, kills[0])
    finally:
        cb.FINAL_HITS_HOOK = prev
    assert getattr(st, 'pending_card_option', None) is None
    assert sorted(hand) == sorted([army[1], chars[1]]), hand


# ------------------------------------------------------------------- MC4
def _blinded_root():
    """Root = FP defender at defender_combat_card with the attacker's
    card committed and UNREVEALED (the blinded window)."""
    st = setup_game()
    osg = next(k for k in st.armies if 'osgiliath' in sanitize(str(k)))
    mm = next(k for k in st.armies if 'morgul' in sanitize(str(k)))
    x = next(c for c in st.sh_char_deck
             if (cname(c) or '').strip())
    st.sh_char_deck.remove(x)
    st.discard_face_up['sh_character'].append(x)
    del st.hands[FP].hand[:]
    set_dice(st, FP, [Face.CHARACTER, Face.ARMY])
    set_dice(st, SH, [Face.ARMY, Face.MUSTER])
    battle(st, mm, osg, SH, atk_card=x)
    root = ai.MCTSNode(state=ai.clone_state(st), side=FP)
    root.state.pending_combat['_commit_nonce'] = True
    return root


def _built_under(branch):
    return (getattr(branch.state, '_last_round_call', None) or {}).get(
        'atk_card')


def test_mc4_blinded_root_kernel_sets_follow_the_sim_sample():
    """MC4: at a blinded root a kernel-expanded child must serve each
    simulation the outcome set of THAT simulation's sampled commit -
    never the first sample frozen."""
    import random
    random.seed(5)
    root = _blinded_root()
    act = ai.Action(die_idx=-1, action_type='combat_card_none',
                    params={'side': 'fp'})
    ai._SEARCH_ROOT_SIDE = FP
    try:
        mix = ai._atk_commit_mixture(root.state, root.state.pending_combat)
        assert len(mix) >= 2, mix

        def world(cid):
            ov = ai._SimOverlay(root.state, FP)
            ov.commit_cid = cid
            ai._SIM_OVERLAY = ov
            return ov

        # two sampled worlds with DIFFERENT combat sides whose apply
        # rolls and kernel-expands
        cands = []
        for cid, _w in mix:
            nm = cname(cid)
            if any(nm == c[1] for c in cands):
                continue
            world(cid)
            ai._overlay_prepare_apply(root.state, FP, act)
            st2, used = ai._apply_action_detect_rng(root.state, act)
            if not used or not isinstance(
                    getattr(st2, '_last_round_call', None), dict):
                continue
            probe = ai.MCTSNode(state=None, side=FP, parent=root,
                                action=act, prior=1.0)
            if ai._kernel_expand_combat(probe, st2):
                cands.append((cid, nm))
            if len(cands) == 2:
                break
        assert len(cands) == 2, 'fixture: need two kernel-ok worlds: %r' % cands
        (A, _an), (B, _bn) = cands

        child = ai.MCTSNode(state=None, side=FP, parent=root, action=act,
                            prior=1.0)
        world(A)
        ai._overlay_prepare_apply(root.state, FP, act)
        stA, _ = ai._apply_action_detect_rng(root.state, act)
        assert ai._kernel_expand_combat(child, stA)
        n0 = ai.KERNEL_STATS['kernel_nodes']
        brA = ai._select_outcome(child)
        assert _built_under(brA) == A, (_built_under(brA), A)
        world(B)
        brB = ai._select_outcome(child)
        assert _built_under(brB) == B, (
            'MC4: simulation sampled %s but descended a branch built '
            'under %s (first sample frozen)' % (B, _built_under(brB)))
        world(A)
        brA2 = ai._select_outcome(child)
        assert _built_under(brA2) == A, (_built_under(brA2), A)
        assert ai.KERNEL_STATS['kernel_nodes'] - n0 == 1, (
            'MC4: expected exactly one extra expansion (B), got %d'
            % (ai.KERNEL_STATS['kernel_nodes'] - n0))
    finally:
        ai._SIM_OVERLAY = None
        ai._SEARCH_ROOT_SIDE = None


# ------------------------------------------------------------------- MC7
def test_mc7_state_sig_distinguishes_companion_and_nation():
    """MC7: outcomes that differ only in WHICH same-level companion or
    WHICH nation's unit is gone must not merge."""
    base = setup_game()
    s1 = ai.clone_state(base)
    take_companion(s1, 'Legolas')
    s2 = ai.clone_state(base)
    take_companion(s2, 'Gimli')
    for s in (s1, s2):
        assert ai._state_sig(s) is not None
    assert ai._state_sig(s1) != ai._state_sig(s2), \
        'MC7: Legolas-gone and Gimli-gone fellowships share a _state_sig'
    assert ai._chance_sig(s1) != ai._chance_sig(s2), \
        'MC7: Legolas-gone and Gimli-gone fellowships share a _chance_sig'
    # same-level companions leading two armies, swapped
    s3 = ai.clone_state(base)
    game_logic.get_or_create_army(s3, 'dale').leaders.append(
        take_companion(s3, 'Legolas'))
    game_logic.get_or_create_army(s3, 'erebor').leaders.append(
        take_companion(s3, 'Gimli'))
    s4 = ai.clone_state(base)
    game_logic.get_or_create_army(s4, 'dale').leaders.append(
        take_companion(s4, 'Gimli'))
    game_logic.get_or_create_army(s4, 'erebor').leaders.append(
        take_companion(s4, 'Legolas'))
    assert ai._state_sig(s3) != ai._state_sig(s4), \
        'MC7: swapped same-level army leaders share a _state_sig'
    # which nation's regular died
    s5 = ai.clone_state(base)
    place(s5, 'dale', Nation.THE_NORTH, regulars=2, clear=True)
    place(s5, 'dale', Nation.DWARVES, regulars=1)
    s6 = ai.clone_state(base)
    place(s6, 'dale', Nation.THE_NORTH, regulars=1, clear=True)
    place(s6, 'dale', Nation.DWARVES, regulars=2)
    assert ai._state_sig(s5) != ai._state_sig(s6), \
        'MC7: 2N+1D and 1N+2D at Dale share a _state_sig'
    assert ai._chance_sig(s5) != ai._chance_sig(s6)


# ----------------------------------------------------- 1.230 principle
def test_1230_in_place_declare_offered_and_completes():
    """The in-place declaration at progress 0 is a legal declaration
    (zero-effect plays are legal, always, even in self-play - 1.230):
    offered in the declare phase; executing it completes with the pawn
    where it was, progress 0, still hidden.  The teacher's historical
    progress > 0 gate is unchanged."""
    st = setup_game()
    f = st.fellowship
    assert f.hidden and f.progress == 0 and f.corruption == 0
    assert sanitize(game_logic.fellowship_region(st)) == 'rivendell'
    assert st.political.active.get(Nation.ELVES)
    st._declare_phase = True
    try:
        menu = ai.get_legal_actions(st, FP)
    finally:
        st._declare_phase = False
    decl = [a for a in menu if a.action_type == 'fellowship_declare'
            and sanitize(str(a.params.get('region'))) == 'rivendell']
    assert decl, (
        '1.230: the zero-effect in-place declare at progress 0 is pruned '
        '(menu=%r)' % [(a.action_type, a.params) for a in menu])
    assert any(a.action_type == 'pass' for a in menu)
    ai._execute_action(st, decl[0])
    assert f.progress == 0 and f.hidden and f.corruption == 0
    assert sanitize(game_logic.fellowship_region(st)) == 'rivendell'
    assert st.pending_combat is None
    assert ai._get_declare_actions(st) == [], \
        'teacher gate (allow_in_place=False) must stay progress > 0'


# ------------------------------------------------------------------ main
TESTS = [
    test_1254_words_of_power_target_is_learned,
    test_1254_black_breath_target_is_learned,
    test_1254_blade_of_westernesse_target_is_learned,
    test_1254_heroic_death_pick_is_learned,
    test_1254_single_target_auto_resolves,
    test_1254_target_variants_encode_distinctly,
    test_1254_teacher_path_keeps_heuristic,
    test_ex8_table_kill_payment_is_learned,
    test_ex8_teacher_path_auto_pays,
    test_mc4_blinded_root_kernel_sets_follow_the_sim_sample,
    test_mc7_state_sig_distinguishes_companion_and_nation,
    test_1230_in_place_declare_offered_and_completes,
]


def main():
    fails = 0
    for t in TESTS:
        try:
            t()
            print('[PASS] %s' % t.__name__)
        except Exception as e:  # noqa: BLE001 - report every failure
            fails += 1
            print('[FAIL] %s: %s: %s' % (t.__name__, type(e).__name__,
                                         str(e)[:600]))
            traceback.print_exc()
    print('TOTAL=%d FAIL=%d' % (len(TESTS), fails))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
