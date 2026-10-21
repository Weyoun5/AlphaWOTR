# -*- coding: ascii -*-
"""DFX regression pins - decision-detection census fix wave (2026-08-30).

USER charge (verbatim): "the humans did every decision that the AI sees,
the fact that it doesn't means our detection is wrong" - the DET census
proved every gap is an extractor EMISSION failure.  Each pin replays ONE
real corpus game through rendered_extract.extract_one and asserts the
recovered rows/counters.  RED against the pre-fix extractor, GREEN after.

Pins (census priority ids):
  P0-a  ledger-derived face-up discard piles -> card-status block lit
  P0-b  EX-8 table_kill_payment emitted with the ledger-named pair
  P1-a  combat_effect_choice target/magnitude recovery (BB target row,
        Relentless Assault park-gesture magnitude)
  P1-b  card_casualty overkill-wipe solver (1.254), owner-side row
  P2-a  cc_card_absent -> named-card injection (log = truth)
  P2-b  Mordor-Eye hunt batches drive (prescan seeded)
  INST  emission-completeness table present per game

Run: python test_audit0829_DFX.py   (cwd src/, env WOTR_ACT_GEN=v41)
"""
import os
import sys

os.environ.setdefault('WOTR_ACT_GEN', 'v41')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('PYTHONHASHSEED', '0')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import rendered_extract as rx  # noqa: E402


_ZMETA = None
_CACHE = {}


def _extract(stem):
    """extract_one on a corpus game, memoized (several pins share games)."""
    global _ZMETA
    if stem in _CACHE:
        return _CACHE[stem]
    if _ZMETA is None:
        _ZMETA = rx._z_meta()
    path = os.path.join(rx.RLOG_CORPUS, stem + '.rendered.txt')
    samples, line, _cov = rx.extract_one(path, stem, _ZMETA.get(stem, {}))
    led = rx._LEDGER
    ledger_facts = {
        'discard_events': list(getattr(led, 'discard_events', []) or []),
        'out_fp': list(led.out_at('fp', 10 ** 9)) if led else [],
        'out_sh': list(led.out_at('sh', 10 ** 9)) if led else [],
    }
    _CACHE[stem] = (samples, line, ledger_facts)
    return _CACHE[stem]


G_TKILL = '2023-01-20_183326_FP_CircleDakkon_SP_GenericPie'   # census sec 6.1
G_VAYZE = '2023-01-05_093443_FP_Maximo1877_SP_Vayze'          # census sec 6.2
G_CARDCAS = '2023-01-11_162213_FP_Fojtu_SP_MacDadd'           # census sec 6.3
G_FXTGT = '2023-01-06_104116_FP_Mistakentuna_SP_RompSteel'    # HD pairs x2
G_RAPARK = '2023-11-25_234042_FP_Wirbelwind_SP_Luckyruds'     # RA park K=2


def test_p0a_discard_piles_light_card_status():
    """P0-a: the 96-dim card-status block must carry the ledger-proven
    face-up discards (v42 store averaged 0.3 lit dims vs ~39 cards/game
    out of hands).  Bilbo's Song is PLAYED in this game -> its index (2)
    must be 0.5 in post-play samples; turn>=8 samples must light tens of
    dims."""
    import ai
    samples, line, _lf = _extract(G_VAYZE)
    off = ai._CARD_IDX_START + ai.MAX_HAND_SIZE          # card_status start
    bidx = ai._get_card_idx_map()["FREE_PEOPLES_Character_Bilbo's_Song"]
    lit_bilbo = sum(1 for s in samples if s['x'][off + bidx] > 0.25)
    late = [s for s in samples if s['turn'] >= 8]
    assert late, 'no turn>=8 samples in %s' % G_VAYZE
    mean_lit = float(np.mean([
        int(np.sum(np.asarray(s['x'][off:off + 96]) > 0.25))
        for s in late]))
    assert lit_bilbo >= 1, \
        'card_status[Bilbo] lit in %d samples (want >=1)' % lit_bilbo
    assert mean_lit >= 10.0, \
        'turn>=8 mean lit card-status dims %.2f (want >=10)' % mean_lit


def test_p0b_table_kill_payment_emitted():
    """P0-b: the EX-8 Shadow payment pair (ledger-NAMED: Nazgul Search +
    Denethor's Folly in this game) must emit as a table_kill_payment
    sample; the provenance counter stays."""
    samples, line, _lf = _extract(G_TKILL)
    n_pay = line['emit_by_kind'].get('table_kill_payment', 0)
    assert n_pay >= 1, 'table_kill_payment emitted %d (want >=1)' % n_pay
    ph = [s for s in samples if s.get('phase') == 'table_kill_payment']
    assert len(ph) >= 1, 'no sample carries phase table_kill_payment'
    assert ph[0]['side'] == 'sh', \
        'payment row side %r (want sh)' % ph[0]['side']
    paid = line.get('tablekill_paid') or {}
    assert any('+' in k for k in paid), \
        'tablekill_paid provenance lost: %r' % paid


def test_p1a_effect_target_rows():
    """P1-a: 1.254 target-bearing effect menus (Heroic Death pairs here)
    must match on (count, target) - count-only multi-matched and
    dropped.  This game's two HD pair menus and its one count menu all
    emit; the 'leader' picks carry the target on the OPTKIND scalar
    (encode_action's designed slot for a plain-Leader pick)."""
    import ai
    samples, line, _lf = _extract(G_FXTGT)
    dr = line['drop_by_reason']
    assert dr.get('cf_effect_choice', 0) == 0, \
        'cf_effect_choice drops %d (want 0)' % dr.get('cf_effect_choice', 0)
    assert line['detected'].get('cc_effect_target', 0) >= 2, \
        'cc_effect_target detected %d (want >=2)' \
        % line['detected'].get('cc_effect_target', 0)
    n_fx = line['emit_by_kind'].get('combat_effect', 0)
    assert n_fx >= 3, 'combat_effect emitted %d (want >=3)' % n_fx
    ok = ai._ACT_OPTKIND_OFF
    tgt_rows = 0
    for s in samples:
        if s.get('phase') != 'combat_effect_choice':
            continue
        enc = np.asarray(s['encs'])[int(s['human'])]
        if float(enc[ok]) > 0.5:
            tgt_rows += 1
    assert tgt_rows >= 2, \
        'target-carrying rows %d (want >=2 leader-pick rows)' % tgt_rows


def test_p1a_named_target_multihot_unit():
    """P1-a unit pin (BB/WoP/BoW named-figure class, corpus-rare on the
    drivable surface): a targets menu with the victim NAMED in the kill
    window resolves to the target-carrying action, and its encoding
    lights the character multi-hot."""
    import ai
    import board
    import game_state as gs
    from game_state import Side
    st = gs.setup_initial_game(board.load_board())
    la = [ai.Action(die_idx=-1, action_type='combat_effect_choice',
                    params={'card_id': 'X', 'target': 'Strider'}),
          ai.Action(die_idx=-1, action_type='combat_effect_choice',
                    params={'card_id': 'X', 'target': 'Boromir'})]
    batch = {'records': [{'op': 'kill', 'pieces': ['Strider'],
                          'line': 100, 'zone': 'Gondor'}]}
    ec = {'targets': ['Strider', 'Boromir'], 'effect': 'black_breath'}
    from collections import Counter
    stats = {'drop_reasons': Counter()}
    got = rx._effect_target_pick(batch, ec, la, 'SHADOW', 0, 200, stats)
    assert got == 0, 'named-victim pick returned %r (want 0=Strider)' % got
    enc = np.asarray(ai.encode_action(la[0], st, Side.SHADOW))
    hot = float(np.sum(enc[ai._ACT_CHAR_OFF:ai._ACT_CHAR_OFF + 13]))
    assert hot > 0.5, \
        'named target does not light the character multi-hot (%.2f)' % hot


def test_p1a_relentless_assault_park_magnitude():
    """P1-a sibling: the T4 Dale battle commits Relentless Assault and
    parks [2]Sauron Regular onto the card (the same gesture the
    Onslaught magnitude trusts) - K=2 must match and emit; the game's
    only cf_effect_choice drop clears."""
    samples, line, _lf = _extract(G_RAPARK)
    dr = line['drop_by_reason']
    assert dr.get('cf_effect_choice', 0) == 0, \
        'cf_effect_choice drops %d (want 0)' % dr.get('cf_effect_choice', 0)
    n_fx = line['emit_by_kind'].get('combat_effect', 0)
    assert n_fx >= 1, 'combat_effect emitted %d (want >=1)' % n_fx


def test_p1b_card_casualty_overkill_wipe():
    """P1-b: Dreadful Spells rolls 2 hits into a 1-unit Lorien garrison
    (census sec 6.3) - the exact-hit-equation solver could never satisfy
    overkill; the wipe-tolerant solver must match the human's allocation
    (kills the Elven regular, 0 downgrades) and the row must carry the
    OWNER's side (fp)."""
    samples, line, _lf = _extract(G_CARDCAS)
    dr = line['drop_by_reason']
    assert dr.get('cardcas_unmatched', 0) == 0, \
        'cardcas_unmatched %d (want 0)' % dr.get('cardcas_unmatched', 0)
    n = line['emit_by_kind'].get('cardopt_card_casualty', 0)
    assert n >= 1, 'cardopt_card_casualty emitted %d (want >=1)' % n
    fp_copt = [s for s in samples
               if s.get('phase') == 'card_option' and s['side'] == 'fp']
    assert fp_copt, 'no fp-side card_option row (owner-side doctrine)'


def test_p2a_named_combat_card_injected():
    """P2-a: the combat-card line NAMES the committed card; a hand-ledger
    gap must inject it (log = truth), never drop the decision.  This
    game dropped 3 rounds to cc_card_absent."""
    samples, line, _lf = _extract(G_VAYZE)
    dr = line['drop_by_reason']
    det = line['detected']
    assert dr.get('cc_card_absent', 0) == 0, \
        'cc_card_absent %d (want 0)' % dr.get('cc_card_absent', 0)
    assert det.get('cc_card_injected', 0) >= 1, \
        'cc_card_injected %d (want >=1)' % det.get('cc_card_injected', 0)


def test_p2b_mordor_eye_hunts_drive():
    """P2-b: Mordor-track Eye hunts must drive (the box prescan seeds
    batch['_mordor_box']); the resolver_skip:hunt:special(eye) and its
    huntres twin go to zero on the census proof game."""
    samples, line, _lf = _extract(G_TKILL)
    dr = line['drop_by_reason']
    bad = {k: v for k, v in dr.items()
           if k.startswith('resolver_skip:hunt:special(eye')
           or k.startswith('huntres_skip:hunt:special(e')}
    assert not bad, 'Mordor-Eye hunt batches still skipped: %r' % bad


def test_hidden_info_discards_stay_dark():
    """USER hidden-info ruling (2026-08-31, verbatim: "discards from
    hand are still kept face down. the net does not see those,
    correct?"): the P0-a installer mirrors the ENGINE's face-up pile,
    never the omniscient ledger.  (1) after the replayed table-kill the
    TABLE card shows 0.5 in the payment row's state while BOTH masked
    payment cards show 0.0; (2) a hand-limit / masked discard NEVER
    lights card_status in any stored sample."""
    import ai
    samples, line, lf = _extract(G_TKILL)
    cmap = ai._get_card_idx_map()
    off = ai._CARD_IDX_START + ai.MAX_HAND_SIZE

    def _idx(frag):
        hits = [i for c, i in cmap.items() if frag in c]
        assert len(hits) == 1, 'card id frag %r ambiguous: %s' % (frag, hits)
        return hits[0]

    i_tbl = _idx('Power_too_Great')          # the killed table card
    i_naz = _idx('_Search')                  # payment half 1 (masked)
    i_den = _idx("Denethor's_Folly")         # payment half 2 (masked)
    pay = [s for s in samples if s.get('phase') == 'table_kill_payment']
    assert pay, 'no payment row (P0-b prerequisite)'
    x = np.asarray(pay[0]['x'])
    assert x[off + i_tbl] > 0.25, \
        'killed table card not face-up in the payment state (%.2f)' \
        % x[off + i_tbl]
    assert x[off + i_naz] < 0.25 and x[off + i_den] < 0.25, \
        'MASKED payment cards leaked into card_status (%.2f / %.2f)' \
        % (x[off + i_naz], x[off + i_den])
    # (2) masked/hand-limit discards: a cid that never exited face-up
    # must stay 0.0 in every OPPOSING-side row (1.261: the owner's own
    # rows MAY light its own face-down discards; the opponent's never)
    face_up = set(lf['out_fp']) | set(lf['out_sh'])
    dark = [(t, cid) for (t, _ln, cid) in lf['discard_events']
            if cid is not None and cid not in face_up]
    assert dark, 'no masked discard in %s (pin needs one)' % G_TKILL
    for t, cid in dark:
        j = cmap.get(cid)
        if j is None:
            continue
        opp = 'sh' if t == 'fp' else 'fp'
        vals = [float(np.asarray(s['x'])[off + j]) for s in samples
                if s['side'] == opp]
        worst = max(vals) if vals else 0.0
        assert worst < 0.25, \
            'masked %s discard %s leaked into a %s-side row (%.2f)' \
            % (t, cid, opp, worst)


def test_1261_payment_pair_facedown_per_side():
    """USER ruling 1.261: the EX-8 payment pair files to the PAYER's
    discarded_face_down (never the public piles).  Pin on the installer
    itself: the ledger classifies the CircleDakkon pair (Nazgul Search +
    Denethor's Folly) as Shadow FACE-DOWN outs, and
    _install_ledger_discards carries them into
    st.hands[SHADOW].discarded_face_down while the public
    discard_face_up piles exclude them.  (The per-side 0.5 fill in
    encode_state is EVIS's concurrent ai.py change - this pin covers
    the extraction's half of the contract.)"""
    import board
    import game_state as gs
    from game_state import Side
    stem = G_TKILL
    path = os.path.join(rx.RLOG_CORPUS, stem + '.rendered.txt')
    led = rx._build_hand_ledger(path, stem)
    assert led is not None, 'no hand ledger for %s' % stem

    def _one(frag, cids):
        hits = [c for c in cids if frag in str(c)]
        return hits[0] if len(hits) == 1 else None

    fd_sh = led.outfd_at('sh', 10 ** 9)
    fu = set(led.out_at('fp', 10 ** 9)) | set(led.out_at('sh', 10 ** 9))
    c_naz = _one('_Search', fd_sh)
    c_den = _one("Denethor's_Folly", fd_sh)
    assert c_naz is not None and c_den is not None, \
        'payment pair not classified Shadow face-down: %r' \
        % [str(c)[-34:] for c in fd_sh]
    assert c_naz not in fu and c_den not in fu, \
        'payment pair leaked into the FACE-UP out history'
    old_led = rx._LEDGER
    rx._LEDGER = led
    try:
        st = gs.setup_initial_game(board.load_board())
        rx._install_ledger_discards(st, 10 ** 9)
        fd_st = st.hands[Side.SHADOW].discarded_face_down
        assert c_naz in fd_st and c_den in fd_st, \
            'installer did not carry the pair into Shadow ' \
            'discarded_face_down: %r' % [str(c)[-30:] for c in fd_st]
        pub = [c for pile in st.discard_face_up.values() for c in pile]
        assert c_naz not in pub and c_den not in pub, \
            'installer put the masked pair into the PUBLIC piles'
    finally:
        rx._LEDGER = old_led


def test_inst_emission_completeness_table():
    """INST: every extracted game reports the emitted-vs-opened pending
    table (the honest instrument the census had to reconstruct by
    hand)."""
    samples, line, _lf = _extract(G_TKILL)
    pt = line.get('pend_table')
    assert isinstance(pt, dict) and 'open' in pt and 'emit' in pt, \
        'pend_table missing from manifest line: %r' % (pt,)
    opened = pt['open']
    emitted = pt['emit']
    assert opened.get('combat_card', 0) >= 1, \
        'no combat_card pendings counted opened: %r' % opened
    assert emitted.get('combat_card', 0) <= opened.get('combat_card', 0), \
        'emitted > opened for combat_card: %r vs %r' % (emitted, opened)


def test_p3_n_leaders_split_variant_match():
    """P3 (census-proven class): the LA-8 per-count n_leaders menu
    variants must match by their EXPLICIT count - the src_ldr fallback
    computed eff_ldr=src for every variant, multi-matched every
    full-leader partial-unit move out of a multi-leader stack and
    ambiguity-dropped it.  Synthetic menu, no game replay needed."""
    from game_state import Side
    import game_state as gs
    import board
    st = gs.setup_initial_game(board.load_board())
    # find a Shadow army key with units; give it 3 plain leaders
    key = next(k for k, a in st.armies.items()
               if any(u.nation.name in ('SAURON', 'ISENGARD',
                                        'SOUTHRON_EASTERLINGS')
                      for u in a.units))
    import rendered_replay as rr
    frm = rr.sanitize(key)
    army = st.armies[key]
    import game_logic as gl
    army.leaders = [gl.Minion(nation=gs.Nation.SAURON, name='L%d' % i)
                    for i in range(3)]
    for l in army.leaders:
        l.immobile = False

    class _A(object):
        def __init__(self, params):
            self.action_type = 'army_move'
            self.params = params
            self.die_idx = -1
    to = 'somewhere'
    legal = [
        _A({'from': frm, 'to': to, 'n_regular': 2, 'n_elite': 0,
            'take_leaders': True}),                       # all 3 leaders
        _A({'from': frm, 'to': to, 'n_regular': 2, 'n_elite': 0,
            'take_leaders': True, 'n_leaders': 1}),
        _A({'from': frm, 'to': to, 'n_regular': 2, 'n_elite': 0,
            'take_leaders': True, 'n_leaders': 2}),
    ]
    comp = {'reg': 2, 'elite': 0, 'leaders': 3, 'nazgul': 0}
    got = rx._match_army_move(legal, st, Side.SHADOW, frm, to, comp)
    assert got == 0, \
        'full-leader move matched %r (want 0: the take_leaders action; ' \
        'n_leaders variants must not shadow it)' % (got,)
    comp2 = {'reg': 2, 'elite': 0, 'leaders': 2, 'nazgul': 0}
    got2 = rx._match_army_move(legal, st, Side.SHADOW, frm, to, comp2)
    assert got2 == 2, \
        'partial 2-leader move matched %r (want 2: the n_leaders=2 ' \
        'variant)' % (got2,)


def test_p4_we_prove_the_swifter_title_resolves():
    """Card-cover census 2026-08-31: the VASSAL piece title 'We Prove
    the Swifter' (interior 'the' the registry name lacks) resolved in NO
    game - all 2,569 corpus plays of the card dropped, so the card had
    never trained as an event play or combat card.  The interior-variant
    row in ai._lookup_card_id's spelling table is the fix."""
    import rendered_extract as rx
    for t in ('We Prove the Swifter/Daring Defiance',
              'We Prove the Swifter',
              'We Prove Swifter/Daring Defiance'):
        cid = rx._title_to_cid(t)
        assert cid == 'FREE_PEOPLES_Character_We_Prove_Swifter', \
            '%r -> %r (want FREE_PEOPLES_Character_We_Prove_Swifter)' \
            % (t, cid)


def main():
    tests = [
        test_p4_we_prove_the_swifter_title_resolves,
        test_p3_n_leaders_split_variant_match,
        test_p0a_discard_piles_light_card_status,
        test_p0b_table_kill_payment_emitted,
        test_p1a_effect_target_rows,
        test_p1a_named_target_multihot_unit,
        test_p1a_relentless_assault_park_magnitude,
        test_p1b_card_casualty_overkill_wipe,
        test_p2a_named_combat_card_injected,
        test_p2b_mordor_eye_hunts_drive,
        test_hidden_info_discards_stay_dark,
        test_1261_payment_pair_facedown_per_side,
        test_inst_emission_completeness_table,
    ]
    fails = 0
    for t in tests:
        try:
            t()
            print('PASS %s' % t.__name__)
        except AssertionError as e:
            fails += 1
            print('FAIL %s: %s' % (t.__name__, e))
        except Exception as e:
            fails += 1
            print('FAIL %s: %s: %s' % (t.__name__, type(e).__name__, e))
    print('TOTAL=%d FAIL=%d' % (len(tests), fails))
    if fails:
        sys.exit(1)


if __name__ == '__main__':
    main()
