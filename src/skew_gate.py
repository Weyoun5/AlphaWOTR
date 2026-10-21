# -*- coding: utf-8 -*-
"""Train/serve encoding-skew gate (v30 fix wave, 2026-07-26;
bucketed rework 2026-07-27).

WHY: the offline eval scores STORED candidate encodings, so any feature
computed differently at extraction time vs live play is structurally
invisible to every training metric — and lethal at play time.  Two
such skews shipped before this gate existed: the die-INDEX column
(extraction stages its die at idx 0; live indices vary) and the
pass-face column (extraction passes rode the staged die; live passes
are face-less).  The result was a live policy prior of 1.0 on 'pass'
at every root while training accuracy read 71%.

WHAT (2026-07-27 rework — the first full-store run false-positived on
rare decision variants): action encodings are bucketed by
(category, subtype) PAIR, and only buckets present on BOTH sides are
range-compared.  The original category-only bucketing lumped e.g.
plain passes with combat_continue/cease/siege_no_press rows, so a
3-game random rollout that never reached a multi-round battle read
"stored-variable / live-constant" on the subtype bits — coverage, not
skew.  Pair-bucketing keeps the real bug class detectable (the col-81
pass-face bug trips INSIDE the plain-pass bucket) while variant
absence just shrinks the shared-bucket set (reported as coverage).

Checks:
  ACTION side, per shared (category,subtype) bucket:
    A. live value outside the stored range          -> HARD
    B. column constant on one side, variable other  -> HARD
    bucket present on one side only                 -> coverage note
  STATE side:
    B. const-stored vs variable-live (or reverse)   -> HARD
       (the staged-dice class: dice cols were constant in the store
       until the v30 pool-swap fix)
    A. live range outside stored range              -> ADVISORY
       (random rollouts legally reach states human games never do —
       bound overflow / game-length coverage, not encoding skew)

Exit 0 = no HARD findings (advisories printed); exit 1 = HARD findings.

Usage:  python src/skew_gate.py [--games 8] [--max-turns 10] [--ds DIR]
"""
import argparse
import os
import random
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

import numpy as np

TOL = 1e-3
N_TYPE_COLS = 14
SUBTYPE_LO = 34


def _subtype_hi():
    """Derived from ai (the hardcoded 56 went stale when v32 appended
    'hobbit_place' — the block is [34:34+len(_DECISION_SUBTYPES)])."""
    import ai
    return SUBTYPE_LO + len(ai._DECISION_SUBTYPES)
# Action-side B: a column must vary SYSTEMATICALLY in the stored bucket
# (col-81 class varied on ~100% of pass rows) to be skew; a column
# exercised by a minority variant (declare_placement regions inside the
# fellowship_move bucket: ~6%) is rollout coverage -> advisory.
SYSTEMATIC_FRAC = 0.30
MIN_LIVE_BUCKET = 200

# STATE-side stored-constant/live-varies allowlist (reviewed 2026-07-27,
# full-store run + per-column probes; valid ONLY for state dim 3742 —
# the gate hard-errors on any other dim so a new encoding generation
# forces re-review).  Two certified-benign classes:
#   occupancy: per-region occupancy/leader marker cols for regions no
#     human game garrisons (forlindon/harlindon/khand class) — random
#     rollouts reach them, 9,768 human games never do.
#   timing: pending/table-card exposure cols for contexts extraction
#     never emits a sample INSIDE (nazgul_armies followup, We Prove
#     Swifter / Grey Company / The Ring is Mine table windows) — the
#     net trains constant-0 there, so it is INSENSITIVE to (not misled
#     by) them; emission-context parity per family is a v32 item.
# The turn cols 3339/3419 are deliberately NOT listed: they must VARY
# in the store - if they ever read constant the gate aborts
# (regression tripwire).
STATE_ALLOWLIST_DIM = 3742
STATE_ALLOWLIST = {
    1110, 1111, 1112, 1113, 1114, 1115, 1120, 1121, 1124, 1125,
    1304, 1305, 1469, 2896,                 # occupancy class
    3251, 3254, 3300, 3507, 3513, 3719, 3721,   # timing class
    # deck-depletion features: live states
    # deplete real deck lists; extraction census states carry none, so
    # stored reads constant 0 until the ledger-driven deck-size
    # reconstruction lands (v32 item) — blindness, not inversion.
    3222, 3223, 3224, 3225,
    # corpus-coverage cells: legal board
    # states the post-fix live rollouts reach but no human game ever
    # produced — verified per-cell, same class as the occupancy set:
    #   1466 = khand: Sauron regular count (at-war walk via Harad chain)
    #   2897 = vale of the celduin: Dwarven elite count
    #   3530 = pending double-move FROM north ered luin (leg-2 one-hot)
    1466, 2897, 3530,
}


def live_rollout(n_games, max_turns=10, seed=99):
    """Random-legal-action games; collect (state vecs, action encs)."""
    import ai
    import board
    import game_logic
    from game_state import Side, setup_initial_game
    b = board.load_board()
    xs, encs = [], []
    for g in range(n_games):
        random.seed(seed + g)
        np.random.seed(seed + g)
        st = setup_initial_game(b)
        game_logic.initialize_decks(st)
        for turn in range(max_turns):
            game_logic.roll_action_dice(st, pool=Side.FREE_PEOPLES)
            game_logic.roll_action_dice(st, pool=Side.SHADOW)
            # a pending opened across the turn boundary (e.g. a SHADOW
            # hand-overflow discard from the draw phase) owns the first
            # decision — get_legal_actions hard-rejects wrong-side asks
            side = ai._pending_decision_side(st) or Side.FREE_PEOPLES
            for _step in range(60):
                acts = ai.get_legal_actions(st, side)
                if not acts:
                    break
                xs.append(ai.encode_state(st, side))
                for a in acts[:250]:
                    encs.append(ai.encode_action(a, st, side))
                a = random.choice(acts)
                try:
                    st = ai.apply_action(st, a)
                except Exception:
                    break
                res = game_logic.check_all_victory_conditions(st)
                if game_logic.should_game_end(res, end_of_turn=False):
                    break
                nxt = ai._next_acting_side(st, side)
                if nxt is None:
                    break
                side = nxt
            try:
                if ai._safe_next_turn(st):
                    break
            except Exception:
                break
    return (np.asarray(xs, dtype=np.float32),
            np.asarray(encs, dtype=np.float32))


def per_bucket_ranges(encs):
    """{(cat, subtype): (min_vec, max_vec, n)} — subtype = argmax of the
    [34:56] one-hot block, -1 when no subtype bit is set."""
    out = {}
    cat = np.argmax(encs[:, :N_TYPE_COLS], axis=1)
    has_cat = encs[:, :N_TYPE_COLS].max(axis=1) > 0.5
    sub_blk = encs[:, SUBTYPE_LO:_subtype_hi()]
    sub = np.where(sub_blk.max(axis=1) > 0.5,
                   np.argmax(sub_blk, axis=1), -1)
    for c in range(N_TYPE_COLS):
        for s in sorted(set(sub[(cat == c) & has_cat].tolist())):
            sel = encs[has_cat & (cat == c) & (sub == s)]
            if len(sel):
                out[(c, int(s))] = (sel.min(axis=0), sel.max(axis=0),
                                    len(sel))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=8)
    ap.add_argument('--max-turns', type=int, default=10)
    ap.add_argument('--ds', default=os.path.join(
        # shipped-generation default
        ROOT, 'data', 'bc_corpus_v37_rendered_ds'))
    ap.add_argument('--max-stored', type=int, default=2_000_000)
    args = ap.parse_args()

    import ai
    print('[skew-gate] live rollout (%d games x %d turns)...'
          % (args.games, args.max_turns), flush=True)
    lx, le = live_rollout(args.games, max_turns=args.max_turns)
    print('[skew-gate] live: %d states, %d action encs'
          % (len(lx), len(le)), flush=True)

    hard, advis = [], []

    # ---- stored side ----
    s_cands = []
    s_x_min = s_x_max = None
    for tag in ('fp', 'sh'):
        cp = os.path.join(args.ds, '%s_cands.npy' % tag)
        xp = os.path.join(args.ds, '%s_X.npy' % tag)
        if not os.path.exists(cp):
            sys.exit('[skew-gate] missing %s — consolidate first' % cp)
        c = np.load(cp, mmap_mode='r')
        step = max(1, len(c) // (args.max_stored // 2))
        s_cands.append(np.asarray(c[::step], dtype=np.float32))
        X = np.load(xp, mmap_mode='r')
        xstep = max(1, len(X) // 200_000)
        xs = np.asarray(X[::xstep], dtype=np.float32)
        mn, mx = xs.min(axis=0), xs.max(axis=0)
        s_x_min = mn if s_x_min is None else np.minimum(s_x_min, mn)
        s_x_max = mx if s_x_max is None else np.maximum(s_x_max, mx)
    sc = np.concatenate(s_cands)
    print('[skew-gate] stored sample: %d cand rows' % len(sc), flush=True)

    # ACTION-dim generation guard (mirrors the STATE-side hard error):
    # a mismatched-generation cand store mis-slices bucket identity and
    # silently skips live tail columns
    import ai as _ai_dim9
    if sc.shape[1] != _ai_dim9.ACTION_ENCODING_DIM or \
            (len(le) and le.shape[1] != _ai_dim9.ACTION_ENCODING_DIM):
        sys.exit('[skew-gate] action dim mismatch: stored=%d live=%s '
                 'engine=%d — re-extract before trusting the gate'
                 % (sc.shape[1],
                    le.shape[1] if len(le) else '?',
                    _ai_dim9.ACTION_ENCODING_DIM))

    # ---- ACTION side: shared (category,subtype) buckets ----
    live_r = per_bucket_ranges(le)
    stored_r = per_bucket_ranges(sc)
    from ai import ACTION_TYPE_CATEGORIES as CATS

    def bname(key):
        c, s = key
        return '%s/sub%d' % (CATS[c] if 0 <= c < len(CATS) else c, s)

    only_stored = sorted(set(stored_r) - set(live_r))
    only_live = sorted(set(live_r) - set(stored_r))
    if only_stored:
        advis.append('coverage: %d stored-only buckets not reached by '
                     'the rollout: %s'
                     % (len(only_stored),
                        ', '.join(bname(k) for k in only_stored[:12])))
    if only_live:
        advis.append('coverage: %d live-only buckets absent from the '
                     'store (net never trained on them): %s'
                     % (len(only_live),
                        ', '.join(bname(k) for k in only_live[:12])))

    # per-bucket stored row samples for the systematic-variability test
    s_cat = np.argmax(sc[:, :N_TYPE_COLS], axis=1)
    s_blk = sc[:, SUBTYPE_LO:_subtype_hi()]
    s_sub = np.where(s_blk.max(axis=1) > 0.5, np.argmax(s_blk, axis=1), -1)

    def nonmodal_frac(key, col):
        """Fraction of stored bucket rows NOT at the column's modal
        value (sampled).  ~1.0 for the col-81 class; ~0.05 for a
        minority-variant column."""
        c, s = key
        sel = np.flatnonzero((s_cat == c) & (s_sub == s))
        if len(sel) > 100_000:
            sel = sel[:: len(sel) // 100_000 + 1]
        vals = np.round(sc[sel, col].astype(np.float64), 4)
        _, counts = np.unique(vals, return_counts=True)
        return 1.0 - counts.max() / float(len(vals))

    for key in sorted(set(live_r) & set(stored_r)):
        lmin, lmax, ln = live_r[key]
        smin, smax, sn = stored_r[key]
        if ln < MIN_LIVE_BUCKET:
            advis.append('coverage: bucket %s live n=%d < %d — ranges '
                         'not compared' % (bname(key), ln,
                                           MIN_LIVE_BUCKET))
            continue
        for col in range(sc.shape[1]):
            if col < N_TYPE_COLS or SUBTYPE_LO <= col < _subtype_hi():
                continue      # bucket identity columns
            a_out = (lmax[col] > smax[col] + TOL
                     or lmin[col] < smin[col] - TOL)
            l_const = (lmax[col] - lmin[col]) < TOL
            s_const = (smax[col] - smin[col]) < TOL
            b_mismatch = (l_const != s_const) and (
                max(smax[col] - smin[col], lmax[col] - lmin[col]) > 0.05)
            if a_out:
                msg = ('A %-24s col %3d: live [%.3f,%.3f] outside '
                       'stored [%.3f,%.3f] (n=%d/%d)'
                       % (bname(key), col, lmin[col], lmax[col],
                          smin[col], smax[col], ln, sn))
                # OOD on a trained-informative input = HARD; live
                # values on a stored-CONSTANT col = the net trained on
                # a constant and ignores the input (blindness class,
                # e.g. discard_table_card nested-choice cols) = advisory
                if s_const:
                    advis.append(msg + ' [stored-constant]')
                else:
                    hard.append(msg)
            elif b_mismatch:
                msg = ('B %-24s col %3d: const-live=%s const-stored=%s '
                       'live [%.3f,%.3f] stored [%.3f,%.3f] (n=%d/%d)'
                       % (bname(key), col, l_const, s_const,
                          lmin[col], lmax[col], smin[col], smax[col],
                          ln, sn))
                # stored-varies/live-const is skew ONLY when the column
                # varies systematically for this decision type; a
                # minority-variant column is rollout coverage
                if s_const or nonmodal_frac(key, col) >= SYSTEMATIC_FRAC:
                    hard.append(msg)
                else:
                    advis.append(msg + ' [minority-variant]')

    # ---- STATE side ----
    lmn, lmx = lx.min(axis=0), lx.max(axis=0)
    # B (HARD): stored-constant but live-variable.  Reviewed-benign
    # cols are allowlisted ABOVE, valid only for the dim they were
    # reviewed under.
    if len(s_x_min) != STATE_ALLOWLIST_DIM:
        sys.exit('[skew-gate] state dim %d != %d — the state allowlist '
                 'was reviewed for 3742; re-review before trusting it'
                 % (len(s_x_min), STATE_ALLOWLIST_DIM))
    s_const_v = (s_x_max - s_x_min) < TOL
    l_const_v = (lmx - lmn) < TOL
    b_cols = np.flatnonzero(s_const_v & ~l_const_v
                            & ((lmx - lmn) > 0.05))
    for col in b_cols[:60]:
        msg = ('B STATE                  col %4d: stored CONSTANT %.3f '
               'but live varies [%.3f,%.3f]' % (col, s_x_min[col],
                                                lmn[col], lmx[col]))
        if int(col) in STATE_ALLOWLIST:
            advis.append(msg + ' [allowlisted]')
        else:
            hard.append(msg)
    # A (ADVISORY): live range outside stored range — random rollouts
    # legally reach states human games never do (bound overflow,
    # game-length coverage); distribution shift, not encoding skew.
    a_cols = np.flatnonzero(((lmx > s_x_max + TOL)
                             | (lmn < s_x_min - TOL))
                            & ~(s_const_v & ~l_const_v))
    for col in a_cols[:40]:
        advis.append(
            'A STATE                  col %4d: live [%.3f,%.3f] vs '
            'stored [%.3f,%.3f]' % (col, lmn[col], lmx[col],
                                    s_x_min[col], s_x_max[col]))

    if advis:
        print('\n[skew-gate] %d advisories (coverage/distribution, '
              'not skew):' % len(advis))
        for f in advis:
            print('  ' + f)
    if hard:
        print('\n[skew-gate] %d HARD FINDINGS:' % len(hard))
        for f in hard:
            print('  ' + f)
        print('[skew-gate] FAIL — review before trusting self-play')
        return 1
    print('[skew-gate] PASS — no train/serve encoding skew detected')
    return 0


if __name__ == '__main__':
    sys.exit(main())
