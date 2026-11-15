"""Corpus BC training, v30 generation: FULL-MENU softmax CE.

Until v29 the policy was trained as a RANKER: human choice in slot 0
plus K-1 RANDOM negatives (K=16, then 32) — while live play scores the
full legal menu (mean ~166 actions, up to ~1500).  That train/serve gap
meant the training accuracy metric structurally could not see live
misbehavior (2026-07-26 forensics).  v30 trains and measures on the
SAME distribution the net plays under:

Stage 1 (consolidate): streams data/bc_corpus_*/<game>.npz into per-side
RAGGED memmaps — every sample keeps its FULL candidate menu (fp16
storage), offsets index the flat candidate arrays, and a per-candidate
POSITIVE flag marks the human choice plus its byte-identical duplicates
(duplicate-tolerant training, same rule the eval always used).  Menus
beyond CAND_MENU_CAP (default 512) keep all positives and subsample
negatives; capped counts are PRINTED, never silent.  Split is BY GAME
(seeded 5% holdout), unchanged.

Stage 2 (train): candidate-BUDGET batching (CAND_BUDGET candidates per
step, default 49152) over the ragged store; loss =
multi-positive listwise CE  [ -log(sum_{j in pos} p_j) ]
+ 0.5*MSE(value, z), VALUE_CTX_DROPOUT preserved.

Stage 3 (eval): unchanged — duplicate-tolerant top-1 agreement against
the FULL candidate sets from the per-game npz, train-sample + ALL
held-out games.  Train and eval now measure the same thing.

Usage:
  python src/train_corpus.py --stage consolidate
  python src/train_corpus.py --stage train --epochs 10
  python src/train_corpus.py --stage eval
  python src/train_corpus.py --stage all
  python src/train_corpus.py --stage inspect-contracts  # read-only legacy/current census
"""
import argparse
import contextlib
import json
import math
import os
import re
import sys
import time

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# Defaults must track the SHIPPED generation (WIRING 2026-08-11
# doctrine): an env-less run must never build a self-consistent
# WRONG-generation pipeline.  The current generation is v41
# (WOTR_ACT_GEN=v41; corpus_live nets). Set CORPUS_DIR / CORPUS_DS_DIR /
# CORPUS_NET_TAG explicitly for a fresh extraction. Historical directories
# can exist with the same dimensions; the row and dataset identity checks
# below refuse to train them under the current observation/menu semantics.
CORPUS = os.environ.get('CORPUS_DIR') or os.path.join(
    ROOT, 'data', 'bc_corpus_v37_rendered')
DS_DIR = os.environ.get('CORPUS_DS_DIR') or os.path.join(
    ROOT, 'data', 'bc_corpus_v37_rendered_ds')
SPLIT_JSON = os.path.join(DS_DIR, 'split.json')
SEED = 20260706
MENU_CAP = int(os.environ.get('CAND_MENU_CAP', '512'))
CAND_BUDGET = int(os.environ.get('CAND_BUDGET', '49152'))
NET_TAG = os.environ.get('CORPUS_NET_TAG', 'v37')
# v32 extended-training calibration curve (USER 2026-08-03): save a
# checkpoint snapshot every K epochs so declare-mass calibration and
# holdout agreement can be probed per snapshot (early-stop / plateau
# verdict).  0 = off.
SNAP_EVERY = int(os.environ.get('CORPUS_SNAPSHOT_EVERY', '0'))


def game_files():
    return sorted(f for f in os.listdir(CORPUS) if f.endswith('.npz'))


def _current_learning_contract():
    from trace_decisions import current_learning_context
    return current_learning_context()


def _require_learning_contract(value, source):
    expected = _current_learning_contract()
    if not isinstance(value, dict):
        raise ValueError(f'{source}: missing learning_contract; re-extract and consolidate into a fresh dataset')
    for key, want in expected.items():
        if value.get(key) != want:
            raise ValueError(f'{source}: {key}={value.get(key)!r}, expected {want!r}; '
                             'legacy arrays/checkpoints are not current learning inputs')
    return dict(expected)


def _require_dataset_contract(split=None):
    if split is None:
        with open(SPLIT_JSON, encoding='utf-8') as stream:
            split = json.load(stream)
    _require_learning_contract(split.get('learning_contract'), SPLIT_JSON)
    if split.get('policy_target_projection') is not None:
        from consolidation_provenance import verify
        verify(DS_DIR, split['policy_target_projection'])
    return split


def _checkpoint_contract():
    """Only stamp outputs from a dataset whose observation/menu identity passed."""
    return dict(_require_dataset_contract()['learning_contract'])


def _npz_contract_report(data, source):
    """Inspect row identities without relabeling legacy or mixed arrays."""
    report = {'file': source, 'rows': 0, 'errors': [], 'fields': {}}
    if 'X' not in data.files:
        report['empty'] = True
        return report
    report['rows'] = n = len(data['X'])
    expected = _current_learning_contract()
    required = {'observation_contract': expected['observation_contract'],
                'action_encoding_contract': expected['action_encoding_contract'],
                'ai_policy_version': expected['ai_policy_contract'],
                'decision_phase': None}
    for key, want in required.items():
        if key not in data.files:
            report['errors'].append(f'{source}: missing {key}')
            continue
        values = data[key]
        if values.shape != (n,) or values.dtype.kind not in ('U', 'S'):
            report['errors'].append(f'{source}: {key} must be a string array with shape ({n},), '
                                    f'got {values.shape}/{values.dtype}')
            continue
        values = values.astype(str)
        names, counts = np.unique(values, return_counts=True)
        report['fields'][key] = {str(k): int(v) for k, v in zip(names, counts)}
        invalid = (values != want) if want is not None else (np.char.strip(values) == '')
        if invalid.any():
            rows = np.flatnonzero(invalid)[:8].tolist()
            detail = (f'{key} has incompatible rows {rows}; expected {want!r}'
                      if want is not None else f'decision_phase is empty at rows {rows}')
            report['errors'].append(f'{source}: {detail}')
    if 'policy_target_projection' in data.files:
        from consolidation_provenance import read_rows
        try:
            report['projection_rows'] = sum(row is not None for row in read_rows(data, source))
        except (ValueError, TypeError, KeyError, IndexError) as error:
            report['errors'].append(str(error))
    return report


def _require_npz_contract(data, source):
    report = _npz_contract_report(data, source)
    if report['errors']:
        raise ValueError('; '.join(report['errors']))
    return report


def inspect_contracts(names=None):
    """Read-only legacy/current metadata census; this never authorizes training."""
    records = []
    for name in game_files() if names is None else names:
        path = os.path.join(CORPUS, name)
        try:
            with np.load(path, allow_pickle=False) as data:
                records.append(_npz_contract_report(data, path))
        except Exception as exc:
            records.append({'file': path, 'rows': 0, 'fields': {},
                            'errors': [f'{path}: unreadable metadata: {exc}']})
    invalid = sum(bool(r['errors']) for r in records)
    rows = sum(r['rows'] for r in records)
    return {'learning_contract': _current_learning_contract(), 'files': records,
            'rows': rows, 'invalid_files': invalid,
            'accepted_for_current_learning': rows > 0 and invalid == 0}


# SPTR-17: unreadable (corrupt/truncated) npz files are DISTINCT from
# empty ones — recorded here and printed loudly by consolidate();
# silently conflating the two hides data loss.
_UNREADABLE = []


def is_empty_npz(path):
    try:
        with np.load(path) as z:
            return 'X' not in z.files
    except Exception as e:
        _UNREADABLE.append((path, repr(e)))
        return True


def _dup_key(enc, reg):
    return (np.asarray(enc, dtype=np.float32).tobytes(), int(reg))


# columns in the action encoding (see ai.py encode_action)
_COL_DIE_IDX = 22       # arbitrary pool position — dead since the skew fix
_COL_CAT_PASS = 10      # 'pass' category one-hot
_COL_CAT_DECLARE = 8    # 'fellowship_declare' category one-hot
# SPTR-10/SPCP-7: die-face/skip columns are RESOLVED FROM ai (the
# symbols are the truth) — the old hardcoded 81/82 went stale-by-one
# when the v32 hobbit_place subtype shifted the layout, silently
# killing the v30 pass-face strip and aiming it at the card-option
# column instead.  Resolved lazily (ai is heavy; train_corpus stays
# importable without it) and cached in the module globals.
_COL_SKIP = None        # skip flag (skips legitimately carry a face)
_COL_DIEFACE = None     # acting-die face


def _act_cols():
    """(die-face col, skip col) from ai — single source (SPTR-10)."""
    global _COL_DIEFACE, _COL_SKIP
    if _COL_DIEFACE is None:
        import ai
        _COL_DIEFACE = int(ai._ACT_DIEFACE_OFF)
        _COL_SKIP = int(ai._ACT_SKIP_OFF)
        # the category columns are position-stable by design; verify
        # against ai anyway (same stale-constant class)
        assert ai._ACTION_TYPE_TO_IDX['pass'] == _COL_CAT_PASS
        assert ai._ACTION_TYPE_TO_IDX['fellowship_declare'] == \
            _COL_CAT_DECLARE
    return _COL_DIEFACE, _COL_SKIP

# v32: {tag}_dw.npy flags declare-window samples (menus holding a
# fellowship_declare candidate).  REPORTING ONLY (USER 2026-08-03): no
# declare-window CE weight - with gate-(e) 100% per-turn coverage the
# sample distribution IS the human conditional, and the net must learn
# it from the data unweighted.  If the unweighted net fails the by-turn
# calibration gate, that is a capacity/architecture FINDING to report,
# never something to patch with weights.


def _normalize_cands(arr):
    """v30 train/serve skew fix (2026-07-26): canonicalize STORED
    candidate encodings to match the live encoder — (1) die-INDEX
    column zeroed (extraction stages its die at idx 0; live indices
    vary), (2) pass-category rows that are not skips lose their die
    face (extraction passes rode the staged die; live passes are
    face-less).  Applied at consolidation AND eval so training,
    measurement, and play share one encoding.  Mutates + returns arr
    (float array)."""
    _dieface9, _skip9 = _act_cols()
    arr[:, _COL_DIE_IDX] = 0.0
    _pmask = (arr[:, _COL_CAT_PASS] > 0.5) & (arr[:, _skip9] < 0.5)
    arr[_pmask, _dieface9] = 0.0
    return arr


def _label_roundtrip_fail(row):
    """SPTR-7 (honest half): layout-level round-trip of one STORED
    human-label encoding against TODAY's encode_action layout.  States
    are not stored, so menu membership CANNOT be recomputed here —
    re-extraction under the current engine is the real fix; this stat
    catches generation/layout drift loudly instead of silently.
    Returns a reason string, or None when the row is layout-valid."""
    import ai
    row = np.asarray(row)
    if row.shape[-1] != ai.ACTION_ENCODING_DIM:
        return ('dim %d != engine %d'
                % (row.shape[-1], ai.ACTION_ENCODING_DIM))
    ncat = int((row[:len(ai.ACTION_TYPE_CATEGORIES)] > 0.5).sum())
    if ncat != 1:
        return 'category bits %d != 1' % ncat
    _sub_lo9 = ai._ACT_COMP_OFF - len(ai._DECISION_SUBTYPES)
    sub = row[_sub_lo9:ai._ACT_COMP_OFF]
    if int((sub > 0.5).sum()) > 1:
        return 'multiple subtype bits'
    return None


# ------------------------------------------------------------------
# Stage 1: consolidation (ragged full-menu layout)
# ------------------------------------------------------------------

def _menu_rows(menu, regs_slice, h, rng, alts=()):
    """Kept candidate row indices (into the menu slice) + local positive
    flags + the human's new local index.  `menu` must already be
    NORMALIZED (so die-idx-only twins register as duplicates).  Applies
    MENU_CAP: all positives kept, negatives subsampled."""
    k = len(menu)
    enc_h = menu[h]
    pos = np.logical_and(
        (menu == enc_h).all(axis=1),
        regs_slice == regs_slice[h])
    # 1.268 (COORD #2 2026-09-03): DECLARED-equivalent candidates (the
    # npz alt_idx of a ring take whose K enabling faces are all right)
    # join the positive set exactly like encoding twins - the listwise
    # multi-positive CE below trains -log(sum_pos p).  Never a weight.
    # S-002 (sponsor 2026-09-04): `alts` arrives as the npz alt_idx SLICE
    # (a numpy array, empty for every non-ring row) - never ask numpy for
    # its truth value (ValueError under numpy 2.2: killed both v42s chains
    # in stage consolidate); iterate the flattened ints instead.
    _alts9 = (np.asarray(alts, dtype=np.int64).ravel().tolist()
              if alts is not None else [])
    for _a in _alts9:
        _a = int(_a)
        if 0 <= _a < k:
            pos = np.logical_or(pos, np.logical_and(
                (menu == menu[_a]).all(axis=1),
                regs_slice == regs_slice[_a]))
    if k <= MENU_CAP:
        keep = np.arange(k)
    else:
        pos_idx = np.flatnonzero(pos)
        neg_idx = np.flatnonzero(~pos)
        n_neg = MENU_CAP - len(pos_idx)
        if n_neg < 1:
            # a menu with MORE positives than the cap keeps the human
            # plus the first MENU_CAP-1 other positives (all equivalent
            # dups) - never more than the pass-1 reserved
            # min(k, MENU_CAP) slots.
            others = pos_idx[pos_idx != h][:MENU_CAP - 1]
            keep = np.sort(np.concatenate([[h], others]))
        else:
            sel = rng.choice(len(neg_idx), n_neg, replace=False)
            keep = np.sort(np.concatenate([pos_idx, neg_idx[sel]]))
    new_h = int(np.searchsorted(keep, h))
    return keep, pos[keep], new_h


def _check_phi_provenance():
    """HUNT-A F7 consumer (WIRING 2026-08-11): the extraction stamp
    CORPUS\\phi_profiles.json records the regime every sample's phi
    was computed under.  Before this check NOTHING read the stamp —
    'consumers must check this stamp, never assume' was decorative,
    and the one env knob that can silently skew the currency
    (WOTR_PHI_RINGS_RHO) was absent from it.  Consolidation refuses a
    corpus whose stamp is missing, whose profile pair differs from
    the pinned fp=ring / sh=military, whose recorded rho != 0, or
    whose recorded PROFILE_WEIGHT differs from the engine's."""
    p = os.path.join(CORPUS, 'phi_profiles.json')
    if not os.path.exists(p):
        sys.exit('phi provenance stamp missing: %s — phi currency '
                 'unverifiable; re-extract (the stamp is written by '
                 'rendered_extract.run_driver)' % p)
    with open(p, encoding='utf-8') as stream:
        st = json.load(stream)
    if (st.get('fp'), st.get('sh')) != ('ring', 'military'):
        sys.exit('phi_profiles.json profile pair %r != pinned '
                 "('ring', 'military') — the corpus phi was computed "
                 'under a different regime; re-extract'
                 % ((st.get('fp'), st.get('sh')),))
    if 'rho' in st and float(st['rho']) != 0.0:
        sys.exit('phi_profiles.json rho=%r != 0 — WOTR_PHI_RINGS_RHO '
                 'must be UNSET during extraction (D-F2 struck); '
                 're-extract' % (st['rho'],))
    if 'profile_weight' in st:
        import ai
        if abs(float(st['profile_weight'])
               - float(ai.PROFILE_WEIGHT)) > 1e-12:
            sys.exit('phi_profiles.json profile_weight=%r != '
                     'ai.PROFILE_WEIGHT=%r — corpus phi scale differs '
                     'from the engine; re-extract'
                     % (st['profile_weight'], ai.PROFILE_WEIGHT))
    _missing9 = [k for k in ('rho', 'profile_weight') if k not in st]
    if _missing9:
        print('phi_profiles.json predates the %s stamp field(s) '
              '(pre-2026-08-11 extraction): those values are '
              'UNVERIFIABLE for this corpus; profile pair verified'
              % '/'.join(_missing9), flush=True)


def consolidate():
    from consolidation_provenance import read_rows, record, publish
    # Check the entire population, including holdout, before any deletion or
    # memmap creation. Same dimensions do not imply the same observations/menu.
    files = game_files()
    admission = inspect_contracts(files)
    if admission['invalid_files']:
        first = next(r for r in admission['files'] if r['errors'])
        raise ValueError('; '.join(first['errors']))
    if not admission['accepted_for_current_learning']:
        raise ValueError('No nonempty versioned corpus rows to consolidate')
    _check_phi_provenance()
    os.makedirs(DS_DIR, exist_ok=True)
    # An interrupted rebuild must not leave a completed manifest over partial
    # memmaps. It is recreated only after every array has been populated.
    if os.path.exists(SPLIT_JSON):
        os.remove(SPLIT_JSON)
    # HUNT-A F3: a re-consolidate invalidates any derived ztd —
    # leaving one behind lets a later jt train on targets computed
    # from the PREVIOUS dataset
    for _tag9 in ('fp', 'sh'):
        for _f9 in (f'{_tag9}_ztd.npy', f'{_tag9}_ztd_meta.json'):
            _p9 = os.path.join(DS_DIR, _f9)
            if os.path.exists(_p9):
                os.remove(_p9)
                print(f'  invalidated stale {_f9}', flush=True)
    import ai
    rng = np.random.RandomState(SEED)
    usable = [f for f in files if not is_empty_npz(os.path.join(CORPUS, f))]
    if _UNREADABLE:
        # SPTR-17: corrupt files counted separately from empty ones
        print('LOUD: %d npz files UNREADABLE (corrupt/truncated), '
              'dropped from the population - first: %r'
              % (len(_UNREADABLE), _UNREADABLE[0]), flush=True)
    rng.shuffle(usable)
    n_hold = max(1, int(0.05 * len(usable)))
    hold = sorted(usable[:n_hold])
    train_g = sorted(usable[n_hold:])
    print(f"usable games: {len(usable)} -> train {len(train_g)}, "
          f"held-out {len(hold)}", flush=True)

    # pass 1: per-side sample + candidate counts (post-cap)
    counts = {0: [0, 0], 1: [0, 0]}     # side -> [n_samples, n_cands]
    capped = 0
    D = A = None
    for i, f in enumerate(train_g):
        with np.load(os.path.join(CORPUS, f)) as z:
            _require_npz_contract(z, f)
            side, offs = z['side'], z['offsets']
            if D is None:
                D = z['X'].shape[1]
                A = z['cand_encs'].shape[1]
            for si in range(len(side)):
                k = int(offs[si + 1] - offs[si])
                if k > MENU_CAP:
                    capped += 1
                    k = MENU_CAP
                c = counts[int(side[si])]
                c[0] += 1
                c[1] += k
        if i % 2000 == 0:
            print(f"  count pass {i}/{len(train_g)}", flush=True)
    print(f"train samples: fp={counts[0][0]} (cands {counts[0][1]}) "
          f"sh={counts[1][0]} (cands {counts[1][1]}); "
          f"menus capped at {MENU_CAP}: {capped}", flush=True)
    if A is not None and A != ai.ACTION_ENCODING_DIM:
        # SPTR-7: a cross-generation store must never consolidate
        # silently (skew_gate would catch it later; loud NOW)
        sys.exit('LOUD: store action dim %d != engine %d '
                 '(WOTR_ACT_GEN mismatch) - re-extract under the '
                 'current generation' % (A, ai.ACTION_ENCODING_DIM))

    mm = {}
    for sc, tag in ((0, 'fp'), (1, 'sh')):
        n, m = counts[sc]
        mm[sc] = {
            'X': np.lib.format.open_memmap(
                os.path.join(DS_DIR, f'{tag}_X.npy'), mode='w+',
                dtype=np.float32, shape=(n, D)),
            'z': np.lib.format.open_memmap(
                os.path.join(DS_DIR, f'{tag}_z.npy'), mode='w+',
                dtype=np.float32, shape=(n,)),
            # v37 currency coherence: per-sample side-signed phi,
            # aligned with {tag}_X.npy (residual value targets)
            'phi': np.lib.format.open_memmap(
                os.path.join(DS_DIR, f'{tag}_phi.npy'), mode='w+',
                dtype=np.float32, shape=(n,)),
            'human': np.lib.format.open_memmap(
                os.path.join(DS_DIR, f'{tag}_human.npy'), mode='w+',
                dtype=np.int32, shape=(n,)),
            'off': np.lib.format.open_memmap(
                os.path.join(DS_DIR, f'{tag}_off.npy'), mode='w+',
                dtype=np.int64, shape=(n + 1,)),
            'cands': np.lib.format.open_memmap(
                os.path.join(DS_DIR, f'{tag}_cands.npy'), mode='w+',
                dtype=np.float16, shape=(m, A)),
            'regs': np.lib.format.open_memmap(
                os.path.join(DS_DIR, f'{tag}_regs.npy'), mode='w+',
                dtype=np.int32, shape=(m,)),
            'pos': np.lib.format.open_memmap(
                os.path.join(DS_DIR, f'{tag}_pos.npy'), mode='w+',
                dtype=np.uint8, shape=(m,)),
            # v32: per-sample declare-window flag (menu holds a
            # fellowship_declare candidate) for the CE weighting
            'dw': np.lib.format.open_memmap(
                os.path.join(DS_DIR, f'{tag}_dw.npy'), mode='w+',
                dtype=np.uint8, shape=(n,)),
            'n': 0, 'm': 0,
        }
        mm[sc]['off'][0] = 0

    # deterministic per-sample rng for the negative subsampling
    cap_rng = np.random.RandomState(SEED + 1)
    projection_lines = []
    # SPTR-7 loud stat: stored human labels checked against TODAY's
    # encode_action layout (menu membership needs states, which the
    # npz does not hold - re-extraction is the real fix)
    _lrt_fail9 = 0
    _lrt_first9 = None
    t0 = time.time()
    for i, f in enumerate(train_g):
        with np.load(os.path.join(CORPUS, f)) as zf:
            _require_npz_contract(zf, f)
            projection_rows = read_rows(zf, f)
            X, side = zf['X'], zf['side']
            human, z, offs = zf['human'], zf['z'], zf['offsets']
            # v37 schema: phi REQUIRED (KeyError on a pre-v37 npz is
            # the intended loud failure — re-extract, never default)
            phi_g = zf['phi']
            ce, cr = zf['cand_encs'], zf['cand_regs']
            # 1.268: optional alternate positives (absent on older npz)
            _alt_i = zf['alt_idx'] if 'alt_idx' in zf.files else None
            _alt_o = zf['alt_offsets'] if 'alt_offsets' in zf.files else None
            for si in range(len(X)):
                d = mm[int(side[si])]
                lo, hi = int(offs[si]), int(offs[si + 1])
                _why9 = _label_roundtrip_fail(ce[lo + int(human[si])])
                if _why9 is not None:
                    _lrt_fail9 += 1
                    if _lrt_first9 is None:
                        _lrt_first9 = '%s sample %d: %s' % (f, si, _why9)
                menu = _normalize_cands(ce[lo:hi].astype(np.float32))
                keep, posf, new_h = _menu_rows(
                    menu, cr[lo:hi], int(human[si]), cap_rng,
                    alts=(_alt_i[int(_alt_o[si]):int(_alt_o[si + 1])]
                          if _alt_i is not None else ()))
                n, m = d['n'], d['m']
                provenance = record(f, si, side[si], n, projection_rows[si],
                                    hi - lo, int(human[si]), keep, new_h)
                if provenance is not None:
                    projection_lines.append(provenance)
                k = len(keep)
                d['X'][n] = X[si]
                d['z'][n] = float(z[si])
                d['phi'][n] = float(phi_g[si])
                d['human'][n] = new_h
                d['dw'][n] = np.uint8(
                    (menu[:, _COL_CAT_DECLARE] > 0.5).any())
                d['cands'][m:m + k] = menu[keep].astype(np.float16)
                d['regs'][m:m + k] = cr[lo:hi][keep]
                d['pos'][m:m + k] = posf.astype(np.uint8)
                d['n'] = n + 1
                d['m'] = m + k
                d['off'][d['n']] = d['m']
        if i % 1000 == 0:
            print(f"  fill pass {i}/{len(train_g)} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    for sc in (0, 1):
        assert mm[sc]['n'] == counts[sc][0], (mm[sc]['n'], counts[sc][0])
        assert mm[sc]['m'] == counts[sc][1], (mm[sc]['m'], counts[sc][1])
        for v in mm[sc].values():
            if hasattr(v, 'flush'):
                v.flush()
    if _lrt_fail9:
        print('LOUD: %d stored human labels FAIL today\'s '
              'encode_action layout round-trip (first: %s) - the '
              'store predates the current engine; RE-EXTRACTION is '
              'the real fix (SPTR-7/SPCP-2).  Menu membership cannot '
              'be checked without states.' % (_lrt_fail9, _lrt_first9),
              flush=True)
    else:
        print('label layout round-trip: 0 failures (%d labels; menu '
              'membership still needs a re-extraction to verify)'
              % (mm[0]['n'] + mm[1]['n']), flush=True)
    # Holdout has no consolidated row, but its explicit label transformations
    # remain traceable to the unchanged per-game samples used for evaluation.
    for name in hold:
        with np.load(os.path.join(CORPUS, name), allow_pickle=False) as data:
            sides, offsets, human = data['side'], data['offsets'], data['human']
            for si, metadata in enumerate(read_rows(data, name)):
                provenance = record(name, si, sides[si], None, metadata,
                                    int(offsets[si + 1] - offsets[si]),
                                    int(human[si]), partition='holdout')
                if provenance is not None:
                    projection_lines.append(provenance)
    projection_binding = publish(DS_DIR, projection_lines)
    with open(SPLIT_JSON, 'w', encoding='utf-8') as f:
        json.dump({'train': train_g, 'holdout': hold,
                   'policy_target_projection': projection_binding,
                   'learning_contract': admission['learning_contract'],
                   'contract_rows_checked': admission['rows'],
                   'label_roundtrip_fails': _lrt_fail9,
                   'fp_n': counts[0][0], 'sh_n': counts[1][0],
                   'fp_m': counts[0][1], 'sh_m': counts[1][1],
                   'fp_declare_windows': int(mm[0]['dw'][:].sum()),
                   'sh_declare_windows': int(mm[1]['dw'][:].sum()),
                   'scheme': 'full_menu_v30', 'menu_cap': MENU_CAP,
                   'capped_menus': capped, 'seed': SEED}, f, indent=1)
    print(f"consolidated in {time.time() - t0:.0f}s; wrote {SPLIT_JSON}",
          flush=True)


# ------------------------------------------------------------------
# Stage 2: training (full-menu listwise CE)
# ------------------------------------------------------------------

def batched_scores(net, h, ne, cands, regs):
    """score_actions_spatial semantics over (B, K) padded candidates,
    returning raw logits (softmax deferred to the loss)."""
    import torch
    B, K, A = cands.shape
    E = ne.shape[-1]
    mean_emb = ne.mean(dim=1)                                # (B, E)
    safe = regs.clamp(min=0).long()                          # (B, K)
    reg_emb = torch.gather(
        ne, 1, safe.unsqueeze(-1).expand(B, K, E))           # (B, K, E)
    nonreg = (regs < 0).unsqueeze(-1)
    reg_emb = torch.where(nonreg, mean_emb.unsqueeze(1).expand(B, K, E),
                          reg_emb)
    feats, card_emb = net._embed_action_encs(cands.reshape(B * K, A))
    combined = torch.cat(
        [h.unsqueeze(1).expand(B, K, h.shape[-1]).reshape(B * K, -1),
         reg_emb.reshape(B * K, E), feats, card_emb], dim=1)
    return net.spatial_action_scorer(combined).squeeze(-1).reshape(B, K)


def _budget_batches(order, off, budget, b_max=1024):
    """Yield lists of sample indices whose summed menu sizes fit the
    candidate budget (at least one sample per batch).

    Singleton batches freeze BatchNorm statistics in the caller; they remain
    valid training observations and must not silently disappear.
    """
    cur, cur_c = [], 0
    for si in order:
        k = int(off[si + 1] - off[si])
        if cur and (cur_c + k > budget or len(cur) >= b_max):
            yield cur
            cur, cur_c = [], 0
        cur.append(int(si))
        cur_c += k
    if cur:
        yield cur


def _require_fit_schedule(epochs, lr, *, batch=None):
    if type(epochs) is not int or epochs < 1:
        raise ValueError('Corpus fitting requires a positive epoch count')
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError('Corpus fitting requires a finite positive learning rate')
    if batch is not None and (type(batch) is not int or batch < 1):
        raise ValueError('Corpus fitting requires a positive batch size')


def _corpus_schedule(epochs, lr, off):
    """Bind the epoch schedule to the actual menu layout and batching budget."""
    import hashlib
    return {'scheme': 'actual_batch_epoch_cosine_v1', 'epochs': epochs,
            'learning_rate': lr, 'candidate_budget': CAND_BUDGET,
            'batch_max_rows': 1024,
            'offsets_sha256': hashlib.sha256(
                np.asarray(off, dtype='<i8').tobytes()).hexdigest()}


def _corpus_step_lr(opt, schedule, epoch, batch_index, batch_count):
    """Set the pre-update rate from completed progress; every batch gets a rate.

    Menu lengths and the 1024-row cap can make batch counts vary by epoch.
    Each epoch spans its share of the cosine using its actual batch count.
    The zero endpoint is reached only after the final requested update.
    """
    progress = (epoch + batch_index / batch_count) / schedule['epochs']
    rate = schedule['learning_rate'] * .5 * (1. + math.cos(math.pi * progress))
    if not math.isfinite(rate) or rate <= 0:
        raise RuntimeError('Corpus fitting requires a positive effective learning rate; refusing update')
    for group in opt.param_groups:
        group['lr'] = rate
    return rate


@contextlib.contextmanager
def _singleton_batch_norm(net, size):
    """Use stored BatchNorm statistics for one row, preserving affine gradients."""
    import torch
    modules = ([module for module in net.modules()
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.training]
               if size == 1 else [])
    try:
        for module in modules:
            module.eval()
        yield
    finally:
        for module in modules:
            module.train()


def train(epochs, lr, value_weight, joint=False):
    """BC training (joint=False) or the JOINT TD-TARGET pass
    (joint=True, USER 2026-07-27): warm-starts from the BC {NET_TAG}
    checkpoint and trains the FULL net — policy CE unchanged, value MSE
    toward the lambda-return targets {tag}_ztd.npy emitted by
    --stage value-td — so the nuanced value signal shapes the backbone
    while the policy loss keeps it honest.  Outputs {NET_TAG}jt nets."""
    _require_fit_schedule(epochs, lr, batch=CAND_BUDGET)
    split = _require_dataset_contract()
    import torch
    import ai
    try:
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    results = {}
    sfx = 'jt' if joint else ''
    return _train_impl(split, results, sfx, joint, epochs, lr,
                       value_weight)


def _bc_cert_partner(tag):
    """Fixed partner for BC snapshot certs (WOTR_BC_CERT note in
    _train_impl; override dir via WOTR_BC_CERT_PARTNER_DIR).  Partner
    = the loop's frozen BASELINE roster — always installed, and the
    same standing reference the arena scores against.  (The original
    v5_preserved default died when the user purged old checkpoints
    2026-08-12; its silent None return crashed the v38 chain.)"""
    import ai
    otag = 'sh' if tag == 'fp' else 'fp'
    pdir = os.environ.get('WOTR_BC_CERT_PARTNER_DIR') or os.path.join(
        ROOT, 'data', 'v33loop')
    name = ('baseline_fp_ring.pt' if otag == 'fp'
            else 'baseline_sh_military.pt')
    p = os.path.join(pdir, name)
    if not os.path.exists(p):
        sys.exit('BC cert partner missing: %s — install a loop '
                 'baseline roster (or set WOTR_BC_CERT_PARTNER_DIR) '
                 'before training with WOTR_BC_CERT=1' % p)
    net = ai.load_value_net(p, side=otag)
    if net is None:
        sys.exit('BC cert partner failed to load: %s' % p)
    net.eval()
    return net


def _head_act_of(net):
    """Checkpoint-described head activation (A6 contract): the R4
    leg-5 escape was save paths omitting this field — a scaled2-native
    net silently reloads as Tanh."""
    tail = net.value_head[-1]
    return ('scaled2' if type(tail).__name__ == '_ScaledTanhHead'
            else 'tanh')


def _reconcile_head(net, ck, who):
    """HUNT-B F3: cross-stage loads built the net from ENV
    (WOTR_VALUE_HEAD_ACT) and loaded weights from a checkpoint whose
    head may differ — both heads are parameter-free, so a mismatch
    loads cleanly and silently mis-shapes every value.  The
    checkpoint is the authority; env only shapes NEW nets."""
    import ai
    want = ck.get('head_act', 'tanh')
    if want != _head_act_of(net):
        if want == 'scaled2':
            net.value_head[-1] = ai._ScaledTanhHead(2.0)
        else:
            import torch.nn as _nn9
            net.value_head[-1] = _nn9.Tanh()
        print(f'[{who}] head reconciled to checkpoint: {want}',
              flush=True)


def _fresh_head(net, who):
    """SPTR-2/SPCP-3 (run-7 bug 2 re-bought in the seed pipeline):
    a FRESHLY-constructed seed net takes its head from
    WOTR_VALUE_HEAD_ACT with the ADOPTED default scaled2 — the v41
    chain ran env-less, ai's constructor defaulted to Tanh, and
    save() stamped tanh seeds while 42 percent of residual targets
    exceed the Tanh range.  Loud print of what save() will stamp.
    Checkpoint loads still override afterwards via _reconcile_head
    (the checkpoint stays the authority, HUNT-B F3)."""
    import ai
    env = os.environ.get('WOTR_VALUE_HEAD_ACT')
    want = env or 'scaled2'
    if want not in ('tanh', 'scaled2'):
        sys.exit('WOTR_VALUE_HEAD_ACT must be tanh|scaled2, got %r'
                 % env)
    if _head_act_of(net) != want:
        if want == 'scaled2':
            net.value_head[-1] = ai._ScaledTanhHead(2.0)
        else:
            import torch.nn as _nn9
            net.value_head[-1] = _nn9.Tanh()
    print('[%s] fresh net head_act=%s (WOTR_VALUE_HEAD_ACT=%s) - '
          'this is the value the checkpoint stamps'
          % (who, _head_act_of(net), env or 'unset -> default scaled2'),
          flush=True)
    return net


def _td_target_clip(net):
    """SPTR-2: residual-target clip for the value-td stage = the
    ADOPTED operator's head-matched reachable range
    (value_trainer_v2._head_clip) — ONE constant shared with the v2
    loop trainer.  The old hardcoded +-2.0 was the v1 op's clip and
    left every target unreachable under a Tanh head (run-7 bug 2)."""
    import value_trainer_v2 as _vt9
    return float(_vt9._head_clip(net))


def _head_train_bn_eval(head):
    """1.224e scope (SPTR-14): head Linears train, BatchNorm buffers
    stay frozen.  ONE implementation for value_td AND value_anchor."""
    import torch
    head.train()
    for m in head.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            m.eval()
    return head


def _bc_cert_hist_path(tag):
    return os.path.join(DS_DIR, f'{tag}_bc_cert_hist.json')


def _bc_cert_hist_save(tag, hist, paths):
    """Persist the BC cert history as certs complete (HUNT-A F8,
    WIRING 2026-08-11): a killed run resumed via CORPUS_RESUME used
    to restart with an EMPTY history — certified pre-kill epochs
    were forgotten, so an all-fail post-resume tail raised
    SystemExit(3) despite passing snapshots on disk.  Atomic write;
    epoch parsed from each snapshot filename (the path IS the epoch
    encoding)."""
    import re as _re9
    entries = []
    for p9, r9 in zip(paths, hist):
        m9 = _re9.search(r'_ep(\d+)\.pt$', os.path.basename(p9))
        entries.append({'epoch': int(m9.group(1)) if m9 else None,
                        'path': p9, 'report': r9})
    hp9 = _bc_cert_hist_path(tag)
    tmp9 = hp9 + '.tmp'
    with open(tmp9, 'w', encoding='utf-8') as f9:
        json.dump({'net_tag': NET_TAG, 'entries': entries}, f9)
    os.replace(tmp9, hp9)


def _bc_cert_hist_reload(tag, start_ep, hist, paths):
    """Resume side of HUNT-A F8: reload cert records for epochs <=
    the resume point whose snapshots still exist.  Returns the count
    reloaded (0 when no usable sidecar — a fresh run's first cert
    overwrites any stale one, so staleness cannot leak forward)."""
    hp9 = _bc_cert_hist_path(tag)
    if not os.path.exists(hp9):
        return 0
    try:
        hd9 = json.load(open(hp9, encoding='utf-8'))
    except (OSError, ValueError):
        return 0
    if hd9.get('net_tag') != NET_TAG:
        return 0
    n0 = len(hist)
    for e9 in hd9.get('entries') or ():
        ep9 = e9.get('epoch')
        if (isinstance(ep9, int) and ep9 <= start_ep
                and e9.get('path') and os.path.exists(e9['path'])):
            paths.append(e9['path'])
            hist.append(e9['report'])
    return len(hist) - n0


def _bc_cert_epoch(tag, ep, net, partner, snap_path, hist, paths):
    """Snapshot + depth battery for one BC epoch (mirrors the jt
    per-epoch block); appends to hist/paths, persists the history
    sidecar, returns the summary."""
    import torch
    import ai
    import jt_cert as jc
    torch.save({**_checkpoint_contract(), 'model_state': net.state_dict(),
                'input_dim': ai.get_feature_dim(),
                'action_dim': ai.ACTION_ENCODING_DIM,
                'arch': 'gnn_v4_full', 'side': tag, 'epoch': ep + 1,
                'scheme': 'full_menu_v30',
                'head_act': _head_act_of(net),
                'value_head_trained': getattr(net, 'value_head_trained', False) is True}, snap_path)
    paths.append(snap_path)
    was_training = net.training
    net.eval()
    rep = jc.cert_report(tag, net, partner)
    if was_training:
        net.train()
    hist.append(rep)
    _bc_cert_hist_save(tag, hist, paths)
    s = jc.summarize(rep)
    print(f"  {tag} ep {ep:>2} BC-CERT: "
          f"{'PASS' if s['pass'] else 'FAIL'} {s['fails'][:4]}",
          flush=True)
    return s


def _bc_cert_finalize(tag, net, hist, paths):
    """Stage output = LAST PASSING epoch; none -> SystemExit(3)
    (mirrors the jt finalize semantics exactly)."""
    import torch
    import ai
    import jt_cert as jc
    best = jc.pick_last_passing(hist)
    if best is None:
        print(f'[{tag}] BC CERT: NO epoch passed the depth battery '
              f'— stage REFUSED (SystemExit 3); per-epoch snapshots '
              f'retained for autopsy', flush=True)
        raise SystemExit(3)
    # HUNT-A F1: ALWAYS load the certified snapshot — with a cert
    # cadence > 1 the in-memory net can be ahead of the last
    # CERTIFIED epoch, and "keep current when best is newest" shipped
    # those uncertified tail epochs (the exact between-epoch-poison
    # window the cert exists for)
    ck = torch.load(paths[best], map_location=ai.DEVICE,
                    weights_only=False)
    _require_learning_contract(ck, paths[best])
    net.load_state_dict(ck['model_state'])
    net.value_head_trained = ck.get('value_head_trained') is True
    print(f'[{tag}] BC CERT: selected epoch {best + 1}/{len(paths)} '
          f'(last passing)', flush=True)
    return best


def _train_impl(split, results, sfx, joint, epochs, lr,
                value_weight):
    _require_fit_schedule(epochs, lr, batch=CAND_BUDGET)
    _require_dataset_contract(split)
    import torch
    import ai
    # DEPTH-BATTERY SEED CERT (USER 2026-08-10, v34 autopsy: the jt
    # stage silently repriced depth-2+ interior values; one-step
    # ladders passed poisoned rungs — only live search-Q fixture
    # behavior catches it).  WOTR_JT_CERT=1: per-epoch jt snapshots +
    # jt_cert.depth_battery; the stage OUTPUT becomes the LAST
    # PASSING epoch; no epoch passing -> SystemExit(3) (a retrain
    # must self-certify BEFORE anything is installed).  Partner =
    # same-era other-side BASE checkpoint (both-poisoned -> cert
    # fails -> correct outcome).  WOTR_JT_FREEZE_VALUE=1: value head
    # frozen during jt (params + BN buffers; 1.224e scope), value
    # gradient still shapes the trunk unless --value-weight 0.
    # Defaults OFF -> byte-identical stage.
    _jt_cert_on = joint and os.environ.get('WOTR_JT_CERT') == '1'
    _jt_freeze_val = joint and \
        os.environ.get('WOTR_JT_FREEZE_VALUE') == '1'
    # WOTR_BC_CERT=1 (R2 reset stage): the SAME depth-battery cert at
    # BC snapshot points — the v34 autopsy located the poison entering
    # between BC epochs 5 and 10, so the BC stage must self-certify
    # exactly like jt does.  Mirrors WOTR_JT_CERT semantics: per-epoch
    # snapshot -> depth battery -> stage OUTPUT = LAST PASSING epoch,
    # SystemExit(3) if none.  PARTNER = the PRESERVED KNOWN-GOOD v5
    # pair (not the same-era other side, which does not exist yet
    # during a fresh reset BC run): always available, 11/11 on
    # record, and a FIXED partner isolates the snapshot side (the
    # 1.224c arena-pin principle).  Cadence WOTR_BC_CERT_EVERY
    # (default 1).  Default OFF -> byte-identical stage.
    _bc_cert_on = (not joint) and os.environ.get('WOTR_BC_CERT') == '1'
    _bc_cert_every = int(os.environ.get('WOTR_BC_CERT_EVERY', '1')
                         or 1)
    # v32: side selection for split FP/SH schedules
    _sides = [s.strip().lower() for s in
              (os.environ.get('CORPUS_TRAIN_SIDES') or 'fp,sh').split(',')
              if s.strip()]
    if not _sides or any(s not in ('fp', 'sh') for s in _sides):
        sys.exit('CORPUS_TRAIN_SIDES must be a comma list of fp/sh, '
                 'got %r' % os.environ.get('CORPUS_TRAIN_SIDES'))
    for tag, sc, out_name in (
            ('fp', 0, f'wotr_fp_corpus_{NET_TAG}{sfx}.pt'),
            ('sh', 1, f'wotr_sh_corpus_{NET_TAG}{sfx}.pt')):
        if tag not in _sides:
            continue
        X = np.load(os.path.join(DS_DIR, f'{tag}_X.npy'), mmap_mode='r')
        zname = f'{tag}_ztd.npy' if joint else f'{tag}_z.npy'
        zp = os.path.join(DS_DIR, zname)
        if joint and not os.path.exists(zp):
            sys.exit('train-joint needs %s — run --stage value-td first'
                     % zp)
        if joint:
            # HUNT-A F3: refuse a ztd without the residual-currency
            # marker (a stale absolute-currency file trains the value
            # head in the WRONG currency while logging success)
            _zmp9 = os.path.join(DS_DIR, f'{tag}_ztd_meta.json')
            if not os.path.exists(_zmp9):
                sys.exit(f'{zname} has no currency marker — stale '
                         f'pre-v37 targets; re-run --stage value-td')
            _zm9 = json.load(open(_zmp9, encoding='utf-8'))
            _require_learning_contract(_zm9, _zmp9)
            if _zm9.get('currency') != 'residual-z-phi':
                sys.exit(f'{zname} currency {_zm9.get("currency")!r} '
                         f'!= residual-z-phi — re-run --stage value-td')
        # v37 CURRENCY COHERENCE (USER root-cause order 2026-08-10):
        # the value head trains the RESIDUAL (search evaluates
        # net+phi; the TD loop trains G-phi — one currency end to
        # end).  BC targets = z - phi; the value-td stage emits ztd
        # ALREADY residual, so the joint pass consumes it unchanged.
        # A pre-v37 ds (no {tag}_phi.npy) fails LOUDLY: silent
        # phi=0 would re-create the absolute-currency training this
        # order abolishes.
        _php = os.path.join(DS_DIR, f'{tag}_phi.npy')
        if not os.path.exists(_php):
            sys.exit('v37 currency coherence needs %s — re-extract '
                     '+ re-consolidate (pre-v37 ds is superseded)'
                     % _php)
        phi_arr = np.load(_php, mmap_mode='r')
        # v37b ANCHOR BLEND (smoke-refusal fix, measured in
        # experiments\VALUE_corpus_rings): pure-ztd jt DILUTES
        # state-conditional structure (fp spend-sensitivity -0.0203
        # certified-BC -> -0.0069 jt) while pure z-phi head anchoring
        # destroys lambda-trajectory structure (mordor qgap 0.309 ->
        # flip).  WOTR_JT_VALUE_BLEND=b mixes the two target
        # families: value target = (1-b)*ztd + b*(z-phi), residual
        # both.  Default 0 = pure ztd = byte-identical stage.
        _jvb = float(os.environ.get('WOTR_JT_VALUE_BLEND', '0')
                     or 0)
        if joint:
            z = np.load(zp, mmap_mode='r')       # ztd: residual already
            if _jvb > 0:
                _zr = (np.asarray(np.load(
                    os.path.join(DS_DIR, f'{tag}_z.npy'),
                    mmap_mode='r'), dtype=np.float32)
                    - np.asarray(phi_arr, dtype=np.float32))
                z = ((1.0 - _jvb) * np.asarray(z, dtype=np.float32)
                     + _jvb * _zr)
                print(f"[{tag}] JT VALUE BLEND {_jvb}: targets = "
                      f"{1 - _jvb:.2f}*ztd + {_jvb:.2f}*(z-phi) "
                      f"(residual both; v37b smoke fix)", flush=True)
        else:
            z = (np.asarray(np.load(zp, mmap_mode='r'),
                            dtype=np.float32)
                 - np.asarray(phi_arr, dtype=np.float32))
        print(f"[{tag}] VALUE TARGETS ARE RESIDUAL (z - phi; v37 "
              f"currency coherence) — the logged mse is RESIDUAL-mse, "
              f"not absolute-outcome mse; do not compare its magnitude "
              f"to pre-v37 curves", flush=True)
        off = np.load(os.path.join(DS_DIR, f'{tag}_off.npy'))
        cands = np.load(os.path.join(DS_DIR, f'{tag}_cands.npy'),
                        mmap_mode='r')
        regs = np.load(os.path.join(DS_DIR, f'{tag}_regs.npy'),
                       mmap_mode='r')
        pos = np.load(os.path.join(DS_DIR, f'{tag}_pos.npy'),
                      mmap_mode='r')
        # v32: declare-window flag (reporting only, loss unweighted)
        _dwp = os.path.join(DS_DIR, f'{tag}_dw.npy')
        # every post-v32 consolidate() writes dw; a missing file is a
        # broken dataset and must fail loudly (like the phi gate above)
        dw = np.load(_dwp)
        n, m = X.shape[0], cands.shape[0]
        if n < 1:
            raise ValueError(f'Corpus fitting [{tag}] requires at least one training row')
        _ndw = int(dw.sum())
        print(f"\n[{tag}]{' JOINT-TD' if joint else ''} {n} samples, "
              f"{m} candidates (mean menu {m / max(1, n):.1f}), "
              f"{epochs} epochs, budget {CAND_BUDGET}, "
              f"declare-windows {_ndw} (unweighted CE)", flush=True)
        torch.manual_seed(SEED)
        net = ai.WOTRValueNet(ai.get_feature_dim(), side=tag).to(ai.DEVICE)
        net.value_head_trained = False
        # SPTR-2: env-honored head on the fresh seed (default scaled2,
        # loud); the jt/resume checkpoint loads below still override
        _fresh_head(net, tag)
        # CORPUS_RESUME=1 continues from the newest {tag}_{NET_TAG}_ep*.pt
        # snapshot that carries optimizer state - exact continuation
        # (model+opt+sched+np/torch RNG).
        _start_ep = 0
        _resume_ck = None
        if os.environ.get('CORPUS_RESUME') == '1' and not joint:
            import glob as _glob
            _snaps = sorted(
                _glob.glob(os.path.join(
                    ROOT, f'wotr_{tag}_corpus_{NET_TAG}_ep*.pt.resume')),
                # numeric epoch order: lexicographic put ep100 < ep99
                key=lambda p9: int(
                    re.search(r'_ep(\d+)\.pt\.resume$', p9).group(1)))
            if _snaps:
                _resume_ck = torch.load(_snaps[-1],
                                        map_location=ai.DEVICE,
                                        weights_only=False)
                _require_learning_contract(_resume_ck, _snaps[-1])
                print(f"  [{tag}] RESUME from {_snaps[-1]} "
                      f"(epoch {_resume_ck['epoch']})", flush=True)
        if joint:
            # warm start from the BC checkpoint (policy stays anchored
            # by the CE loss while ztd reshapes the value pathway)
            _ck_jt = torch.load(
                os.path.join(ROOT, f'wotr_{tag}_corpus_{NET_TAG}.pt'),
                map_location=ai.DEVICE, weights_only=False)
            _require_learning_contract(_ck_jt, f'{tag} joint-training seed')
            net.load_state_dict(_ck_jt['model_state'])
            net.value_head_trained = _ck_jt.get('value_head_trained') is True
            _reconcile_head(net, _ck_jt, f'{tag} jt-warm')
            if _jt_freeze_val:
                for _p9 in net.value_head.parameters():
                    _p9.requires_grad = False
                print(f'  [{tag}] JT value head FROZEN '
                      f'(WOTR_JT_FREEZE_VALUE=1)', flush=True)
        _cert_hist = []
        _snap_paths = []
        _partner9 = None
        if _jt_cert_on:
            import jt_cert as _jc9
            _otag = 'sh' if tag == 'fp' else 'fp'
            _pck9 = torch.load(
                os.path.join(ROOT,
                             f'wotr_{_otag}_corpus_{NET_TAG}.pt'),
                map_location=ai.DEVICE, weights_only=False)
            _require_learning_contract(_pck9, f'{_otag} joint certificate partner')
            _partner9 = ai.WOTRValueNet(ai.get_feature_dim(),
                                        side=_otag).to(ai.DEVICE)
            _partner9.load_state_dict(_pck9['model_state'])
            _reconcile_head(_partner9, _pck9, f'{_otag} cert-partner')
            _partner9.eval()
        _bc_partner9 = _bc_cert_partner(tag) if _bc_cert_on else None
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
        schedule = _corpus_schedule(epochs, lr, off)
        rng = np.random.RandomState(SEED)
        if _resume_ck is not None:
            saved_schedule = _resume_ck.get('lr_schedule')
            if saved_schedule is not None and saved_schedule != schedule:
                raise ValueError('Corpus resume learning-rate schedule differs from the requested '
                                 'epochs, rate, menu layout or budget; use a new checkpoint tag')
            net.load_state_dict(_resume_ck['model_state'])
            net.value_head_trained = _resume_ck.get('value_head_trained') is True
            _reconcile_head(net, _resume_ck, f'{tag} resume')
            opt.load_state_dict(_resume_ck['opt_state'])
            rng.set_state(_resume_ck['np_rng_state'])
            torch.set_rng_state(_resume_ck['torch_rng_state'].cpu())
            if torch.cuda.is_available() and \
                    _resume_ck.get('cuda_rng_state') is not None:
                torch.cuda.set_rng_state(
                    _resume_ck['cuda_rng_state'].cpu())
            _start_ep = int(_resume_ck['epoch'])
            if saved_schedule is None:
                print(f'  [{tag}] legacy resume schedule rebuilt at completed epoch '
                      f'{_start_ep}/{epochs} using lr={lr:g}; obsolete scheduler '
                      'state and its current rate are not reused', flush=True)
            if _bc_cert_on:
                _nrl9 = _bc_cert_hist_reload(
                    tag, _start_ep, _cert_hist, _snap_paths)
                print(f'  [{tag}] BC-CERT resume: {_nrl9} certified '
                      f'epoch record(s) reloaded from sidecar',
                      flush=True)
        if _start_ep >= epochs:
            raise ValueError(f'Corpus fitting [{tag}] has no remaining epochs; refusing publication')
        _p_cd = float(os.environ.get('VALUE_CTX_DROPOUT', '0') or 0)
        # batch assembly is the wall (memmap gathers + menu padding, GPU
        # ~8% util inline) — numpy-only spawn workers overlap it
        import corpus_feed
        corpus_feed.init(DS_DIR, tag)
        _fw = int(os.environ.get('WOTR_FEED_WORKERS', '3') or 0)
        _pool = None
        if _fw > 0:
            try:
                import multiprocessing as _mp
                _pool = _mp.get_context('spawn').Pool(
                    _fw, initializer=corpus_feed.init,
                    initargs=(DS_DIR, tag))
            except Exception as _e:
                print(f'  [{tag}] feed pool unavailable ({_e!r}); '
                      f'inline assembly', flush=True)
                _pool = None
        t0 = time.time()
        curve = []
        optimizer_steps = 0
        try:
          for ep in range(_start_ep, epochs):
            net.train()
            if _jt_freeze_val:
                # frozen scope incl. BN buffers (1.224e: requires_grad
                # does not freeze running stats)
                net.value_head.eval()
            perm = rng.permutation(n)
            tot_ce = tot_mse = tot_acc = 0.0
            nb = 0
            _bat = list(_budget_batches(perm, off, CAND_BUDGET))
            _bit = (corpus_feed.warm_iter(_pool, _bat, _fw * 4)
                    if _pool is not None
                    else (corpus_feed.assemble(b) for b in _bat))
            for bi, xbn, cb, rb, pb, mb in _bit:
                _corpus_step_lr(opt, schedule, ep, nb, len(_bat))
                B = len(bi)
                xb = torch.from_numpy(xbn).to(ai.DEVICE)
                if xb.dtype == torch.float16:
                    xb = xb.float()
                zb = torch.from_numpy(
                    np.ascontiguousarray(z[bi])).to(ai.DEVICE)
                cb_t = torch.from_numpy(cb).to(ai.DEVICE)
                if cb_t.dtype == torch.float16:
                    cb_t = cb_t.float()
                rb_t = torch.from_numpy(rb).to(ai.DEVICE)
                pb_t = torch.from_numpy(pb).to(ai.DEVICE)
                mb_t = torch.from_numpy(mb).to(ai.DEVICE)
                with _singleton_batch_norm(net, B):
                    h, ne = net._process_input(xb)
                    # VALUE context-dropout (amendment 1.130) preserved
                    if _p_cd > 0:
                        _cs = ai.get_pending_context_start()
                        xb2 = xb.clone()
                        _dm = (torch.rand(xb2.shape[0], device=xb2.device)
                               < _p_cd)
                        xb2[_dm, _cs:] = 0.0
                        h2, _ = net._process_input(xb2)
                        v = net.value_head(h2).squeeze(-1)
                    else:
                        v = net.value_head(h).squeeze(-1)
                    logits = batched_scores(net, h, ne, cb_t, rb_t)
                logits = logits.masked_fill(~mb_t, -1e9)
                # multi-positive listwise CE: -log(sum_pos p)
                lse_all = torch.logsumexp(logits, dim=1)
                lse_pos = torch.logsumexp(
                    logits.masked_fill(~pb_t, -1e9), dim=1)
                ce = (lse_all - lse_pos).mean()
                mse = torch.nn.functional.mse_loss(v, zb)
                loss = ce + value_weight * mse
                opt.zero_grad()
                loss.backward()
                opt.step()
                optimizer_steps += 1
                if value_weight > 0 and not _jt_freeze_val:
                    net.value_head_trained = True
                with torch.no_grad():
                    am = logits.argmax(1)
                    tot_acc += float(pb_t.gather(
                        1, am.unsqueeze(1)).float().mean())
                tot_ce += float(ce.detach())
                tot_mse += float(mse.detach())
                nb += 1
            if not nb:
                raise RuntimeError(f'Corpus fitting [{tag}] completed no optimizer steps; refusing publication')
            curve.append({'ep': ep, 'ce': tot_ce / nb,
                          'mse': tot_mse / nb,
                          'acc_full': tot_acc / nb})
            print(f"  {tag} ep {ep:>2}: ce={tot_ce / nb:.4f} "
                  f"mse={tot_mse / nb:.4f} "
                  f"acc_full={tot_acc / nb:.3f} "
                  f"[{time.time() - t0:.0f}s, {nb} steps]", flush=True)
            if _jt_cert_on:
                import jt_cert as _jc9
                snap9 = os.path.join(
                    ROOT,
                    f'wotr_{tag}_corpus_{NET_TAG}jt_ep{ep + 1:02d}.pt')
                torch.save({**_checkpoint_contract(), 'model_state': net.state_dict(),
                            'input_dim': ai.get_feature_dim(),
                            'action_dim': ai.ACTION_ENCODING_DIM,
                            'arch': 'gnn_v4_full', 'side': tag,
                            'epoch': ep + 1,
                            'scheme': 'full_menu_v30+joint_td',
                            'head_act': _head_act_of(net),
                            'value_head_trained': net.value_head_trained}, snap9)
                _snap_paths.append(snap9)
                net.eval()
                rep9 = _jc9.cert_report(tag, net, _partner9)
                net.train()
                if _jt_freeze_val:
                    net.value_head.eval()
                _cert_hist.append(rep9)
                _sum9 = _jc9.summarize(rep9)
                curve[-1]['cert'] = _sum9
                print(f"  {tag} ep {ep:>2} CERT: "
                      f"{'PASS' if _sum9['pass'] else 'FAIL'} "
                      f"{_sum9['fails'][:4]}", flush=True)
            if _bc_cert_on and (ep + 1) % _bc_cert_every == 0:
                snap9b = os.path.join(
                    ROOT,
                    f'wotr_{tag}_corpus_{NET_TAG}_ep{ep + 1:02d}.pt')
                curve[-1]['cert'] = _bc_cert_epoch(
                    tag, ep, net, _bc_partner9, snap9b,
                    _cert_hist, _snap_paths)
            if SNAP_EVERY and not joint and (ep + 1) % SNAP_EVERY == 0 \
                    and (ep + 1) < epochs:
                snap = os.path.join(
                    ROOT, f'wotr_{tag}_corpus_{NET_TAG}_ep{ep + 1:02d}.pt')
                torch.save({**_checkpoint_contract(), 'model_state': net.state_dict(),
                            'input_dim': ai.get_feature_dim(),
                            'action_dim': ai.ACTION_ENCODING_DIM,
                            'arch': 'gnn_v4_full', 'side': tag,
                            'epoch': ep + 1,
                            'scheme': 'full_menu_v30',
                            'head_act': _head_act_of(net),
                            'value_head_trained': net.value_head_trained}, snap)
                # v32 kill-resilience: exact-resume state rides a
                # SIDECAR (the snapshot itself stays loadable under
                # weights_only=True consumers)
                torch.save({**_checkpoint_contract(), 'model_state': net.state_dict(),
                            'epoch': ep + 1,
                            'head_act': _head_act_of(net),
                            'value_head_trained': net.value_head_trained,
                            'opt_state': opt.state_dict(),
                            'lr_schedule': schedule,
                            'np_rng_state': rng.get_state(),
                            'torch_rng_state': torch.get_rng_state(),
                            'cuda_rng_state': (
                                torch.cuda.get_rng_state()
                                if torch.cuda.is_available() else None)},
                           snap + '.resume')
                print(f"  {tag} snapshot -> {snap}", flush=True)
        finally:
            if _pool is not None:
                _pool.close()
                _pool.join()
        if not optimizer_steps:
            raise RuntimeError(f'Corpus fitting [{tag}] completed no optimizer steps; refusing publication')
        if _jt_cert_on:
            import jt_cert as _jc9
            _best9 = _jc9.pick_last_passing(_cert_hist)
            if _best9 is None:
                results[tag] = {'n': n, 'm': m, 'curve': curve,
                                'cert': 'NO PASSING EPOCH'}
                with open(os.path.join(DS_DIR, 'train_curves.json'),
                          'w', encoding='utf-8') as _f9:
                    json.dump(results, _f9, indent=1)
                print(f'[{tag}] JT CERT: NO epoch passed the depth '
                      f'battery — stage REFUSED (SystemExit 3); '
                      f'per-epoch snapshots + curves retained for '
                      f'autopsy', flush=True)
                raise SystemExit(3)
            # HUNT-A F1 (jt twin): unconditional load of the
            # certified snapshot — never ship the in-memory tail
            _ck9 = torch.load(_snap_paths[_best9],
                              map_location=ai.DEVICE,
                              weights_only=False)
            _require_learning_contract(_ck9, _snap_paths[_best9])
            net.load_state_dict(_ck9['model_state'])
            net.value_head_trained = _ck9.get('value_head_trained') is True
            print(f'[{tag}] JT CERT: selected epoch '
                  f'{_best9 + 1}/{len(_snap_paths)} '
                  f'(last passing)', flush=True)
        if _bc_cert_on:
            try:
                _bc_cert_finalize(tag, net, _cert_hist, _snap_paths)
            except SystemExit:
                # refusal autopsy trail mirrors the jt path: curves
                # written BEFORE the raise
                results[tag] = {'n': n, 'm': m, 'curve': curve,
                                'cert': 'NO PASSING EPOCH'}
                with open(os.path.join(DS_DIR, 'train_curves.json'),
                          'w', encoding='utf-8') as _f9:
                    json.dump(results, _f9, indent=1)
                raise
        out = os.path.join(ROOT, out_name)
        import torch as _t
        _t.save({**_checkpoint_contract(), 'model_state': net.state_dict(),
                 'input_dim': ai.get_feature_dim(),
                 'action_dim': ai.ACTION_ENCODING_DIM,
                 'arch': 'gnn_v4_full', 'side': tag,
                 'trained_on': f'{CORPUS} ({len(split["train"])} games, '
                               f'{n} samples, full-menu v30'
                               f'{", joint TD targets" if joint else ""})',
                 'scheme': ('full_menu_v30+joint_td' if joint
                            else 'full_menu_v30'),
                 'head_act': _head_act_of(net),
                 'value_head_trained': net.value_head_trained}, out)
        print(f"[{tag}] saved {out}")
        results[tag] = {'n': n, 'm': m, 'curve': curve}
    with open(os.path.join(DS_DIR, 'train_curves.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=1)
    return results


# ------------------------------------------------------------------
# Stage 3: full-candidate-set evaluation (duplicate-tolerant) — the
# same measurement as before; train and eval now share a distribution.
# ------------------------------------------------------------------

def eval_games(nets, names, label, max_games=None):
    import torch
    import ai
    rng = np.random.RandomState(SEED)
    if max_games and len(names) > max_games:
        names = list(rng.choice(names, max_games, replace=False))
    agg = {0: [0, 0], 1: [0, 0]}
    per_game = {}
    for gi, f in enumerate(names):
        p = os.path.join(CORPUS, f)
        if is_empty_npz(p):
            continue
        with np.load(p) as zf:
            _require_npz_contract(zf, p)
            X, side = zf['X'], zf['side']
            human, offs = zf['human'], zf['offsets']
            ce, cr = zf['cand_encs'], zf['cand_regs']
            _alt_i = zf['alt_idx'] if 'alt_idx' in zf.files else None
            _alt_o = zf['alt_offsets'] if 'alt_offsets' in zf.files else None
            g_ok = g_n = 0
            with torch.no_grad():
                for lo in range(0, len(X), 256):
                    hi = min(lo + 256, len(X))
                    xb = torch.from_numpy(X[lo:hi].astype(
                        np.float32)).to(ai.DEVICE)
                    hs = {}
                    for sc, tag in ((0, 'fp'), (1, 'sh')):
                        hs[sc] = nets[tag]._process_input(xb)
                    for i in range(lo, hi):
                        sc = int(side[i])
                        net = nets['fp' if sc == 0 else 'sh']
                        h, ne = hs[sc]
                        menu = _normalize_cands(
                            ce[offs[i]:offs[i + 1]].astype(np.float32))
                        e = torch.from_numpy(menu).to(ai.DEVICE)
                        r = torch.from_numpy(
                            cr[offs[i]:offs[i + 1]].astype(
                                np.int64)).to(ai.DEVICE)
                        pri = net.score_actions_spatial(
                            h[i - lo:i - lo + 1], ne[i - lo:i - lo + 1],
                            e, r)
                        a = int(pri.argmax())
                        hidx = int(human[i])
                        rr = cr[offs[i]:offs[i + 1]]
                        # 1.268: any declared-equivalent alternate (and
                        # its encoding twins) is a correct prediction too
                        _pk = {_dup_key(menu[hidx], rr[hidx])}
                        if _alt_i is not None:
                            for _a in _alt_i[int(_alt_o[i]):int(_alt_o[i + 1])]:
                                _a = int(_a)
                                if 0 <= _a < len(menu):
                                    _pk.add(_dup_key(menu[_a], rr[_a]))
                        ok = (a == hidx) or (
                            _dup_key(menu[a], rr[a]) in _pk)
                        agg[sc][0] += int(ok)
                        agg[sc][1] += 1
                        g_ok += int(ok)
                        g_n += 1
            per_game[f] = {'correct': g_ok, 'n': g_n,
                           'acc': g_ok / max(1, g_n)}
        if gi % 25 == 0:
            print(f"  [{label}] {gi}/{len(names)}", flush=True)
    out = {'fp': {'correct': agg[0][0], 'n': agg[0][1],
                  'acc': agg[0][0] / max(1, agg[0][1])},
           'sh': {'correct': agg[1][0], 'n': agg[1][1],
                  'acc': agg[1][0] / max(1, agg[1][1])},
           'per_game': per_game}
    print(f"[{label}] FP {out['fp']['correct']}/{out['fp']['n']} "
          f"({out['fp']['acc']:.3f})  SH {out['sh']['correct']}/"
          f"{out['sh']['n']} ({out['sh']['acc']:.3f})", flush=True)
    return out


def evaluate():
    split = _require_dataset_contract()
    import torch
    import ai
    nets = {}
    for tag, name in (('fp', f'wotr_fp_corpus_{NET_TAG}.pt'),
                      ('sh', f'wotr_sh_corpus_{NET_TAG}.pt')):
        ckpt = torch.load(os.path.join(ROOT, name),
                          map_location=ai.DEVICE, weights_only=False)
        _require_learning_contract(ckpt, name)
        net = ai.WOTRValueNet(ckpt['input_dim'], side=tag).to(ai.DEVICE)
        net.load_state_dict(ckpt['model_state'])
        _reconcile_head(net, ckpt, f'{tag} stage-load')
        net.eval()
        nets[tag] = net
    rep = {
        'holdout': eval_games(nets, split['holdout'], 'HELD-OUT'),
        'train_sample': eval_games(nets, split['train'],
                                   'TRAIN-SAMPLE', max_games=200),
    }
    with open(os.path.join(DS_DIR, 'eval_report.json'), 'w',
              encoding='utf-8') as f:
        json.dump(rep, f, indent=1)
    print('wrote', os.path.join(DS_DIR, 'eval_report.json'))


# ------------------------------------------------------------------
# Stage 4: TD(lambda) VALUE retraining over the replayed human games
# ------------------------------------------------------------------
# The BC stage labels every state of a game with the same final +/-1 —
# correct on average, astronomically noisy per sample (a turn-3 state
# is blamed/credited for 200 later decisions), which is why the value
# head plateaued near MSE 0.61.  This stage rebuilds the targets as
# offline lambda-returns bootstrapped from the CURRENT net:
#
#   per game, per side, states s_1..s_N in turn order, outcome z:
#     G_N = z
#     G_t = (1 - lam) * V(s_{t+1}) + lam * G_{t+1}
#
# lam=1 recovers the raw-outcome scheme; lam~0.8 localizes credit (an
# eval collapse between s_t and s_{t+1} moves the TARGET at s_t).  Only
# the VALUE HEAD trains — backbone + policy scorer stay frozen, so the
# BC policy is untouched and the backbone embeddings can be cached once
# (h is constant under a frozen backbone; rounds re-bootstrap V from
# the updated head over the cached h).  Iterated --td-rounds re-derive
# the targets with the improved head (policy iteration on V).
# Input nets: wotr_{fp,sh}_corpus_{NET_TAG}.pt
# Output:     wotr_{fp,sh}_corpus_{NET_TAG}td.pt + td_value_report.json
# ------------------------------------------------------------------

def _td_collect_side(nets, names, sc, tag):
    """Stream games; return (H, z, game_id, turn, alidx) for side `sc`
    with H = frozen-backbone embeddings, trajectory order = (game,
    proven decision boundary). alidx = each row's index in the CONSOLIDATED layout
    (games in `names` order, samples in npz order) so lambda-return
    targets can be scattered into a {tag}_ztd.npy aligned with
    {tag}_X.npy for the joint TD-target pass."""
    import torch
    import ai
    net = nets[tag]
    net.eval()
    Hs, zs, gids, turns, alidxs, phis = [], [], [], [], [], []
    base = 0
    for gi, f in enumerate(names):
        p = os.path.join(CORPUS, f)
        if is_empty_npz(p):
            continue
        with np.load(p) as zf:
            _require_npz_contract(zf, p)
            side = zf['side']
            sel = np.flatnonzero(side == sc)
            if len(sel) == 0:
                continue
            X = zf['X'][sel]
            z = zf['z'][sel].astype(np.float32)
            # v37 schema: per-sample phi REQUIRED (KeyError on a
            # pre-v37 npz = the intended loud failure)
            phi = zf['phi'][sel].astype(np.float32)
            turn = zf['turn'][sel].astype(np.int32)
            from rendered_history import CHRONOLOGY_CONTRACT, chronological_order
            if ('decision_time' not in zf or 'chronology_contract' not in zf or
                    np.any(zf['chronology_contract'][sel] != CHRONOLOGY_CONTRACT)):
                raise ValueError(f'{p}: TD needs re-extracted decision chronology; unordered data is policy-only')
            order = chronological_order(zf['decision_time'][sel], turns=turn)
            X, z, turn = X[order], z[order], turn[order]
            phi = phi[order]
            alidxs.append(base + order.astype(np.int64))
            base += len(sel)
            with torch.no_grad():
                for lo in range(0, len(X), 512):
                    xb = torch.from_numpy(
                        X[lo:lo + 512].astype(np.float32)).to(ai.DEVICE)
                    h, _ = net._process_input(xb)
                    Hs.append(h.cpu().numpy().astype(np.float32))
            zs.append(z)
            gids.append(np.full(len(X), gi, dtype=np.int32))
            turns.append(turn)
            phis.append(phi)
        if gi % 200 == 0:
            print(f"  [{tag}] embed {gi}/{len(names)}", flush=True)
    if not Hs:
        return None
    return (np.concatenate(Hs), np.concatenate(zs),
            np.concatenate(gids), np.concatenate(turns),
            np.concatenate(alidxs), np.concatenate(phis))


def _td_lambda_returns(V, z, gid, lam):
    """Backward lambda-return per game trajectory (proven decision order)."""
    G = np.empty_like(V)
    n = len(V)
    i = n - 1
    while i >= 0:
        g = gid[i]
        j = i
        while j >= 0 and gid[j] == g:
            j -= 1
        lo = j + 1                       # trajectory = [lo, i]
        G[i] = z[i]                      # terminal anchor
        for t in range(i - 1, lo - 1, -1):
            G[t] = (1.0 - lam) * V[t + 1] + lam * G[t + 1]
        i = lo - 1
    return G


def _td_diagnostics(head, H, z, turn, label, phi=None):
    """Holdout sanity: COMPOSED value (head+phi, v37 residual
    currency) vs final z + Pearson r, per turn bucket.  phi=None
    keeps the legacy absolute reading (never used post-v37)."""
    import torch
    import ai
    with torch.no_grad():
        V = []
        for lo in range(0, len(H), 8192):
            hb = torch.from_numpy(H[lo:lo + 8192]).to(ai.DEVICE)
            V.append(head(hb).squeeze(-1).cpu().numpy())
    V = np.concatenate(V)
    if phi is not None:
        V = V + phi          # composed total value in outcome units
    out = {}
    for name, msk in (('early_t1-3', turn <= 3),
                      ('mid_t4-7', (turn >= 4) & (turn <= 7)),
                      ('late_t8+', turn >= 8),
                      ('all', np.ones(len(turn), dtype=bool))):
        if msk.sum() < 10:
            continue
        v, zz = V[msk], z[msk]
        # r undefined when either side is constant (e.g. a tiny holdout
        # where every sample shares one game's outcome)
        if v.std() > 1e-6 and zz.std() > 1e-6:
            r = round(float(np.corrcoef(v, zz)[0, 1]), 4)
        else:
            r = None
        out[name] = {'n': int(msk.sum()),
                     'mse_vs_z': float(np.mean((v - zz) ** 2)),
                     'pearson_r': r}
    print(f"  [{label}] " + "  ".join(
        "%s: mse=%.3f r=%s" % (k, d['mse_vs_z'],
                               '-' if d['pearson_r'] is None
                               else '%.3f' % d['pearson_r'])
        for k, d in out.items()), flush=True)
    return out


def value_td(lam, rounds, epochs, lr, batch):
    _require_fit_schedule(epochs, lr, batch=batch)
    if type(rounds) is not int or rounds < 1:
        raise ValueError('Corpus TD fitting requires a positive round count')
    split = _require_dataset_contract()
    import torch
    import ai
    report = {'lambda': lam, 'rounds': rounds, 'epochs': epochs}
    nets = {}
    for tag, name in (('fp', f'wotr_fp_corpus_{NET_TAG}.pt'),
                      ('sh', f'wotr_sh_corpus_{NET_TAG}.pt')):
        ckpt = torch.load(os.path.join(ROOT, name),
                          map_location=ai.DEVICE, weights_only=False)
        _require_learning_contract(ckpt, name)
        net = ai.WOTRValueNet(ckpt['input_dim'], side=tag).to(ai.DEVICE)
        net.load_state_dict(ckpt['model_state'])
        _reconcile_head(net, ckpt, f'{tag} stage-load')
        net.eval()
        nets[tag] = net
    for tag, sc in (('fp', 0), ('sh', 1)):
        net = nets[tag]
        print(f"\n[{tag}] TD(lambda={lam}) value retraining", flush=True)
        tr = _td_collect_side(nets, split['train'], sc, tag)
        ho = _td_collect_side(nets, split['holdout'], sc, tag)
        if tr is None:
            print(f"  [{tag}] no samples — skipped")
            continue
        H, z, gid, turn, alidx, phi = tr
        print(f"  [{tag}] {len(H)} train states, "
              f"{0 if ho is None else len(ho[0])} holdout states",
              flush=True)
        print(f"  [{tag}] RESIDUAL CURRENCY (v37): lambda-returns "
              f"over TOTAL values (head+phi), targets stored as "
              f"G-phi — mirrors run_td_v33loop._value_update_op; "
              f"td-mse below is residual-mse", flush=True)
        head = net.value_head
        # SPTR-2: head-matched clip shared with the adopted operator
        # (value_trainer_v2._head_clip prints LOUDLY on a legacy Tanh
        # head) — the old hardcoded +-2.0 was the v1 op's constant
        _clip9 = _td_target_clip(net)
        if ho is not None:
            report[f'{tag}_before'] = _td_diagnostics(
                head, ho[0], ho[1], ho[3], f'{tag} BEFORE',
                phi=ho[5])
        opt = torch.optim.Adam(head.parameters(), lr=lr,
                               weight_decay=1e-5)
        rng = np.random.RandomState(SEED)
        # SPTR-14: fewer samples than one batch must not silently
        # train zero steps while printing td-mse 0.0000
        _bs9 = min(batch, len(H))
        if _bs9 < 1:
            raise RuntimeError(
                'value_td[%s]: only %d samples - nothing to fit'
                % (tag, len(H)))
        optimizer_steps = 0
        for rd in range(rounds):
            # v37: bootstrap over TOTAL values (residual head + phi),
            # mirroring td_value_update: v_total = net(enc) + phi;
            # G anchors at z (z IS total — outcome); target = G - phi
            with torch.no_grad():
                V = np.concatenate([
                    head(torch.from_numpy(H[lo:lo + 8192]).to(
                        ai.DEVICE)).squeeze(-1).cpu().numpy()
                    for lo in range(0, len(H), 8192)])
            G = _td_lambda_returns(V + phi, z, gid, lam)
            # HUNT-A F4 + SPTR-2: residual-target clip matched to the
            # head's reachable range (ONE constant with the adopted
            # operator, value_trainer_v2._head_clip)
            T = np.clip(G - phi, -_clip9, _clip9).astype(np.float32)
            _head_train_bn_eval(head)      # 1.224e scope (SPTR-14)
            for ep in range(epochs):
                perm = rng.permutation(len(H))
                tot = nb = 0
                for lo in range(0, len(H), _bs9):
                    bi = perm[lo:lo + _bs9]
                    hb = torch.from_numpy(H[bi]).to(ai.DEVICE)
                    gb = torch.from_numpy(T[bi]).to(ai.DEVICE)
                    v = head(hb).squeeze(-1)
                    loss = torch.nn.functional.mse_loss(v, gb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    optimizer_steps += 1
                    tot += float(loss.detach())
                    nb += 1
                print(f"  {tag} round {rd} ep {ep}: "
                      f"td-mse={tot / max(1, nb):.4f} (residual)",
                      flush=True)
            head.eval()
        if not optimizer_steps:
            raise RuntimeError(f'value_td[{tag}] completed no optimizer steps; refusing publication')
        if ho is not None:
            report[f'{tag}_after'] = _td_diagnostics(
                head, ho[0], ho[1], ho[3], f'{tag} AFTER',
                phi=ho[5])
        # emit CONSOLIDATED-ALIGNED RESIDUAL lambda-return targets for
        # the joint TD-target pass (--stage train-joint consumes ztd
        # unchanged — it is already in the residual currency).
        with torch.no_grad():
            V = np.concatenate([
                head(torch.from_numpy(H[lo:lo + 8192]).to(
                    ai.DEVICE)).squeeze(-1).cpu().numpy()
                for lo in range(0, len(H), 8192)])
        G = _td_lambda_returns(V + phi, z, gid, lam)
        ztd = np.empty(len(H), dtype=np.float32)
        # HUNT-A F4 + SPTR-2: ztd export uses the SAME head-matched
        # clip as the fit above (value_trainer_v2._head_clip)
        ztd[alidx] = np.clip((G - phi), -_clip9,
                             _clip9).astype(np.float32)
        np.save(os.path.join(DS_DIR, f'{tag}_ztd.npy'), ztd)
        # HUNT-A F3: currency marker — a leftover pre-v37 ABSOLUTE
        # ztd is indistinguishable from a residual one by content;
        # the jt stage refuses to train without this sidecar
        with open(os.path.join(DS_DIR, f'{tag}_ztd_meta.json'), 'w',
                  encoding='utf-8') as _zf9:
            json.dump({'currency': 'residual-z-phi',
                       **_checkpoint_contract(),
                       'net_tag': NET_TAG, 'corpus': CORPUS}, _zf9)
        print(f"  [{tag}] wrote {tag}_ztd.npy (RESIDUAL "
              f"lambda-return targets G-phi, aligned)", flush=True)
        out = os.path.join(ROOT, f'wotr_{tag}_corpus_{NET_TAG}td.pt')
        torch.save({**_checkpoint_contract(), 'model_state': net.state_dict(),
                    'input_dim': ai.get_feature_dim(),
                    'action_dim': ai.ACTION_ENCODING_DIM,
                    'arch': 'gnn_v4_full', 'side': tag,
                    'trained_on': f'{CORPUS} value-td '
                                  f'(lam={lam}, rounds={rounds})',
                    'scheme': 'full_menu_v30+value_td',
                    'head_act': _head_act_of(net),
                    'value_head_trained': True}, out)
        print(f"[{tag}] saved {out}", flush=True)
    with open(os.path.join(DS_DIR, 'td_value_report.json'), 'w',
              encoding='utf-8') as f:
        json.dump(report, f, indent=1)
    print('wrote', os.path.join(DS_DIR, 'td_value_report.json'))


def value_anchor(epochs=2, lr=3e-4, n_train=120000, batch=4096):
    """CORPUS-ANCHORED VALUE EPOCHS (v37 smoke-refusal fix; the
    pre-registered anti-erosion response, comms\\25/29).

    Root cause (measured, experiments\\VALUE_corpus_rings\\
    v37_ring_sensitivity.py): the jt stage's lambda-return targets
    (3 self-bootstrap td rounds, lam 0.8) DILUTE outcome-conditional
    structure — fp spend-sensitivity fell -0.0203 (certified BC) ->
    -0.0069 (jt) while the corpus data itself carries the signal in
    BOTH currencies (z-slope on own-rings +0.118, z-phi slope +0.086;
    late-game +0.221/+0.216).  v34's celebrated -0.056 was partly the
    deck-artifact confound (it collapsed to -0.0105 the moment the
    leg-7 staple fix landed, BEFORE the residual currency existed);
    the honest data-supported level is the certified-BC one
    (~-0.02..-0.03 fp).

    Fix: after train-joint, refit the VALUE HEAD ONLY on frozen-trunk
    features toward the residual corpus anchor (z - phi) — restores
    the outcome-conditional structure lambda-smoothing removed while
    keeping jt's policy/trunk gains.  Head BN frozen (1.224e scope).
    Runs on wotr_{tag}_corpus_{NET_TAG}jt.pt in place (atomic save);
    re-run conf-stamp + depth battery afterwards (recipe)."""
    _require_fit_schedule(epochs, lr, batch=batch)
    if type(n_train) is not int or n_train < 1:
        raise ValueError('Corpus anchor fitting requires a positive sample count')
    _require_dataset_contract()
    import torch
    import ai
    # target stage suffix: 'jt' (default) or 'td' (WOTR_ANCHOR_STAGE
    # — option-A recipes anchor the td checkpoints); sides via
    # WOTR_ANCHOR_SIDES (default both; measured: sh benefits at every
    # dose with fixtures green, fp anchoring trades against the
    # mordor lambda-structure — see comms\37)
    _sfx9 = os.environ.get('WOTR_ANCHOR_STAGE', 'jt')
    _sides9 = [s.strip().lower() for s in
               (os.environ.get('WOTR_ANCHOR_SIDES') or
                'fp,sh').split(',') if s.strip()]
    if not _sides9 or any(s not in ('fp', 'sh') for s in _sides9):
        sys.exit('WOTR_ANCHOR_SIDES must be a comma list of fp/sh, '
                 'got %r' % os.environ.get('WOTR_ANCHOR_SIDES'))
    for tag in ('fp', 'sh'):
        if tag not in _sides9:
            continue
        p = os.path.join(ROOT,
                         f'wotr_{tag}_corpus_{NET_TAG}{_sfx9}.pt')
        if not os.path.exists(p):
            print('[value-anchor %s] no %s checkpoint — skipped'
                  % (tag, _sfx9), flush=True)
            continue
        X = np.load(os.path.join(DS_DIR, f'{tag}_X.npy'),
                    mmap_mode='r')
        z = np.asarray(np.load(os.path.join(DS_DIR,
                                            f'{tag}_z.npy'),
                               mmap_mode='r'))
        _php = os.path.join(DS_DIR, f'{tag}_phi.npy')
        if not os.path.exists(_php):
            sys.exit('value-anchor needs %s (v37 schema)' % _php)
        phi = np.asarray(np.load(_php, mmap_mode='r'))
        ck = torch.load(p, map_location='cpu', weights_only=False)
        _require_learning_contract(ck, p)
        net = ai.load_value_net(p, side=tag)
        net.eval()
        n = X.shape[0]
        if n < 1:
            raise ValueError(f'Corpus anchor fitting [{tag}] requires at least one training row')
        rng = np.random.RandomState(SEED)
        rows = np.sort(rng.choice(n, size=min(n_train, n),
                                  replace=False))
        Xtr = np.array(X[rows], dtype=np.float32)
        ttr = (z[rows] - phi[rows]).astype(np.float32)
        print('[value-anchor %s] %d states, %d epochs, lr %g '
              '(RESIDUAL corpus anchor z-phi; head-only, BN frozen)'
              % (tag, len(Xtr), epochs, lr), flush=True)
        H = []
        with torch.no_grad():
            for lo in range(0, len(Xtr), 512):
                hh, _ = net._process_input(
                    torch.from_numpy(Xtr[lo:lo + 512]).to(ai.DEVICE))
                H.append(hh.cpu().numpy().astype(np.float32))
        H = np.concatenate(H)
        head = net.value_head
        opt = torch.optim.Adam(head.parameters(), lr=lr,
                               weight_decay=1e-5)
        _head_train_bn_eval(head)            # 1.224e scope (shared)
        optimizer_steps = 0
        for ep in range(epochs):
            perm = rng.permutation(len(H))
            tot = nb = 0
            for lo in range(0, len(H), batch):
                bi = perm[lo:lo + batch]
                hb = torch.from_numpy(H[bi]).to(ai.DEVICE)
                tb = torch.from_numpy(ttr[bi]).to(ai.DEVICE)
                v = head(hb).squeeze(-1)
                loss = torch.nn.functional.mse_loss(v, tb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                optimizer_steps += 1
                tot += float(loss.detach())
                nb += 1
            print('  %s anchor ep %d: residual-mse %.4f'
                  % (tag, ep, tot / max(1, nb)), flush=True)
        head.eval()
        if not optimizer_steps:
            raise RuntimeError(f'value_anchor[{tag}] completed no optimizer steps; refusing publication')
        ck = torch.load(p, map_location='cpu', weights_only=False)
        ck['model_state'] = {k: v.cpu() for k, v in
                             net.state_dict().items()}
        ck['value_head_trained'] = True
        ck['value_anchor'] = {'epochs': epochs, 'lr': lr,
                              'n': int(len(H)), 'optimizer_steps': optimizer_steps}
        tmp = p + '.tmp'
        torch.save(ck, tmp)
        os.replace(tmp, p)
        print('[value-anchor %s] saved %s' % (tag, p), flush=True)


def conf_stamp():
    """CONFIDENCE STAMP stage (comms\\30 sec 1 contract; R2/R3 glue).

    Per side: fit (or reuse) the RND Familiarity reference on that
    side's OWN consolidated encodings ({tag}_X.npy — the same-encoding
    invariant: exactly what that side's net consumes), persist the
    sha-manifestable artifacts under DS_DIR\\conf_<tag>\\, then stamp
    conf_mech/conf_state/conf_calib/conf_meta into EVERY existing
    checkpoint of this tag+NET_TAG generation ('', 'td', 'jt').
    Idempotent (re-run re-stamps); atomic saves; refits re-run this
    stage so calibration provenance follows the artifact it rides.
    RND is state-referenced (not head-referenced), so base/td/jt of
    one generation share one fitted mechanism per side."""
    _require_dataset_contract()
    import hashlib as _hl
    import time as _time
    import torch
    import confidence as _conf

    def _ds_identity(tag):
        # HUNT-A F5: pin the reference to the EXACT dataset it was
        # fitted on — an aggregate CONF-1 pass cannot prove a
        # mechanism fitted on pre-fix encodings is right at the
        # repaired state classes. Hash every byte in bounded chunks;
        # version the identity so legacy prefix-only receipts refit.
        xp = os.path.join(DS_DIR, f'{tag}_X.npy')
        h = _hl.sha256()
        with open(xp, 'rb') as f:
            for block in iter(lambda: f.read(1 << 20), b''):
                h.update(block)
        return 'sha256-v2:%s' % h.hexdigest()

    for tag in ('fp', 'sh'):
        fam_dir = os.path.join(DS_DIR, 'conf_%s' % tag)
        idp = os.path.join(fam_dir, 'ds_identity.json')
        want_id = _ds_identity(tag)
        cert = None
        fam = None
        try:
            have_id = (json.load(open(idp, encoding='utf-8'))
                       .get('id') if os.path.exists(idp) else None)
            if have_id != want_id:
                print('[conf-stamp %s] artifact ds-identity mismatch '
                      '(%s != %s) — refitting on current data'
                      % (tag, have_id, want_id), flush=True)
                raise ValueError('stale reference')
            fam = _conf.Familiarity.load(fam_dir)
            cert = _conf.conf1_check(fam, DS_DIR, tag)
            if cert.get('pass'):
                print('[conf-stamp %s] reusing CERTIFIED mechanism '
                      'at %s (%s)' % (tag, fam_dir, cert),
                      flush=True)
            else:
                print('[conf-stamp %s] existing artifact FAILS '
                      'CONF-1 (%s) — refitting' % (tag, cert),
                      flush=True)
                fam = None
        except Exception:
            fam = None
        if fam is None:
            # R4 leg-2/4 fix: K-averaged targets + stamp-time CONF-1
            # cert with refit-on-fail (provenance recorded); an
            # uncertified confidence artifact never ships
            fam, cert = _conf.seed_certified(DS_DIR, tag,
                                             base_seed=SEED)
            fam.save(fam_dir)
            with open(idp, 'w', encoding='utf-8') as _if9:
                json.dump({'id': want_id}, _if9)
        meta = {'mech': _conf.CONF_MECH, 'corpus_tag': NET_TAG,
                'ds_dir': DS_DIR, 'side': tag,
                'calib_n': len(fam.calib), 'k': fam.k,
                'seed': SEED, 'conf1_cert': cert,
                'date': _time.strftime('%Y-%m-%d %H:%M:%S')}
        fields = _conf.to_checkpoint_fields(fam, meta)
        n_stamped = 0
        for sfx in ('', 'td', 'jt'):
            p = os.path.join(ROOT,
                             f'wotr_{tag}_corpus_{NET_TAG}{sfx}.pt')
            if not os.path.exists(p):
                continue
            ck = torch.load(p, map_location='cpu',
                            weights_only=False)
            if not isinstance(ck, dict):
                print('[conf-stamp %s] SKIP non-dict ckpt %s'
                      % (tag, p), flush=True)
                continue
            _require_learning_contract(ck, p)
            ck.update(fields)
            tmp = p + '.tmp'
            torch.save(ck, tmp)
            os.replace(tmp, p)
            n_stamped += 1
            print('[conf-stamp %s] stamped %s' % (tag, p),
                  flush=True)
        if n_stamped == 0:
            print('[conf-stamp %s] WARNING: no checkpoints found '
                  'for tag %s' % (tag, NET_TAG), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all',
                    choices=['consolidate', 'train', 'eval', 'value-td',
                             'train-joint', 'conf-stamp',
                             'value-anchor', 'inspect-contracts', 'policy-warmstart', 'all'])
    ap.add_argument('--anchor-epochs', type=int, default=2)
    ap.add_argument('--anchor-lr', type=float, default=3e-4)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--value-weight', type=float, default=0.5)
    ap.add_argument('--td-lambda', type=float, default=0.8)
    ap.add_argument('--td-rounds', type=int, default=3)
    ap.add_argument('--td-epochs', type=int, default=2)
    ap.add_argument('--td-lr', type=float, default=3e-4)
    ap.add_argument('--td-batch', type=int, default=8192)
    ap.add_argument('--jt-epochs', type=int, default=4)
    ap.add_argument('--jt-lr', type=float, default=3e-4)
    ap.add_argument('--fp-seed')
    ap.add_argument('--sh-seed')
    ap.add_argument('--policy-output')
    ap.add_argument('--policy-admission')
    args = ap.parse_args()
    if args.stage == 'policy-warmstart':
        from corpus_policy_warmstart import run
        run(args)
        return
    if args.stage == 'inspect-contracts':
        print(json.dumps(inspect_contracts(), indent=2))
        return
    if args.stage in ('consolidate', 'all'):
        consolidate()
    if args.stage in ('train', 'all'):
        train(args.epochs, args.lr, args.value_weight)
    if args.stage in ('eval', 'all'):
        evaluate()
    if args.stage in ('value-td', 'all'):
        value_td(args.td_lambda, args.td_rounds, args.td_epochs,
                 args.td_lr, args.td_batch)
    if args.stage in ('train-joint', 'all'):
        train(args.jt_epochs, args.jt_lr, args.value_weight, joint=True)
    if args.stage == 'conf-stamp':
        # explicit-only (never inside 'all'): the R-sequence calls it
        # after BC and after every refit (comms\30 R2/R3)
        conf_stamp()
    if args.stage == 'value-anchor':
        # explicit-only, post-jt (v37 smoke-refusal fix); re-run
        # conf-stamp + depth battery afterwards
        value_anchor(args.anchor_epochs, args.anchor_lr)


if __name__ == '__main__':
    main()
