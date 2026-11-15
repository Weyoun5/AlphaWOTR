"""Shared corpus utilities (curated game list + npz packing).

TEACHER RETIRED (USER order 2026-07-27): the closed-loop raw-log
extraction that lived here is deleted — the rendered pipeline
(rendered_extract.py) is the sole extraction path.  This module
keeps only the shared utilities it imports: corpus_rows() (the
curated game list), stem_of(), pack_npz().

corpus_rows() invariant: base game only (no expansions), both
passwords present, log on disk, first-CSV-row-wins dedupe, and
corpus_curation exclusions applied unless CORPUS_NO_CURATION=1.
"""
import csv
import json
import os
import sys

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
# insert only when absent (one copy per process; src/ module names are unique on sys.path)
if SRC not in sys.path:
    sys.path.insert(0, SRC)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

DATA = os.path.join(ROOT, 'data')
LOGS_DIR = os.path.join(DATA, 'game_logs')
CSV_PATH = os.path.join(DATA, 'gameReports_with_passwords.csv')


def corpus_rows():
    """Base-game (no expansions) rows with a log on disk; deduped."""
    have = set(os.listdir(LOGS_DIR))
    rows = {}
    with open(CSV_PATH, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            # both passwords present + no expansions (mirrored by
            # corpus_curation._population_stems — keep in sync)
            if not (r.get('free_password') and r.get('shadow_password')):
                continue
            exp = r.get('expansions', '').strip()
            if exp and exp.lower() != 'none':
                continue  # engine models the base game only
            name = os.path.basename(r['log_file'])
            if name in have and name not in rows:
                rows[name] = {'winner_side': r['side'],
                              'victory': r['victory'],
                              'turns': r['turns']}
    # v22 curation: drop duplicate /
    # misjoined / mistake / undecodable-span games.  See corpus_curation
    # module docstring; CORPUS_NO_CURATION=1 restores the raw population.
    if not os.environ.get('CORPUS_NO_CURATION'):
        from corpus_curation import exclusion_map
        excl = exclusion_map()
        rows = {n: m for n, m in rows.items()
                if stem_of(n) not in excl}
    return rows


def stem_of(name):
    return os.path.splitext(name)[0]


class AliasedActionUnderV41Error(RuntimeError):
    """W4-032 STAGE-1 GUARD (sponsor condition 1, 2026-09-03): a sample
    stamped alias_v41 trains a Saruman Muster-die ability action (USER
    rulings 1.266 / 1.266-b / 1.270) that the v41 / v42 action encoder
    encodes byte-identically to the plain Orthanc recruit - the v42 nets
    cannot tell the ability from the plain Orthanc recruit; priors
    identical; disambiguated by the v43 encoder line.  Packing such a row
    into a TRAINING corpus under a pre-v43 tag raises this error - never a
    silent skip, never a filter of the legal surface.  The 1:1 certificate
    tool acknowledges explicitly (allow_v41_alias=True) and prints the
    count."""


def act_gen_below_v43():
    """True while the action-encoding generation (WOTR_ACT_GEN, ai.py's
    single generation switch) is below the v43 tag: '' (un-pinned 92-dim),
    'v41' (live 93-dim) and 'v42' are below; 'v43' and later are not."""
    tag = os.environ.get('WOTR_ACT_GEN', '').strip().lower()
    digits = ''.join(ch for ch in tag if ch.isdigit())
    if not digits:
        return True
    return int(digits) < 43


def saruman_encoding_distinct():
    """S-001 (2026-09-04): True once ai.encode_action tells the Saruman
    ability actions from their plain Orthanc twins (ai.ACT_ENC_SARUMAN_DISTINCT,
    pinned functionally) - the CAPABILITY the W4-032 alias guard waits for.
    Read at pack time from the loaded engine module; a tree without the
    encoder line answers False and the guard stays armed whatever the tag
    (act_gen_below_v43 remains the generation test for the v43 bump)."""
    try:
        import ai
    except Exception:
        return False
    return bool(getattr(ai, 'ACT_ENC_SARUMAN_DISTINCT', False))


class AliasedActionUnderV53Error(RuntimeError):
    """R3-V53-FOLLOWONS packing guard (USER D5 2026-10-22, encoding
    generation v53).  A sample stamped alias_v53 trains a row whose menu
    holds a byte-identical encoding twin (same row, same region) that the
    v53 action tail tells apart: other figures moved (E5-01), another
    companion ability (A1-01) or another Through a Day and a Night route
    (E3-02).  Under an encoder WITHOUT the matching ai.ACT_ENC_* capability
    that merge is the known pre-v53 behaviour (the trainer's multi-positive
    CE treats the twins as one target; pack_npz announces and records it).
    Under an encoder that CLAIMS the capability it is an encoder regression
    and the pack is refused - never a silent skip, never a menu filter."""


# alias class -> the ai capability flag that makes those rows encode distinctly
V53_ALIAS_CLASSES = (('A1-01', 'ACT_ENC_COMPANION_ADVANCE_DISTINCT'),
                     ('E5-01', 'ACT_ENC_ARMY_FIGURES_DISTINCT'),
                     ('E3-02', 'ACT_ENC_TDAAN_ROUTE_DISTINCT'))
CAPABILITY_FLAGS = ('OBS_HAND_FULLY_VISIBLE', 'OBS_THREAT_COUNTS_EXACT',
                    'ACT_ENC_COMPANION_ADVANCE_DISTINCT',
                    'ACT_ENC_ARMY_FIGURES_DISTINCT',
                    'ACT_ENC_TDAAN_ROUTE_DISTINCT', 'ACT_ENC_SARUMAN_DISTINCT')


def encoding_capabilities():
    """The loaded engine's encoder capability flags (recorded in every
    packed npz; read by the alias guards)."""
    import ai
    return {name: bool(getattr(ai, name, False)) for name in CAPABILITY_FLAGS}


def v53_alias_classes(ai, state, side, legal, hidx, encs, regs):
    """Sorted alias classes ('A1-01' / 'E5-01' / 'E3-02') in which the
    chosen row legal[hidx] has a byte-identical encoding twin (same encs
    row, same region index) whose encoding_v53 action tail differs - the
    rows the loaded encoder merges.  Always [] under v53: the tail is part
    of the encoding there, so twins share it."""
    from encoding_v53 import TAIL_ACTION_TYPES, action_tail
    chosen = legal[hidx]
    if chosen.action_type not in TAIL_ACTION_TYPES:
        return []
    E = np.asarray(encs)
    R = np.asarray(regs)
    twins = [int(j) for j in np.flatnonzero((E == E[hidx]).all(axis=1) & (R == R[hidx]))
             if int(j) != int(hidx) and legal[int(j)].action_type == chosen.action_type]
    if not twins:
        return []
    mine = action_tail(ai, chosen, state, side)
    if any(not np.array_equal(mine, action_tail(ai, legal[j], state, side)) for j in twins):
        return [{'advance_political': 'A1-01', 'army_move': 'E5-01',
                 'victor_advance': 'E5-01', 'card_option': 'E3-02'}[chosen.action_type]]
    return []


def pack_npz(samples, path, extras=None, allow_v41_alias=False):
    excluded = [i for i, s in enumerate(samples) if s.get('ai_policy_exclusion')]
    if excluded:
        raise ValueError('Fixed-policy/off-policy rows must be filtered before packing: %s'
                         % excluded[:8])
    # W4-032 stage-1 guard (see AliasedActionUnderV41Error); S-001 lifts
    # it by CAPABILITY - the encoder line - never by tag alone
    _alias_idx = [i for i, s in enumerate(samples) if s.get('alias_v41')]
    if (_alias_idx and not saruman_encoding_distinct()
            and not allow_v41_alias):
        raise AliasedActionUnderV41Error(
            'ALIAS_V41: %d sample(s) train a Saruman ability action the '
            '%s action encoder aliases with the plain Orthanc recruit '
            '(rows %s%s; phases %s) - refusing to pack a training corpus '
            'before the v43 encoder line; the 1:1 certificate tool passes '
            'allow_v41_alias=True'
            % (len(_alias_idx),
               os.environ.get('WOTR_ACT_GEN', '').strip() or 'un-pinned',
               _alias_idx[:8], '...' if len(_alias_idx) > 8 else '',
               sorted(set(str(samples[i].get('phase')) for i in _alias_idx))))
    # R3-V53-FOLLOWONS: the v53 alias guard reads the capability flags -
    # a stamped class the loaded encoder claims to distinguish is an encoder
    # regression (refused); without the capability the merge is the known
    # pre-v53 behaviour: announced here, recorded per sample in the npz
    _caps9 = encoding_capabilities()
    _flag9 = dict(V53_ALIAS_CLASSES)
    _v53_alias = [(i, tuple(s.get('alias_v53') or ())) for i, s in enumerate(samples)
                  if s.get('alias_v53')]
    _lying9 = [(i, c) for i, classes in _v53_alias for c in classes if _caps9[_flag9[c]]]
    if _lying9:
        raise AliasedActionUnderV53Error(
            'ALIAS_V53: %d sample(s) carry a byte-identical encoding twin in a '
            'class the loaded encoder claims to encode distinctly (%s) - '
            'refusing to pack: the action encoder regressed'
            % (len(_lying9), _lying9[:8]))
    if _v53_alias:
        print('[ALIAS_V53] %d sample(s) merged with encoding twins by the %s '
              'encoder (classes %s; no v53 capability) - recorded in the npz '
              "'alias_v53' array; the D6 corpus is extracted under WOTR_ACT_GEN=v53"
              % (len(_v53_alias), os.environ.get('WOTR_ACT_GEN', '').strip() or 'un-pinned',
                 sorted(set(c for _, classes in _v53_alias for c in classes))), flush=True)
    N = len(samples)
    D = samples[0]['x'].shape[0]
    A = samples[0]['encs'].shape[1]
    X = np.zeros((N, D), dtype=np.float32)
    side = np.zeros(N, dtype=np.uint8)
    human = np.zeros(N, dtype=np.int32)
    z = np.zeros(N, dtype=np.int8)
    turn = np.zeros(N, dtype=np.int16)
    # v37 CURRENCY-COHERENCE schema: per-sample side-signed profile
    # shaping (see rendered_extract._emit_sample).  REQUIRED — a
    # sample without phi is a schema violation and must fail loudly
    # (never a silent zero: that would silently re-create the
    # absolute-currency training this field exists to end).
    phi = np.zeros(N, dtype=np.float32)
    offsets = np.zeros(N + 1, dtype=np.int64)
    M = sum(len(s['encs']) for s in samples)
    cand_encs = np.zeros((M, A), dtype=np.float32)
    cand_regs = np.zeros(M, dtype=np.int32)
    # 1.268 (COORD #2 2026-09-03): per-sample ALTERNATE positive candidate
    # indices (a ring take whose K enabling faces are equally right) -
    # flat alt_idx + alt_offsets (N + 1), empty for every other sample;
    # train_corpus._menu_rows unions them into the positive set.
    alt_offsets = np.zeros(N + 1, dtype=np.int64)
    alt_idx = np.zeros(sum(len(s.get('alt') or ()) for s in samples),
                       dtype=np.int32)
    apos = 0
    pos = 0
    for i, s in enumerate(samples):
        X[i] = s['x']
        side[i] = 0 if s['side'] == 'fp' else 1
        human[i] = s['human']
        z[i] = int(s['z'])
        turn[i] = s['turn']
        phi[i] = float(s['phi'])          # KeyError = loud schema fail
        k = len(s['encs'])
        cand_encs[pos:pos + k] = s['encs']
        cand_regs[pos:pos + k] = s['regs']
        pos += k
        offsets[i + 1] = pos
        _al = [int(v) for v in (s.get('alt') or ())]
        alt_idx[apos:apos + len(_al)] = _al
        apos += len(_al)
        alt_offsets[i + 1] = apos
    tmp = path + '.tmp.npz'
    # v32 gate-(e): optional sidecar arrays (declare-coverage records)
    # ride the same npz under their own keys; every existing consumer
    # reads by name and is unaffected.
    np.savez_compressed(tmp, X=X, side=side, human=human, z=z, turn=turn,
                        decision_time=np.asarray([s.get('decision_time') or (-1, -1) for s in samples], dtype=np.int64),
                        chronology_contract=np.asarray([str(s.get('chronology_contract', 'unproven')) for s in samples]),
                        phi=phi, offsets=offsets, cand_encs=cand_encs,
                        cand_regs=cand_regs, alt_offsets=alt_offsets,
                        alt_idx=alt_idx,
                        action_encoding_contract=np.asarray([str(s.get('action_encoding_contract', 'legacy'))
                                                             for s in samples]),
                        decision_phase=np.asarray([str(s.get('phase', ''))
                                                   for s in samples]),
                        ai_policy_version=np.asarray([str(s.get('ai_policy_version', 'legacy'))
                                                      for s in samples]),
                        policy_target_projection=np.asarray([
                            json.dumps(s.get('policy_target_projection'), sort_keys=True)
                            for s in samples]),
                        observation_contract=np.asarray([str(s.get('observation_contract', 'legacy'))
                                                         for s in samples]),
                        encoding_capabilities=np.asarray(json.dumps(_caps9, sort_keys=True)),
                        alias_v53=np.asarray([','.join(s.get('alias_v53') or ()) for s in samples]),
                        **(extras or {}))
    os.replace(tmp, path)
    return len(_alias_idx)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        sys.exit('extract_corpus.py no longer takes flags; extraction '
                 'lives in rendered_extract.py')
    rows = corpus_rows()
    print('curated corpus: %d games' % len(rows))
