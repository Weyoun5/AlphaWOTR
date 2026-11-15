"""Standing corpus-curation filter (introduced for the v22
re-extraction, session 33 close-out; applied by extract_corpus.
corpus_rows() to EVERY curated extraction since).

Builds the exclusion map that turns the raw base-game population into
the CURATED extraction population approved by the user (2026-07-18):
base games minus

  variant        log's own variant token is not base2 (decodability.json;
                 catches CSV rows mislabeled as expansion-free)
  checksumless   >3 checksum-less real-play lines: the card decoder
                 cannot decode those spans, so card identity (and every
                 hand-composition feature downstream) has holes
  lome-deck      LoME deck-size signature (28/26-card decks vs base
                 24x4) on a base2-labeled row -- USER ruling t138:
                 "only look at BASE GAMES"
  bad-filename   trailing-space stems (Windows path stripping makes
                 them unverifiable by the client-replay axis)
  md5-dup        byte-identical log uploaded under another name; the
                 first stem (sorted) that is IN the extraction
                 population (CSV row + both passwords) stays -- first
                 sorted overall if none is -- the rest are excluded
  outcome-identity  ALL comparable fields of the parsed outcome
                 disagree with the game's CSV report row (victory,
                 side, turns, corruption, mordor, strongholds) -- the
                 log on disk is almost certainly not the game the row
                 describes, so the z label (CSV winner) is untrustworthy
  mistake-game   rulings 1.164/1.165: games whose logs contain the
                 box-to-box garrison-transfer / marker-kill human
                 mistakes ("those are mistakes, do not implement
                 those") -- the un-mirrored mistake corrupts every
                 downstream board state, so the whole game is dropped
  scan-error     the decodability scan recorded an error for the stem
                 (undecodable log)

Run as a script for the census report:
  python src/corpus_curation.py
"""
import hashlib
import json
import os

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
DATA = os.path.join(ROOT, 'data')
LOGS_DIR = os.path.join(DATA, 'game_logs')
RV_DIR = os.path.join(DATA, 'replay_verify')
DECODE_JSON = os.path.join(RV_DIR, 'decodability.json')
OUTCOME_JSONL = os.path.join(RV_DIR, 'outcome_check.jsonl')

# Rulings 1.164 (Q7 stronghold-box->stronghold-box garrison transfers)
# and 1.165 (Q8 kill grammars aimed at fellowship markers): named in
# data/rules_amendments.md; stems resolved against data/game_logs.
MISTAKE_STEMS = {
    '2023-11-03_181333_FP_GJB_SP_RompSteel':        'mistake-game:1.165',
    '2024-01-21_154200_FP_StephanoKing_SP_Ace_xD':  'mistake-game:1.165',
    '2024-05-12_192623_FP_MrM0th_SP_Ace_xD':        'mistake-game:1.164',
    '2024-05-13_225019_FP_MrM0th_SP_Ace_xD':        'mistake-game:1.164',
    '2025-01-19_174820_FP_Elladan_SP_igforce77':    'mistake-game:1.165',
    '2025-03-15_171840_FP_Adobim_SP_estolfo':       'mistake-game:1.164',
    '2025-04-19_121854_FP_krechet_SP_Skaarj':       'mistake-game:1.165',
    '2025-04-22_225557_FP_Rizino1_SP_gus833':       'mistake-game:1.164',
    '2025-12-22_212102_FP_Stella Rossa_SP_Skaarj':  'mistake-game:1.164',
}

_cached = None


def _population_stems():
    """Stems eligible for extraction: CSV row with both passwords, base
    game, log on disk (mirrors extract_corpus.corpus_rows()'s filter;
    duplicated here because extract_corpus imports THIS module)."""
    import csv
    csv_path = os.path.join(DATA, 'gameReports_with_passwords.csv')
    have = set(os.listdir(LOGS_DIR))
    stems = set()
    with open(csv_path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if not (r.get('free_password') and r.get('shadow_password')):
                continue
            exp = r.get('expansions', '').strip()
            if exp and exp.lower() != 'none':
                continue
            name = os.path.basename(r['log_file'])
            if name in have:
                stems.add(os.path.splitext(name)[0])
    return stems


def _md5_dup_losers():
    """Stems of byte-identical logs beyond the kept one of each group.
    Keeper = first sorted stem IN the extraction population (a
    population-blind first-sorted keeper orphaned games whose only
    CSV-population copy sorted later); first sorted overall if none."""
    pop = _population_stems()
    by_hash = {}
    for fn in sorted(os.listdir(LOGS_DIR)):
        if not fn.lower().endswith('.log'):
            continue
        path = os.path.join(LOGS_DIR, fn)
        h = hashlib.md5(open(path, 'rb').read()).hexdigest()
        by_hash.setdefault(h, []).append(fn[:-4])
    losers = {}
    for stems in by_hash.values():
        in_pop = [s for s in stems if s in pop]
        keeper = in_pop[0] if in_pop else stems[0]
        for s in stems:
            if s != keeper:
                losers[s] = 'md5-dup:' + keeper[:40]
    return losers


def _outcome_identity_suspects():
    """Stems whose parsed outcome mismatches the CSV row on EVERY
    comparable field (the outcome_check all-fields-mismatch set)."""
    out = {}
    if not os.path.exists(OUTCOME_JSONL):
        raise RuntimeError('outcome_check.jsonl missing (%s) -- the '
                           'outcome-identity exclusion class would '
                           'silently vanish' % OUTCOME_JSONL)
    bad = 0
    with open(OUTCOME_JSONL, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                bad += 1
                continue
            p, c = r.get('parser') or {}, r.get('csv') or {}
            if not p or not c:
                continue
            n_cmp = n_miss = 0
            for k in ('victory', 'side', 'turns', 'corruption', 'mordor'):
                pv, cv = p.get(k), c.get(k)
                if pv is None or cv is None:
                    continue
                n_cmp += 1
                if str(pv) != str(cv):
                    n_miss += 1
            ps, cs = p.get('strongholds'), c.get('strongholds')
            if ps is not None and cs is not None:
                n_cmp += 1
                if set(ps) != set(cs):
                    n_miss += 1
            if n_cmp >= 4 and n_miss == n_cmp:
                _stem = r.get('stem')
                if _stem:
                    out[_stem] = 'outcome-identity'
    if bad:
        raise RuntimeError('outcome_check.jsonl: %d unparseable line(s)'
                           ' -- file corrupt/truncated' % bad)
    return out


def exclusion_map(refresh=False):
    """{stem: reason} for every curation-excluded stem (cached)."""
    global _cached
    if _cached is not None and not refresh:
        return _cached
    excl = {}

    try:
        decode = json.load(open(DECODE_JSON, encoding='utf-8'))
    except Exception as e:
        raise RuntimeError('decodability.json missing/unreadable (%s): '
                           '%r -- the variant/checksumless/lome-deck/'
                           'scan-error exclusion classes would silently '
                           'vanish' % (DECODE_JSON, e))
    for stem, rec in decode.items():
        if stem != stem.rstrip():
            excl[stem] = 'bad-filename'
            continue
        if 'error' in rec:
            excl.setdefault(stem, 'scan-error')
            continue
        if rec.get('variant', 'base2') != 'base2':
            excl.setdefault(stem, 'variant:' + str(rec.get('variant')))
        elif rec.get('checksumless', 0) > 3:
            excl.setdefault(stem, 'checksumless')
        elif any(n in (26, 28)
                 for key in ('fp_cracked', 'sa_cracked')
                 for n in (rec.get(key) or [])):
            excl.setdefault(stem, 'lome-deck')

    for stem, reason in _md5_dup_losers().items():
        excl.setdefault(stem, reason)
    for stem, reason in _outcome_identity_suspects().items():
        excl.setdefault(stem, reason)
    for stem, reason in MISTAKE_STEMS.items():
        excl[stem] = reason

    _cached = excl
    return excl


def reason_counts(excl=None):
    counts = {}
    for reason in (excl or exclusion_map()).values():
        key = reason.split(':', 1)[0]
        counts[key] = counts.get(key, 0) + 1
    return counts


def _report():
    excl = exclusion_map()
    print('curation exclusions: %d stems' % len(excl))
    for key, n in sorted(reason_counts(excl).items()):
        print('  %-18s %d' % (key, n))
    subj = sorted(s for s, r in excl.items() if r == 'outcome-identity')
    print('outcome-identity suspects (%d):' % len(subj))
    for s in subj:
        print('  ' + s.encode('ascii', 'backslashreplace').decode())
    dups = sorted((s, r) for s, r in excl.items()
                  if r.startswith('md5-dup'))
    print('md5-dup losers (%d):' % len(dups))
    for s, r in dups:
        print(('  %s  <= %s' % (s, r[8:])).encode(
            'ascii', 'backslashreplace').decode())


if __name__ == '__main__':
    _report()
