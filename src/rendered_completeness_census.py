# -*- coding: utf-8 -*-
"""RENDERED ALL-TYPE COMPLETENESS CENSUS (gate (f), USER 2026-08-03).

Standing gate: every human action in the rendered text must become
either an EMITTED sample or a COUNTED, CLASSIFIED drop — zero silent
losses.  Two instruments:

  A. PER-BATCH RUNTIME ACCOUNTING (the hard guarantee): re-process
     games read-only with process_batch and every per-batch List-2
     emitter wrapped; a die-spend batch that yields no sample AND
     moves no classification counter (kinds_emit / drop_reasons /
     detected / nomatch / ambiguous / diverged) is a SILENT LOSS,
     recorded with its game, turn, kind and die face.  Ground truth =
     the batch stream itself (one batch per human die-spend in the
     rendered text).

  B. GAME-LEVEL EVENT IDENTITIES (descriptive per-type table): for
     event families detected outside the batch loop (eyes, passes,
     declares, reveals, discards, guides, hunt cards), compare the
     census 'detected' ground truth against emitted samples + the
     family's mapped drop reasons; the residual per family is
     reported (mapping gaps surface as residual, never hidden).

Usage:
  python src/rendered_completeness_census.py --driver --workers 8
  python src/rendered_completeness_census.py --shard 0 --nshards 8
  python src/rendered_completeness_census.py --report   # aggregate only

Output: RCC_OUT_DIR (default data/rendered_completeness_census/)
  rcc-shard<i>.jsonl   per-game accounting lines
  report printed by --report / at driver end
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

RCC_DIR = os.environ.get('RCC_OUT_DIR') or os.path.join(
    ROOT, 'data', 'rendered_completeness_census')

# ---------------------------------------------------------------------------
# instrument A: per-batch accounting via monkeypatch
# ---------------------------------------------------------------------------
_ACCT = {'cur': None, 'rows': []}     # cur batch accounting row


def _counter_total(stats):
    return (sum(stats['kinds_emit'].values())
            + sum(stats['drop_reasons'].values())
            + sum(stats['detected'].values())
            + stats['nomatch'] + stats['ambiguous'] + stats['diverged'])


def _n_samples(ret):
    if ret is None:
        return 0
    if isinstance(ret, list):
        return sum(1 for x in ret if x is not None)
    return 1


def _install(rx):
    orig_pb = rx.process_batch
    per_batch_emitters = [
        '_catalog_unemitted', '_emit_elven_rings', '_emit_token_uses',
        '_emit_card_interiors', '_emit_combat_family',
        '_emit_siege_withdrawal', '_emit_dwarven_rings',
        '_emit_hunt_resolution', '_emit_hunt_cards']

    def pb(batch, stats):
        _ACCT['stats_ref'] = stats     # FULL counters for game_line
        row = {'idx': batch.get('idx'), 'turn': batch.get('turn'),
               'kind': batch.get('kind'), 'face': batch.get('die_face'),
               'side': batch.get('die_side'), 'samples': 0, 'moved': 0,
               'mstep': 0, 'mfel': 0, 'mcard': 0}
        # gate-(f) ADDENDUM (USER): flag MORDOR-TRACK step batches
        # (truth fel.step advance) for the CSV-anchored floor check
        try:
            _fb = (batch.get('truth_before') or {}).get('fel') or {}
            _fa = (batch.get('truth') or {}).get('fel') or {}
            row['mstep'] = int(bool(_fb.get('mordor'))
                               and int(_fa.get('step') or 0)
                               > int(_fb.get('step') or 0))
        except Exception:
            pass
        _ACCT['rows'].append(row)
        _ACCT['cur'] = row
        t0 = _counter_total(stats)
        try:
            ret = orig_pb(batch, stats)
        except Exception:
            row['moved'] += 1     # extract_one counts batch_exc:* itself
            _ACCT['cur'] = None
            raise
        row['samples'] += _n_samples(ret)
        if row['mstep'] and isinstance(ret, dict):
            if ret.get('phase') == 'fellowship_move':
                row['mfel'] = 1   # emitted track-step move sample
            else:
                # card-EFFECT advance: the decision is the card play
                # (play_event sample) — the step itself is no separate
                # decision (probe 2026-08-03: both pilot floor misses
                # were this class)
                row['mcard'] = 1
        # __extra_samples ride stats; count them as samples
        row['samples'] += len(stats.get('__extra_samples') or [])
        row['moved'] += int(_counter_total(stats) - t0)
        return ret

    rx.process_batch = pb

    def wrap_extra(name):
        orig = getattr(rx, name)

        def w(batch, *a, **k):
            row = _ACCT['cur']
            stats = None
            for x in a:
                if isinstance(x, dict) and 'kinds_emit' in x:
                    stats = x
            if k.get('stats') is not None:
                stats = k['stats']
            t0 = _counter_total(stats) if stats is not None else None
            ret = orig(batch, *a, **k)
            if row is not None:
                row['samples'] += _n_samples(ret)
                if t0 is not None:
                    row['moved'] += int(_counter_total(stats) - t0)
            return ret

        setattr(rx, name, w)

    missing = [nm for nm in per_batch_emitters
               if not hasattr(rx, nm)]
    if missing:
        raise RuntimeError('census emitter list stale: %s' % missing)
    for nm in per_batch_emitters:
        wrap_extra(nm)


# families for instrument B: family -> (detected keys, emit keys,
# drop-reason PREFIXES, non-decision DETECTED keys counted with drops).
# Residual = det - emit - drops - nondecision.  DESCRIPTIVE table (the
# hard zero-silent gate is instrument A); requires run-2+ census lines
# (FULL drop counters — run-1's top-15 truncation manufactured
# phantom residuals).
_FAMILIES = {
    'eye_allocate': (['eye_allocate'], ['eye_allocate'], ['eye_'], []),
    'pass': (['pass'], ['pass'], ['pass_'], []),
    'discard_card': (['discard_card'], ['discard_card'], ['discard_'],
                     []),
    'choose_guide': (['choose_guide_real'], ['choose_guide'],
                     ['guide_'], []),
    # det fires AFTER the unsourced gate -> post-det drop keys only
    'reveal_place': (['reveal_place'], ['fellowship_reveal_place'],
                     ['revplace_recon', 'revplace_legal',
                      'revplace_unmatched', 'revplace_forced'], []),
    'declare_windows(gate-e)': (
        ['declare_progress_reset', 'declare_mordor_entry',
         'declare_heal0'],
        ['fellowship_declare'],
        ['declare_', 'healdecl_', 'decline_recon', 'decline_legal',
         'decline_no_pass'], []),
    # exact prefixes: a bare 'ring_' also matched the post-emit
    # ALL-KIND guard counter ring_truth_zero_hold (not a ring drop)
    'elven_ring': (['elven_ring', 'elven_ring_implied'], ['elven_ring'],
                   ['ring_face', 'ring_src_', 'ring_no_counter',
                    'ring_recon', 'ring_staples', 'ring_legal',
                    'ring_unmatched', 'ring_implied_ambiguous'], []),
    'dwarven_ring': (['dwarven_ring_use'], ['dwarven_ring'],
                     ['dwring_'], []),
    # offer-based: det = the card-OFFER census (axebow/horn share one
    # counter); emit = the parent counters (each already sums its
    # use/decline split)
    'hunt_cards': (['huntcard_axebow_horn', 'huntcard_mithril',
                    'huntcard_wizardstaff'],
                   ['hunt_reduction', 'mithril', 'wizardstaff'],
                   # horn shares the axebow_ drop counters
                   ['axebow_', 'mithril_', 'wizard_'],
                   []),
}


def game_line(stem, line, secs):
    rows = _ACCT['rows']
    silent = [r for r in rows if r['samples'] == 0 and r['moved'] == 0]
    by_kind = Counter(r['kind'] for r in rows)
    by_kind_silent = Counter(r['kind'] for r in silent)
    by_kind_sampled = Counter(r['kind'] for r in rows if r['samples'])
    # FULL drop/detected counters (the manifest line truncates
    # drop_by_reason to most_common(15) — that truncation manufactured
    # a phantom 1,196-game mordor-entry 'gap' in census run 1)
    sref = _ACCT.get('stats_ref')
    if line is not None and sref is not None:
        line = dict(line)
        line['drop_by_reason'] = dict(sref['drop_reasons'])
        line['detected'] = dict(sref['detected'])
        line['emit_by_kind'] = dict(sref['kinds_emit'])
    return {
        'game': stem, 'batches': len(rows),
        'silent': len(silent),
        'mordor_steps_gt': sum(r.get('mstep', 0) for r in rows),
        'mordor_steps_emitted': sum(r.get('mfel', 0) for r in rows),
        'mordor_steps_card': sum(r.get('mcard', 0) for r in rows),
        'silent_detail': [{'idx': r.get('idx'), 'turn': r['turn'],
                           'kind': r['kind'],
                           'face': r['face'], 'side': r['side']}
                          for r in silent[:20]],
        'by_kind': dict(by_kind),
        'by_kind_silent': dict(by_kind_silent),
        'by_kind_sampled': dict(by_kind_sampled),
        'emit_by_kind': (line or {}).get('emit_by_kind'),
        'detected': (line or {}).get('detected'),
        'drop_by_reason': (line or {}).get('drop_by_reason'),
        'quarantined': (line or {}).get('quarantined'),
        'error': (line or {}).get('error'),
        'secs': secs,
    }


def run_shard(shard, nshards, limit=None):
    import rendered_extract as rx
    _install(rx)
    os.makedirs(RCC_DIR, exist_ok=True)
    games = list(rx._game_iter('corpus'))[shard::nshards]
    if limit:
        games = games[:limit]
    outp = os.path.join(RCC_DIR, 'rcc-shard%d.jsonl' % shard)
    # GLOBAL resume set: skip games any shard file already processed
    # (lets the driver be restarted with a different worker count)
    done = set()
    for fn in os.listdir(RCC_DIR):
        if not fn.startswith('rcc-shard') or not fn.endswith('.jsonl'):
            continue
        with open(os.path.join(RCC_DIR, fn), encoding='utf-8') as f:
            for ln in f:
                try:
                    done.add(json.loads(ln)['game'])
                except Exception:
                    pass
    for path, stem, meta in games:
        if stem in done:
            continue
        _ACCT['rows'] = []
        _ACCT['cur'] = None
        _ACCT['stats_ref'] = None
        t0 = time.time()
        line = None
        err = None
        try:
            _s, line, _cov = rx.extract_one(path, stem, meta)
        except rx._Quarantine as q:
            line = {'quarantined': q.reason}
        except KeyboardInterrupt:
            raise
        except BaseException as e:
            err = ('%s: %s' % (type(e).__name__, e))[:160]
        gl = game_line(stem, line, round(time.time() - t0, 1))
        if err:
            gl['error'] = err
        with open(outp, 'a', encoding='utf-8') as f:
            f.write(json.dumps(gl) + '\n')
    print('[rcc shard %d] DONE' % shard, flush=True)


def _csv_mordor_positions():
    """stem -> CSV mordor-track position N (int) for base-game rows
    with passwords (same population corpus_rows admits)."""
    import csv
    out = {}
    path = os.path.join(ROOT, 'data', 'gameReports_with_passwords.csv')
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if not (r.get('free_password') and r.get('shadow_password')):
                continue
            exp = (r.get('expansions') or '').strip()
            if exp and exp.lower() != 'none':
                continue
            m = (r.get('mordor') or '').strip()
            if not m.isdigit():
                continue
            name = os.path.basename(r.get('log_file') or '')
            stem = name[:-4] if name.endswith('.log') else name
            out.setdefault(stem, int(m))
    return out


def aggregate_report():
    tot = Counter()
    kind_tot = Counter()
    kind_silent = Counter()
    kind_sampled = Counter()
    emit = Counter()
    det = Counter()
    drops = Counter()
    games = 0
    silent_games = []
    mordor_rows = []      # (game, gt_track_steps, emitted_track_steps(fel+card), fel_move_total)
    for fn in os.listdir(RCC_DIR):
        if not fn.startswith('rcc-shard'):
            continue
        with open(os.path.join(RCC_DIR, fn), encoding='utf-8') as f:
            for ln in f:
                try:
                    d = json.loads(ln)
                except Exception:
                    continue
                games += 1
                if d.get('quarantined'):
                    # a quarantined game is excluded from the corpus
                    # wholesale — that exclusion IS its classification;
                    # its batches leave the hard-gate denominator.
                    tot['quarantined'] += 1
                    continue
                if d.get('error'):
                    tot['errors'] += 1
                    continue
                tot['batches'] += d.get('batches', 0)
                tot['silent'] += d.get('silent', 0)
                for k, c in (d.get('by_kind') or {}).items():
                    kind_tot[k] += c
                for k, c in (d.get('by_kind_silent') or {}).items():
                    kind_silent[k] += c
                for k, c in (d.get('by_kind_sampled') or {}).items():
                    kind_sampled[k] += c
                for k, c in (d.get('emit_by_kind') or {}).items():
                    emit[k] += c
                for k, c in (d.get('detected') or {}).items():
                    det[k] += c
                for k, c in (d.get('drop_by_reason') or {}).items():
                    drops[k] += c
                if d.get('silent', 0) > 0:
                    silent_games.append((d['silent'], d['game'],
                                         d.get('silent_detail')))
                mordor_rows.append(
                    (d['game'], d.get('mordor_steps_gt', 0),
                     d.get('mordor_steps_emitted', 0)
                     + d.get('mordor_steps_card', 0),
                     (d.get('emit_by_kind') or {}).get(
                         'fellowship_move', 0)))
    silent_games.sort(key=lambda t: (-t[0], t[1]))
    print('=== GATE (f): RENDERED ALL-TYPE COMPLETENESS CENSUS ===')
    print('games=%d quarantined=%d errors=%d' % (
        games, tot['quarantined'], tot['errors']))
    print('\n--- A. per-batch accounting (hard gate) ---')
    print('die-spend batches=%d  SILENT (no sample, no counter)=%d '
          '(%.4f%%)' % (tot['batches'], tot['silent'],
                        100.0 * tot['silent'] / max(1, tot['batches'])))
    print('%-10s %9s %9s %9s %9s' % ('kind', 'batches', 'sampled',
                                     'clsf-only', 'SILENT'))
    for k in sorted(kind_tot):
        print('%-10s %9d %9d %9d %9d'
              % (k, kind_tot[k], kind_sampled.get(k, 0),
                 kind_tot[k] - kind_sampled.get(k, 0)
                 - kind_silent.get(k, 0), kind_silent.get(k, 0)))
    if silent_games:
        print('\nworst offender games (silent batches):')
        for n, g, detail in silent_games[:10]:
            print('  %3d  %s' % (n, g))
            for r in (detail or [])[:4]:
                print('       t%s %s %s %s' % (r.get('turn'),
                                               r.get('kind'),
                                               r.get('face'),
                                               r.get('side')))
    print('\n--- B. game-level event identities ---')
    print('%-24s %9s %9s %9s %9s %9s' % ('family', 'detected',
                                         'emitted', 'drops', 'nondec',
                                         'residual'))
    for fam, (dks, eks, dps, nds) in _FAMILIES.items():
        dv = sum(det.get(k, 0) for k in dks)
        ev = sum(emit.get(k, 0) for k in eks)
        rv = sum(c for k, c in drops.items()
                 if any(k.startswith(p) for p in dps))
        nv = sum(det.get(k, 0) for k in nds)
        print('%-24s %9d %9d %9d %9d %9d'
              % (fam, dv, ev, rv, nv, dv - ev - rv - nv))
    # ---- gate-(f) ADDENDUM: CSV-anchored Mordor-track floor ----------
    print('\n--- C. CSV-anchored fellowship-movement lower bound ---')
    csv_n = _csv_mordor_positions()
    deltas = Counter()
    fails = []
    n_mgames = 0
    for game, gt, emitted, fel_total in mordor_rows:
        N = csv_n.get(game)
        if not N or N < 1:
            continue
        n_mgames += 1
        deltas[emitted - N] += 1
        if emitted < N:
            fails.append((game, N, emitted, gt, fel_total))
    print('admitted games with CSV mordor position N>=1: %d' % n_mgames)
    print('(track-moves-emitted - N) distribution: %s'
          % dict(sorted(deltas.items())))
    print('games BELOW the floor (SUSPECTED SILENT DROPS): %d'
          % len(fails))
    for game, N, emitted, gt, fel_total in fails[:25]:
        print('  FAIL %-46s N=%d emitted_track=%d gt_track_steps=%d '
              'fel_move_total=%d' % (game[:46], N, emitted, gt,
                                     fel_total))
    # pre-Mordor sanity: total fellowship_move >= min map distance
    # Rivendell -> nearest gateway (progress is earned 1/move)
    try:
        import board as _bd
        b = _bd.load_board()
        seen = {'rivendell'}
        frontier = {'rivendell'}
        dist = None
        for step in range(1, 30):
            frontier = {str(a).lower() for r in frontier
                        for a in b.adjacent(r)} - seen
            if not frontier:
                break
            if 'morannon' in frontier or 'minas morgul' in frontier:
                dist = step
                break
            seen |= frontier
        if dist:
            absurd = [(g, ft) for g, gt, em, ft in mordor_rows
                      if csv_n.get(g, 0) >= 1 and ft < dist]
            print('min map distance rivendell->gateway: %d; mordor '
                  'games with total fellowship_move < that: %d'
                  % (dist, len(absurd)))
            for g, ft in absurd[:10]:
                print('  ABSURD-SHORTFALL %-46s fel_move_total=%d'
                      % (g[:46], ft))
    except Exception as e:
        print('pre-mordor sanity skipped: %s' % str(e)[:60])
    # drop classification completeness: every drop reason must belong
    # to a family prefix or be listed here (nothing unclassified)
    all_pref = [p for _, (_, _, dps, _n) in _FAMILIES.items()
                for p in dps]
    other = Counter({k: c for k, c in drops.items()
                     if not any(k.startswith(p) for p in all_pref)})
    print('\n--- D. drop reasons outside the family map '
          '(classified, batch-attributed) ---')
    for k, c in other.most_common(30):
        print('  %-40s %d' % (k, c))
    return tot


def run_driver(workers, limit=None):
    os.makedirs(RCC_DIR, exist_ok=True)
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK='TRUE',
               PYTHONIOENCODING='utf-8', PYTHONHASHSEED='0',
               CUDA_VISIBLE_DEVICES='-1')
    procs = []
    for i in range(workers):
        logf = open(os.path.join(RCC_DIR, 'rcc-worker%d.log' % i), 'a',
                    encoding='utf-8', errors='replace')
        cmd = [sys.executable, '-u', os.path.abspath(__file__),
               '--shard', str(i), '--nshards', str(workers)]
        if limit:
            cmd += ['--limit', str(limit)]
        procs.append((subprocess.Popen(cmd, stdout=logf,
                                       stderr=subprocess.STDOUT,
                                       env=env, cwd=ROOT), logf))
        print('rcc worker %d spawned (pid %d)' % (i, procs[-1][0].pid),
              flush=True)
    bad = []
    for wi, (p, logf) in enumerate(procs):
        rc = p.wait()
        logf.close()
        if rc:
            bad.append((wi, rc))
    for wi, rc in bad:
        print('WORKER %d EXITED rc=%d - census INCOMPLETE' % (wi, rc),
              flush=True)
    aggregate_report()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--driver', action='store_true')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--shard', type=int, default=None)
    ap.add_argument('--nshards', type=int, default=None)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--report', action='store_true')
    args = ap.parse_args()
    if args.report:
        aggregate_report()
    elif args.driver:
        run_driver(args.workers, args.limit)
    elif args.shard is not None:
        run_shard(args.shard, args.nshards or 1, args.limit)
    else:
        main_err = 'pick --driver, --shard I --nshards K, or --report'
        raise SystemExit(main_err)


if __name__ == '__main__':
    main()
