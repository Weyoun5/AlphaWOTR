"""Per-game VASSAL card-commitment passwords (t127).

data/gameReports_with_passwords.csv is the ladder game-report dump: one
row per reported game with the S3 URL of the exact log file plus BOTH
players' card-deck passwords (free_password / shadow_password) and the
reported outcome metadata (winner/side/victory/turns/corruption/mordor/
strongholds/...).  This module parses it ONCE (lazily, thread-safe) and
keys it by log STEM (basename of log_file without '.log'), which matches
data/game_logs/<stem>.log.

API:
  get_row(stem)        -> the full CSV row dict, or None
  get_passwords(stem)  -> (fp_password, sa_password); None-safe: empty /
                          missing values fall back to the client defaults
                          'lembas' (FP) / 'free' (SA)

Never raises on a missing/broken CSV: falls back to defaults so callers
can wire this in unconditionally.

Run as a script for the join-coverage report:
  python wotr_bridge/game_passwords.py
"""
import csv
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, '..'))
CSV_PATH = os.path.join(PROJ, 'data', 'gameReports_with_passwords.csv')
LOG_DIR = os.path.join(PROJ, 'data', 'game_logs')

DEFAULT_FP_PASSWORD = 'lembas'
DEFAULT_SA_PASSWORD = 'free'

_lock = threading.Lock()
_table = None          # stem -> CSV row dict (last row wins on dupes)
_dup_stems = None      # stems that appeared more than once
_ts_index = None       # 'YYYY-MM-DD_HHMMSS' -> row, ONLY where unique
_TS_LEN = len('2023-01-01_204954')


def stem_of_url(url):
    """log_file URL/path -> log stem ('.log' stripped)."""
    name = (url or '').strip().rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
    if name.lower().endswith('.log'):
        name = name[:-4]
    return name


def _load():
    global _table, _dup_stems, _ts_index
    if _table is not None:
        return _table
    with _lock:
        if _table is not None:
            return _table
        table, dups = {}, set()
        ts, ts_dup = {}, set()
        try:
            with open(CSV_PATH, encoding='utf-8', errors='replace',
                      newline='') as f:
                for row in csv.DictReader(f):
                    stem = stem_of_url(row.get('log_file'))
                    if not stem:
                        continue
                    if stem in table:
                        dups.add(stem)
                    table[stem] = row
                    key = stem[:_TS_LEN]
                    if key in ts:
                        ts_dup.add(key)
                    ts[key] = row
        except Exception:                               # noqa: BLE001
            # Missing/corrupt CSV: behave as an empty table (defaults).
            table, dups, ts, ts_dup = {}, set(), {}, set()
        for key in ts_dup:                # ambiguous -> no fallback
            ts.pop(key, None)
        _dup_stems = dups
        _ts_index = ts
        _table = table
    return _table


def get_row(stem):
    """Full CSV row for a log stem, or None.

    Falls back to the unique timestamp prefix ('YYYY-MM-DD_HHMMSS')
    when the exact stem misses: a couple of local filenames carry
    mojibake player names (e.g. 'Pa\\u2560\\xfcblo' vs the CSV's
    'P\\xe1blo') but the second-resolution timestamp is unique."""
    table = _load()
    row = table.get(stem)
    if row is None and len(stem) >= _TS_LEN:
        row = _ts_index.get(stem[:_TS_LEN])
    return row


def _csv_passwords(stem):
    """(fp, sa) raw CSV passwords, None where absent/blank."""
    row = get_row(stem)
    if not row:
        return None, None
    fp = (row.get('free_password') or '').strip() or None
    sa = (row.get('shadow_password') or '').strip() or None
    return fp, sa


def get_passwords(stem):
    """(fp_password, sa_password) with None-safe default fallbacks."""
    fp, sa = _csv_passwords(stem)
    return (fp or DEFAULT_FP_PASSWORD, sa or DEFAULT_SA_PASSWORD)


def _a(s):
    """cp1252-safe: ASCII with backslash escapes for odd filenames."""
    return str(s).encode('ascii', 'backslashreplace').decode('ascii')


# ------------------------------------------------------ coverage report
def coverage_report():
    table = _load()
    print('CSV: %s' % CSV_PATH)
    print('  rows keyed by stem : %d' % len(table))
    print('  duplicate stems    : %d %s'
          % (len(_dup_stems), _a(sorted(_dup_stems)[:4])))
    n_fp_blank = sum(1 for r in table.values()
                     if not (r.get('free_password') or '').strip())
    n_sa_blank = sum(1 for r in table.values()
                     if not (r.get('shadow_password') or '').strip())
    print('  blank free_password: %d   blank shadow_password: %d'
          % (n_fp_blank, n_sa_blank))
    logs = sorted(os.path.splitext(fn)[0] for fn in os.listdir(LOG_DIR)
                  if fn.lower().endswith('.log'))
    print('logs on disk: %d' % len(logs))
    have = [s for s in logs if get_row(s) is not None]
    miss = [s for s in logs if get_row(s) is None]
    n_exact = sum(1 for s in logs if s in table)
    print('  exact-stem matches : %d  (+%d via timestamp fallback)'
          % (n_exact, len(have) - n_exact))
    print('  with a CSV row     : %d (%.2f%%)'
          % (len(have), 100.0 * len(have) / max(1, len(logs))))
    print('  without a CSV row  : %d' % len(miss))
    for s in miss[:10]:
        print('    missing: %s' % _a(s))
    extra = sorted(set(table) - set(logs))
    print('CSV rows with no local log: %d' % len(extra))
    for s in extra[:10]:
        print('    csv-only: %s' % _a(s))


if __name__ == '__main__':
    sys.exit(coverage_report())
