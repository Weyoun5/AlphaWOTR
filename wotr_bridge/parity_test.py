"""State summaries shared by the bridge's comparators.

summarize(state) is the comparable projection of an engine state (armies
by region/type, fellowship, table, political, VP) that
client_board.proj_board / board_delta and gesture_match key on - the ONE
answerer for the projection; diff_summ(a, b) names the first cross-model
difference.

The AI-vs-AI parity harness that once lived here (ParityGame /
inject_and_verify / the --game and --verify-client CLI) was retired with
the monolith on 2026-08-22.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, '..', 'src'))
for p in (HERE, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)


# ------------------------------------------------------- state summary
def _u(x):
    return getattr(x, 'value', x)


def summarize(state):
    """A comparable projection of the two engine paths.  Only fields
    both log_replay AND apply_action populate; sorted/canonical."""
    from board import sanitize
    out = {}
    armies = {}
    for key, army in state.armies.items():
        k = sanitize(key)
        reg = {}
        for un in army.units:
            t = '%s:%s' % (str(_u(getattr(un, 'nation', '?'))).lower(),
                           str(_u(getattr(un, 'kind',
                                          getattr(un, 'unit_type', '?')))
                               ).lower())
            reg[t] = reg.get(t, 0) + 1
        nleaders = len(getattr(army, 'leaders', []))
        if reg or nleaders:
            armies[k] = (tuple(sorted(reg.items())), nleaders)
    out['armies'] = tuple(sorted(armies.items()))
    fs = state.fellowship
    # RED EX-9 leg 2 (2026-08-17): the declared REGION was invisible
    # to the whole comparison stack - declares to different regions
    # summarized identically, so the matcher could adopt the wrong
    # region and the hidden-divergence counter could not even count
    # it.  APPENDED (index 7) so no index-consumer moves.
    try:
        import game_logic as _gl9
        _fsr9 = str(_gl9.fellowship_region(state))
    except Exception:
        _fsr9 = ''
    out['fellowship'] = (
        int(getattr(fs, 'progress', 0)),
        int(getattr(fs, 'corruption', 0)),
        bool(getattr(fs, 'hidden', True)),
        bool(getattr(fs, 'mordor_entered', False)),
        int(getattr(fs, 'mordor_step', 0)),
        str(getattr(getattr(fs, 'guide', None), 'name', '')),
        tuple(sorted(str(c.name) for c in
                     getattr(fs, 'companions', []))),
        _fsr9,
    )
    # RED EX-10 (2026-08-17): the TABLE-card identity term.  Two
    # discard_table_card targets deltaed AND measured byte-identical
    # (persistent_cards/discard_face_up were outside the projection)
    # - the sole silent-to-measure collision in the 117k-sim
    # round-trip scan.  One term heals matcher AND counter.
    _pc9 = getattr(state, 'persistent_cards', None)
    out['table'] = (
        tuple(sorted(str(getattr(c, 'card_id', c))
                     for c in getattr(_pc9, 'fp_cards', []) or [])),
        tuple(sorted(str(getattr(c, 'card_id', c))
                     for c in getattr(_pc9, 'sh_cards', []) or [])),
        tuple(sorted(
            (str(k), tuple(str(x) for x in v))
            for k, v in (getattr(state, 'discard_face_up', None)
                         or {}).items())),
    )
    pol = getattr(state, 'political', None)
    if pol is not None:
        out['political'] = tuple(sorted(
            (str(_u(n)).lower(), int(b))
            for n, b in getattr(pol, 'boxes', {}).items()))
    out['vp'] = (int(getattr(state, 'fp_vp', getattr(state, 'vp_fp', 0))
                     or 0),
                 int(getattr(state, 'sh_vp', getattr(state, 'vp_sh', 0))
                     or 0))
    return out


def diff_summ(a, b):
    """Human-readable first-difference between two summaries.

    SCOPE (AUTO-94 provenance run, 2026-08-17): 'table' is EXCLUDED
    here on purpose.  diff_summ's callers are all CROSS-MODEL
    (engine vs the log-replay roundtrip: postgame_parity, the sided
    census, _verify_move) and the replayer models NEITHER
    persistent-cards-on-table NOR discard_face_up piles - the EX-10
    term made every old CLEAN bundle diverge on 'table' under the
    strengthened instrument.  EX-10's protections are SAME-MODEL
    consumers (the matcher's _board_delta and the hidden-divergence
    measure compare summarize() to summarize() of the same engine)
    and keep the term through summarize itself."""
    for key in ('armies', 'fellowship', 'political', 'vp'):
        if a.get(key) != b.get(key):
            if key == 'armies':
                da, db = dict(a.get(key, ())), dict(b.get(key, ()))
                for rk in sorted(set(da) | set(db)):
                    if da.get(rk) != db.get(rk):
                        return ('armies@%s: engine=%s roundtrip=%s'
                                % (rk, da.get(rk), db.get(rk)))
            return '%s: engine=%s roundtrip=%s' % (key, a.get(key),
                                                   b.get(key))
    return None
