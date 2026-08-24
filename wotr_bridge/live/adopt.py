"""CLIENT -> ENGINE adoption (USER ruling 2026-08-21).

    "the resync is supposed to take the CLIENT state and give it to
     the engine, not the other way around"

Until now RESYNC only adopted the client board into the VIEW, and
resync_recover said so in its own docstring ("leaves the engine state
untouched").  That made the unstick button structurally unable to fix
any divergence living in engine state - which is where the guide, the
armies and the fellowship all live.

There IS precedent for this direction: game_logic.adopt_rolled_faces
already takes client-rolled dice faces into the engine.  This
generalises it.

WHY THIS DOES NOT BREAK I1
--------------------------
I1 makes the engine the sole authority on RULES.  Adoption does not
hand rules to the client - it corrects the engine's picture of the
PHYSICAL board, which the human has been manipulating directly.  The
client cannot invent state it does not physically have, so an adopted
board can never exceed the physical component limits (a standing USER
ruling): the pieces on the client ARE the components.

Hidden information is never adopted - hands stay engine-side, and the
dump is blinded before we ever see it.

SAFETY: TRANSACTIONAL, WITH AN ORACLE
-------------------------------------
client_board.diff_engine_vs_dump already enumerates every engine-vs-
client divergence and is a pure detector ("never mutates anything").
That makes it the perfect acceptance test for its own inverse: we
snapshot, adopt, re-run the census, and COMMIT ONLY IF THE DIVERGENCE
COUNT IMPROVED.  A partial or buggy adopter therefore cannot corrupt a
game - the worst it can do is decline and say so.
"""
from . import client_board as cb


def _norm(z):
    from board import sanitize
    return sanitize(str(z))


def client_units_by_region(dump, dynamic_types=None):
    """{(region, (Nation, kind_str)): count} for regular/elite pieces.

    Same read the census performs (client_board:363-372) so the two
    cannot drift: whatever the census counts as a unit is exactly what
    we adopt.
    """
    import board as bd
    regions = bd.regions
    out = {}
    for pid, zone in (dump or {}).items():
        z = _norm(zone)
        base = cb._strip_sh(z)
        if base not in regions:
            continue
        try:
            nat, ut = cb.unit_type(pid, dynamic_types)
        except Exception:
            continue
        if ut not in ('regular', 'elite'):
            continue
        out[(base, (nat, ut))] = out.get((base, (nat, ut)), 0) + 1
    return out


def _snapshot_armies(state):
    """Everything adoption may write, shallow.

    Deliberately NOT deepcopy: an Army holds a board.Region reference
    and a Companion carries identity, so copying them would break the
    identity comparisons the engine does elsewhere.  Re-binding the
    original objects restores exactly what was there.

    Covers the FELLOWSHIP too, not just armies - the first cut snapped
    only armies and its rollback silently left adopted companions and
    a recomputed guide in place.  The decline gate is worthless if the
    rollback is partial, so this must track every field the adopters
    touch.
    """
    fs = state.fellowship
    return {
        'armies': {k: (list(a.units), list(a.leaders))
                   for k, a in state.armies.items()},
        'army_keys': set(state.armies),
        'companions': list(getattr(fs, 'companions', []) or []),
        'guide': getattr(fs, 'guide', None),
        'pending_guide': getattr(state, 'pending_guide_change', None),
    }


def _restore_armies(state, snap):
    for k, (units, leaders) in snap['armies'].items():
        a = state.armies.get(k)
        if a is not None:
            a.units = list(units)
            a.leaders = list(leaders)
    for k in [k for k in list(state.armies) if k not in snap['army_keys']]:
        del state.armies[k]          # armies adoption created
    fs = state.fellowship
    fs.companions = list(snap['companions'])
    fs.guide = snap['guide']
    state.pending_guide_change = snap['pending_guide']


def adopt_units(state, dump, dynamic_types=None):
    """Make every region's regular/elite roster match the client."""
    import game_logic as gl
    import game_state as gs
    want = client_units_by_region(dump, dynamic_types)
    have = {}
    for key, army in state.armies.items():
        k = _norm(key)
        for u in army.units:
            kk = (k, (u.nation, cb._nv(u.kind)))
            have[kk] = have.get(kk, 0) + 1
    if want == have:
        return []
    kinds = {'regular': gs.UnitKind.REGULAR, 'elite': gs.UnitKind.ELITE}
    changed = []
    for region in sorted({k[0] for k in set(want) | set(have)}):
        tgt = {k[1]: n for k, n in want.items() if k[0] == region}
        cur = {k[1]: n for k, n in have.items() if k[0] == region}
        if tgt == cur:
            continue
        army = gl.get_or_create_army(state, region)
        army.units = [gs.Unit(nation=nat, kind=kinds[kd])
                      for (nat, kd), n in sorted(
                          tgt.items(), key=lambda x: str(x[0]))
                      for _ in range(n)]
        for k in sorted(set(tgt) | set(cur), key=lambda x: str(x)):
            a, b = cur.get(k, 0), tgt.get(k, 0)
            if a != b:
                changed.append('units %s %s/%s: %d -> %d'
                               % (region, cb._nv(k[0]), k[1], a, b))
    return changed


def client_leaders_by_region(dump, dynamic_types=None):
    """{region: {'leader': {Nation: n}, 'nazgul': n}} from the dump.

    The census lumps leader/nazgul/minion/companion into one per-side
    count (client_board:370-378), but the DUMP distinguishes them via
    unit_type, so adoption can be precise where detection is coarse.
    Named figures (minions, separated companions) are deliberately NOT
    returned: they carry identity and abilities, so we preserve the
    engine's own objects rather than fabricating look-alikes.
    """
    import board as bd
    regions = bd.regions
    out = {}
    for pid, zone in (dump or {}).items():
        z = _norm(zone)
        base = cb._strip_sh(z)
        if base not in regions:
            continue
        try:
            nat, ut = cb.unit_type(pid, dynamic_types)
        except Exception:
            continue
        if ut not in ('leader', 'nazgul'):
            continue
        e = out.setdefault(base, {'leader': {}, 'nazgul': 0})
        if ut == 'nazgul':
            e['nazgul'] += 1
        else:
            e['leader'][nat] = e['leader'].get(nat, 0) + 1
    return out


def _is_named(obj):
    """Minions and companions carry a name; plain Leader/Nazgul are
    bare `nation` dataclasses.  That is the discriminator for what we
    may rebuild vs what we must preserve."""
    return bool(str(getattr(obj, 'name', '') or ''))


def adopt_leaders(state, dump, dynamic_types=None):
    """Rebuild the plain-Leader / Nazgul roster per region, preserving
    every NAMED figure the engine already holds there."""
    import game_logic as gl
    import game_state as gs
    want = client_leaders_by_region(dump, dynamic_types)
    changed = []
    regions = set(want) | {_norm(k) for k in state.armies}
    for region in sorted(regions):
        tgt = want.get(region, {'leader': {}, 'nazgul': 0})
        army = state.armies.get(region) or state.armies.get(
            next((k for k in state.armies if _norm(k) == region), ''))
        if army is None:
            if not tgt['leader'] and not tgt['nazgul']:
                continue
            army = gl.get_or_create_army(state, region)
        named = [l for l in army.leaders if _is_named(l)]
        cur_plain = sum(1 for l in army.leaders
                        if not _is_named(l)
                        and type(l).__name__ == 'Leader')
        cur_naz = sum(1 for l in army.leaders
                      if type(l).__name__ == 'Nazgul')
        tgt_plain = sum(tgt['leader'].values())
        if cur_plain == tgt_plain and cur_naz == tgt['nazgul']:
            continue
        rebuilt = list(named)
        for nat, n in sorted(tgt['leader'].items(), key=lambda x: str(x[0])):
            rebuilt.extend(gs.Leader(nation=nat) for _ in range(n))
        if tgt['nazgul']:
            naz_nat = next((getattr(l, 'nation', None)
                            for l in army.leaders
                            if type(l).__name__ == 'Nazgul'), None)
            if naz_nat is None:
                naz_nat = getattr(gs.Nation, 'SAURON', None)
            rebuilt.extend(gs.Nazgul(nation=naz_nat)
                           for _ in range(tgt['nazgul']))
        army.leaders = rebuilt
        if cur_plain != tgt_plain:
            changed.append('leaders %s: %d -> %d'
                           % (region, cur_plain, tgt_plain))
        if cur_naz != tgt['nazgul']:
            changed.append('nazgul %s: %d -> %d'
                           % (region, cur_naz, tgt['nazgul']))
    return changed


_FS_ZONES = frozenset(('fellowship box', 'guide box',
                       'fellowship of the ring',
                       'guide of the fellowship'))
_DEAD_ZONES = frozenset(('fp dead', 'sa dead', 'dead', 'fp casualties',
                         'sa casualties'))


def _engine_companions(state):
    """Every Companion object the engine holds, with where it sits:
    ('fellowship', None) or ('region', region_key)."""
    out = {}
    for c in list(getattr(state.fellowship, 'companions', []) or []):
        out[str(c.name)] = (c, 'fellowship', None)
    for key, army in state.armies.items():
        for l in list(army.leaders or []):
            if type(l).__name__ == 'Companion':
                out[str(l.name)] = (l, 'region', key)
    return out


def adopt_companions(state, dump, dynamic_types=None):
    """Move the engine's Companion OBJECTS to where the client shows
    them: Fellowship/Guide box, a map region (separated), or a dead
    pile.  Objects are never fabricated - a companion carries level and
    guide_ability, so a look-alike would be a different piece.  A
    companion the client shows somewhere the engine no longer has an
    object for is REPORTED, not invented.
    """
    import log_replay as lr
    have = _engine_companions(state)
    changed = []
    for pid, zone in (dump or {}).items():
        if str(pid) not in lr.COMPANIONS:
            continue
        z = _norm(zone)
        names = set(lr.names_for_pid(str(pid)) or ())
        match = next((n for n in have if n in names), None)
        if match is None:
            continue                       # already gone, or unknown
        comp, where, wkey = have[match]
        if z in _FS_ZONES:
            want, wantkey = 'fellowship', None
        elif z in _DEAD_ZONES:
            want, wantkey = 'dead', None
        else:
            base = cb._strip_sh(z)
            import board as bd
            if base not in bd.regions:
                continue                   # reinforcement/other box
            want, wantkey = 'region', base
        if want == where and _norm(wkey or '') == _norm(wantkey or ''):
            continue
        # detach
        if where == 'fellowship':
            if comp in state.fellowship.companions:
                state.fellowship.companions.remove(comp)
        else:
            a = state.armies.get(wkey)
            if a is not None and comp in a.leaders:
                a.leaders.remove(comp)
        # reattach
        if want == 'fellowship':
            state.fellowship.companions.append(comp)
        elif want == 'region':
            import game_logic as gl
            # append DIRECTLY, not via place_separated_companion: that
            # helper fires nation-activation (1.149b), and a resync
            # corrects the picture rather than replaying the event
            gl.get_or_create_army(state, wantkey).leaders.append(comp)
        changed.append('companion %s: %s -> %s%s'
                       % (match, where, want,
                          '' if not wantkey else ' (%s)' % wantkey))
    if changed:
        import game_logic as gl
        g = getattr(state.fellowship, 'guide', None)
        if g is not None and g not in state.fellowship.companions:
            gl.change_guide(state, allow_pending=True)
            changed.append('guide recomputed (previous guide left the '
                           'Fellowship)')
    return changed


class AdoptionRefused(RuntimeError):
    """The dump is not a full, stable client read: adopting it would
    rewrite engine regions to an ABSENT roster.  Raised BEFORE any
    write; the engine is untouched."""


def _stable_dump(dump, dump_path):
    """The dump to adopt, proven full and stable, or AdoptionRefused.

    Invariants: fewer than client_board.MIN_DUMP_PIECES pieces is a
    torn/missing read (client fact 14), never a board; with a
    dump_path, an INDEPENDENT torn-safe read of the file must agree
    with the caller's dump - a dump that disagrees with the file is
    stale or mid-write.  The census diffs against the independent read
    when one exists (the caller's dict cannot vouch for itself)."""
    n = len(dump or {})
    if n < cb.MIN_DUMP_PIECES:
        raise AdoptionRefused(
            'ADOPTION REFUSED: dump holds %d piece(s), a full client '
            'board has >= %d - a torn or missing read; engine left '
            'untouched' % (n, cb.MIN_DUMP_PIECES))
    if dump_path is None:
        return dict(dump)
    fresh = cb.read_board(dump_path)
    if fresh is None:
        raise AdoptionRefused(
            'ADOPTION REFUSED: no stable independent read of %s '
            '(torn across retries); engine left untouched'
            % (dump_path,))
    if fresh != dump:
        diff = sorted(p for p in set(fresh) | set(dump)
                      if fresh.get(p) != dump.get(p))
        raise AdoptionRefused(
            'ADOPTION REFUSED: the dump disagrees with an independent '
            'read of the file on %d piece(s) (first: %s) - unstable '
            'across two reads; engine left untouched'
            % (len(diff), ','.join(diff[:6])))
    return fresh


def adopt_tracks(state, dump):
    """The TRACK markers the client shows, taken as given: political
    boxes (client 'Political N' = engine box 3-N; the active flag is
    the human's separate flip and stays), the Fellowship progress
    (FSP 247/248 on 'Track N') and the corruption marker (252).
    Lenient-mode companion of the piece adopters (USER 2026-08-22:
    'takes the new game state as given after each action')."""
    import re as _re
    from log_replay import POLITICAL_MARKERS
    out = []
    for pid, zone in (dump or {}).items():
        z = str(zone)
        m = _re.match(r'^Political (\d)$', z)
        if m and str(pid) in POLITICAL_MARKERS:
            nation = POLITICAL_MARKERS[str(pid)]
            box = 3 - int(m.group(1))
            if state.political.boxes.get(nation) != box:
                out.append('political %s box %s -> %d'
                           % (getattr(nation, 'value', nation),
                              state.political.boxes.get(nation), box))
                state.political.boxes[nation] = box
            continue
        m = _re.match(r'^Track (\d+)$', z)
        if m and str(pid) in ('247', '248', '252'):
            n = int(m.group(1))
            fs = state.fellowship
            if str(pid) == '252':
                if getattr(fs, 'corruption', None) != n:
                    out.append('corruption %s -> %d'
                               % (getattr(fs, 'corruption', None), n))
                    fs.corruption = n
            elif getattr(fs, 'progress', None) != n:
                out.append('fellowship progress %s -> %d'
                           % (getattr(fs, 'progress', None), n))
                fs.progress = n
    return out


def adopt_client_board(state, dump, dynamic_types=None, ai_module=None,
                       guide_hook=None, dump_path=None):
    """Adopt the client's physical board into the ENGINE.

    guide_hook: optional zero-arg callable returning a note string (the
    driver passes its reconcile_guide_from_client, which routes the
    guide through game_logic.change_guide so the RULES still validate
    the pick).
    dump_path: the client dump file; when given, the census diffs
    against an INDEPENDENT read that must agree with `dump`.

    Returns (report_lines, n_before, n_after).  Refuses (AdoptionRefused,
    engine untouched) a dump below MIN_DUMP_PIECES or unstable across
    two reads; commits only when the census divergence count did not
    get worse.
    """
    report = []
    census_dump = _stable_dump(dump, dump_path)
    census = (lambda st: cb.diff_engine_vs_dump(
        st, census_dump, dynamic_types=dynamic_types,
        ai_module=ai_module))
    before = census(state)
    if before is None:
        raise AdoptionRefused(
            'ADOPTION REFUSED: the census cannot judge this dump '
            '(diff_engine_vs_dump returned None); engine left untouched')
    snap = _snapshot_armies(state)
    try:
        report.extend(adopt_units(state, census_dump, dynamic_types))
        report.extend(adopt_leaders(state, census_dump, dynamic_types))
        # companions BEFORE the guide: the guide must be picked from a
        # Fellowship that already reflects who is still in it
        report.extend(adopt_companions(state, census_dump,
                                       dynamic_types))
        if guide_hook is not None:
            note = guide_hook()
            if note:
                report.append(note)
    except Exception as e:
        _restore_armies(state, snap)
        report.append('unit adoption failed, rolled back: %r' % (e,))
        return report, len(before), len(before)
    after = census(state) or []
    if len(after) > len(before):
        _restore_armies(state, snap)
        report.append('ADOPTION DECLINED: divergences would have gone '
                      '%d -> %d, so nothing was written'
                      % (len(before), len(after)))
        return report, len(before), len(before)
    try:
        for _tl in adopt_tracks(state, dump):
            report.append('track adopted: %s' % _tl)
    except Exception as _te:
        report.append('track adoption failed: %r' % (_te,))
    report.append('board adopted: divergences %d -> %d'
                  % (len(before), len(after)))
    return report, len(before), len(after)
