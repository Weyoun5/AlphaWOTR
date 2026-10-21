"""Adopt a stable native-client board into the rules engine.

Ordinary reconciliation takes pieces without overriding pending rule effects.
Explicit recovery or consent also takes visible ownership and track state;
manual recovery additionally takes politics and known card/deck identities.
Unknown human cards remain placeholders. Every adopting leg shares one rollback
snapshot, and a transaction that increases census disagreements is refused.
"""
from . import client_board as cb


def _norm(z):
    from board import sanitize
    return sanitize(str(z))


def _decode_piece(pid, dynamic_types=None):
    """(Nation, kind_str) for a client pid, or None when it does not
    decode to a force piece.  The ONE guarded call to cb.unit_type,
    shared by every by-zone counter below (the decode-skip was
    duplicated per function; consolidating it here drops two silent
    handlers and keeps the three counters reading pids identically)."""
    try:
        return cb.unit_type(pid, dynamic_types)
    except Exception:
        return None


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
        dec = _decode_piece(pid, dynamic_types)
        if dec is None:
            continue
        nat, ut = dec
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
    touch (besieged_strongholds since adopt_sieges, 2026-08-24;
    BRCMP-7 2026-08-29: corruption, enemy_controlled, fp_casualties,
    political boxes/activation, both hands, the four decks and the
    face-up discards - every leg now runs INSIDE the one transaction).
    """
    import copy as _cp
    fs = state.fellowship
    pt = getattr(state, 'political', None)
    snap = {
        'armies': {k: (list(a.units), list(a.leaders))
                   for k, a in state.armies.items()},
        'army_keys': set(state.armies),
        'companions': list(getattr(fs, 'companions', []) or []),
        'guide': getattr(fs, 'guide', None),
        'pending_guide': getattr(state, 'pending_guide_change', None),
        'besieged': set(getattr(state, 'besieged_strongholds', set())
                        or set()),
        'fs_region': getattr(fs, 'region', None),
        'fs_progress': getattr(fs, 'progress', None),
        'mordor': (getattr(fs, 'mordor_entered', None),
                   getattr(fs, 'mordor_step', None)),
        'fs_corruption': getattr(fs, 'corruption', None),
        'fs_hidden': fs.hidden,
        'hunt_reveal_state': {
            name: (hasattr(state, name), _cp.deepcopy(getattr(state, name, None)))
            for name in ('_hunt_reveal_pending', '_reveal_place_due', '_sh_tile_drawn_regions')},
        'enemy_controlled': {
            k: set(v or ()) for k, v in
            (getattr(state, 'enemy_controlled', None) or {}).items()},
        'fp_casualties': _cp.deepcopy(
            getattr(state, 'fp_casualties', None)),
        'political': ((dict(pt.boxes), dict(pt.active))
                      if pt is not None else None),
        'hands': {s: (list(h.hand), bool(getattr(h, 'needs_discard',
                                                 False)))
                  for s, h in (getattr(state, 'hands', None)
                               or {}).items()},
        'decks': {attr: list(getattr(state, attr))
                  for attr in ('fp_char_deck', 'fp_strat_deck',
                               'sh_char_deck', 'sh_strat_deck')
                  if getattr(state, attr, None) is not None},
        'discard_face_up': {
            k: list(v or ()) for k, v in
            (getattr(state, 'discard_face_up', None) or {}).items()},
        'rings': {name: getattr(state, name, None) for name in (
            'fp_elven_rings', 'sh_elven_rings', 'fp_ring_used_this_turn',
            'sh_ring_used_this_turn')},
        'dice_flags': [(die, die.used, die.current)
                       for dice in getattr(state, 'dice', {}).values()
                       for die in dice],
        'hunt_box': list(getattr(state, 'hunt_box', [])),
        'minion_flags': [(obj, obj.in_play, obj.was_ever_in_play)
                         for obj in (getattr(state, name, None) for name in
                                     ('witch_king', 'saruman', 'mouth_of_sauron'))
                         if obj is not None],
        'promotion_flags': [(obj, flag, getattr(obj, flag))
                            for obj, flag in (
                                (getattr(state, 'gandalf_the_white', None), 'in_play'),
                                (getattr(state, 'aragorn_crowned', None), 'crowned'))
                            if obj is not None],
        'persistent_cards': (getattr(state, 'persistent_cards', None),
                             _cp.deepcopy(vars(state.persistent_cards))
                             if getattr(state, 'persistent_cards', None) is not None
                             else None),
    }
    return snap


def _restore_armies(state, snap):
    for obj, flag, value in snap.get('promotion_flags', []):
        setattr(obj, flag, value)
    for obj, in_play, was_in_play in snap.get('minion_flags', []):
        obj.in_play, obj.was_ever_in_play = in_play, was_in_play
    persistent, fields = snap.get('persistent_cards', (None, None))
    if persistent is not None:
        persistent.__dict__.clear()
        persistent.__dict__.update(fields)
    for name, value in snap.get('rings', {}).items():
        setattr(state, name, value)
    for die, used, current in snap.get('dice_flags', []):
        die.used, die.current = used, current
    if 'hunt_box' in snap:
        state.hunt_box[:] = snap['hunt_box']
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
    state.besieged_strongholds = set(snap.get('besieged', set()))
    if 'fs_region' in snap:
        fs.region = snap['fs_region']
        fs.progress = snap['fs_progress']
        fs.mordor_entered, fs.mordor_step = snap['mordor']
    if 'fs_corruption' in snap:
        fs.corruption = snap['fs_corruption']
    if 'fs_hidden' in snap:
        fs.hidden = snap['fs_hidden']
    for name, (present, value) in snap.get('hunt_reveal_state', {}).items():
        if present:
            setattr(state, name, value)
        else:
            vars(state).pop(name, None)
    if 'enemy_controlled' in snap:
        ec = getattr(state, 'enemy_controlled', None)
        if ec is not None:
            ec.clear()
            for k, v in snap['enemy_controlled'].items():
                ec[k] = set(v)
    if 'fp_casualties' in snap \
            and getattr(state, 'fp_casualties', None) is not None:
        state.fp_casualties.clear()
        state.fp_casualties.update(snap['fp_casualties'] or {})
    pt = getattr(state, 'political', None)
    if snap.get('political') is not None and pt is not None:
        boxes, active = snap['political']
        pt.boxes.clear()
        pt.boxes.update(boxes)
        pt.active.clear()
        pt.active.update(active)
    for s, (hand, nd) in (snap.get('hands') or {}).items():
        h = state.hands.get(s)
        if h is not None:
            h.hand[:] = list(hand)
            h.needs_discard = nd
    for attr, cards in (snap.get('decks') or {}).items():
        lst = getattr(state, attr, None)
        if lst is not None:
            lst[:] = list(cards)
    dfu = getattr(state, 'discard_face_up', None)
    if 'discard_face_up' in snap and dfu is not None:
        for k in list(dfu):
            if k not in snap['discard_face_up']:
                del dfu[k]
        for k, v in snap['discard_face_up'].items():
            if k in dfu:
                dfu[k][:] = list(v)
            else:
                dfu[k] = list(v)


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
        dec = _decode_piece(pid, dynamic_types)
        if dec is None:
            continue
        nat, ut = dec
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
    from collections import Counter
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
        cur_nations = Counter(l.nation for l in army.leaders
                              if not _is_named(l)
                              and type(l).__name__ == 'Leader')
        cur_plain = sum(cur_nations.values())
        cur_naz = sum(1 for l in army.leaders
                      if type(l).__name__ == 'Nazgul')
        tgt_plain = sum(tgt['leader'].values())
        if cur_nations == tgt['leader'] and cur_naz == tgt['nazgul']:
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
        if cur_nations != tgt['leader']:
            def roster(counts):
                return ', '.join('%s=%d' % (getattr(nat, 'value', nat), n)
                                 for nat, n in sorted(counts.items(), key=lambda x: str(x[0]))
                                 if n) or 'none'
            changed.append('leader nations %s: %s -> %s'
                           % (region, roster(cur_nations), roster(tgt['leader'])))
        if cur_naz != tgt['nazgul']:
            changed.append('nazgul %s: %d -> %d'
                           % (region, cur_naz, tgt['nazgul']))
    return changed


_FS_ZONES = frozenset(('fellowship box', 'guide box',
                       'fellowship of the ring',
                       'guide of the fellowship'))
_DEAD_ZONES = frozenset(('fp dead', 'sa dead', 'dead', 'fp casualties',
                         'sa casualties'))


def _engine_companions(state, pending_companions=()):
    """Every Companion object the engine holds, with where it sits:
    ('fellowship', None) or ('region', region_key)."""
    out = {}
    for c in list(getattr(state.fellowship, 'companions', []) or []):
        out[str(c.name)] = (c, 'fellowship', None)
    for key, army in state.armies.items():
        for l in list(army.leaders or []):
            if type(l).__name__ == 'Companion':
                out[str(l.name)] = (l, 'region', key)
    # Explicit Skip retains these exact objects before abandoning a Hunt.
    # Ordinary adoption never borrows a pending figure: its placement remains
    # a live rules decision until the player explicitly abandons it.
    for companion in pending_companions:
        out.setdefault(str(companion.name), (companion, 'pending', None))
    return out


def adopt_companions(state, dump, dynamic_types=None, pending_companions=()):
    """Move the engine's Companion OBJECTS to where the client shows
    them: Fellowship/Guide box, a map region (separated), or a dead
    pile.  Objects are never fabricated - a companion carries level and
    guide_ability, so a look-alike would be a different piece.  A
    companion the client shows somewhere the engine no longer has an
    object for is REPORTED, not invented.
    """
    import log_replay as lr
    have = _engine_companions(state, pending_companions)
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
        elif want == 'dead':
            # the engine's leaves-play booking for a companion out of
            # the game (casualty=False: no WWSAT - a resync corrects
            # the picture, it never replays the casualty) so the
            # companion-linked table cards (Wizard's Staff / Horn of
            # Gondor / Axe and Bow) re-derive NOW, reported per card
            import game_logic as gl
            pc = getattr(state, 'persistent_cards', None)
            before = [c.card_id for c in (pc.fp_cards if pc else [])]
            gl._on_companion_casualty(state, match, casualty=False)
            after = [c.card_id for c in (pc.fp_cards if pc else [])]
            for cid in before:
                if cid not in after:
                    changed.append('table card discarded (%s left '
                                   'play): %s' % (match, cid))
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


def adopt_minions(state, dump, manual=False):
    """Relocate the engine's EXISTING named-Minion objects (Witch-king
    253 / Saruman 254 / Mouth 255) to the map region the client dump
    shows (USER bug 2026-08-24: a leniently-accepted army move
    stranded the Witch-King engine-side and no adopter leg could ever
    repair it - client_leaders_by_region deliberately returns no named
    figures).  RELOCATE ONLY, never fabricate: a minion pid the engine
    holds no object for stays a reported census row.  Saruman is
    adopted like the others (the immobility rule constrains legal
    MOVES; adoption corrects the engine's picture of the physical
    board, resync ruling 2026-08-21) with a loud report line."""
    import board as bd
    out = []
    minion_pids = {'253': 'witch', '254': 'saruman', '255': 'mouth'}
    for pid, tag in minion_pids.items():
        zone = None
        for p, z in (dump or {}).items():
            if str(p) == pid:
                zone = str(z)
                break
        if zone is None:
            continue
        base = cb._strip_sh(_norm(zone))
        on_map = base in bd.regions
        if not on_map and not manual:
            continue
        # find the engine's Minion object of this identity
        holder_key, obj = None, None
        for key, army in state.armies.items():
            for l in list(army.leaders or []):
                # FLB-4 (FLA_HANDOFF 4, par5 g3#6): board minion
                # figures carry EITHER type name - the game_logic
                # Minion wrapper (musters) or the raw ShadowMinion
                # record - the finder must see both
                if type(l).__name__ in ('Minion', 'ShadowMinion') \
                        and tag in str(getattr(l, 'name', '')).lower():
                    holder_key, obj = key, l
                    break
            if obj is not None:
                break
        if manual:
            import game_logic as gl
            canonical = getattr(state, {'witch': 'witch_king',
                                'saruman': 'saruman', 'mouth': 'mouth_of_sauron'}[tag])
            if not on_map and base not in _DEAD_ZONES | {
                    'sa reinf', 'sa reinforcements', 'reinforcements'}:
                raise AdoptionRefused('minion %s is in an unrecognized zone: %s'
                                      % (tag, base))
            canonical.in_play = on_map
            if on_map or base in _DEAD_ZONES:
                canonical.was_ever_in_play = True
            if not on_map:
                if obj is not None:
                    state.armies[holder_key].leaders.remove(obj)
                    out.append('minion %s removed to %s' % (canonical.name, base))
                continue
            if obj is None:
                gl._add_minion_to_region(state, canonical, base)
                out.append('minion %s adopted in %s' % (canonical.name, base))
                continue
        if obj is None:
            continue
        if _norm(holder_key) == base:
            continue                    # already agreed
        import game_logic as gl
        army = state.armies.get(holder_key)
        if army is not None and obj in army.leaders:
            army.leaders.remove(obj)
        gl.get_or_create_army(state, base).leaders.append(obj)
        out.append('minion %s: %s -> %s%s'
                   % (getattr(obj, 'name', tag), _norm(holder_key),
                      base,
                      ' (NOTE: Saruman is rules-immobile - the client'
                      ' board is adopted as physical truth)'
                      if tag == 'saruman' else ''))
    return out


def adopt_sieges(state, dump, dynamic_types=None, allow_drop=False):
    """Derive `state.besieged_strongholds` from the client dump.

    The dump distinguishes inside-the-walls ('<region> Stronghold'
    zone) from the field ('<region>'), so a siege is directly readable:
    exactly one side's UNITS inside, none of that side's units in the
    field, and at least one opposing unit in the field.  Before this
    (2026-08-24, tester 'can't play The Fighting Uruk-hai'), the flag
    was written ONLY by the engine's own combat path - a siege formed
    through a leniently-accepted window existed on the client but not
    in the engine, so every siege-gated rule (the card's besieging
    requirement, the 5-unit cap, sortie direction) silently misjudged,
    and RESYNC could not repair it because nothing adopted the flag.

    Cleanly-shaped sieges are added; a flagged region the dump no
    longer shows siege-shaped is dropped ONLY under `allow_drop` (the
    CHAOS-1 stale-flag class - a MANUAL resync or the human's consent,
    BRCMP-4 2026-08-29: a flagged stronghold with nobody in the
    Stronghold sub-box is the BL-76 seat demand, a garrison the human
    has not dragged inside, NOT a lifted siege; the auto path never
    drops a rules flag); a messy shape (one side split inside AND
    outside) is left untouched and reported."""
    import board as bd
    from combat import side_of_nation
    bs = getattr(state, 'besieged_strongholds', None)
    if bs is None:
        bs = state.besieged_strongholds = set()
    inside = {}
    field = {}
    for pid, zone in (dump or {}).items():
        z = _norm(zone)
        base = cb._strip_sh(z)
        if base not in bd.regions:
            continue
        try:
            nat, ut = cb.unit_type(pid, dynamic_types)
        except Exception:
            continue
        if ut not in ('regular', 'elite'):
            continue
        side = side_of_nation(nat)
        (inside if z != base else field).setdefault(
            base, set()).add(side)
    changed = []
    strongholds = {r for r in (set(inside) | set(field) | set(bs))
                   if bd.regions.get(r, {}).get('is_stronghold')}
    for r in sorted(strongholds):
        ins = inside.get(r, set())
        fld = field.get(r, set())
        shaped = (len(ins) == 1
                  and not (ins & fld)
                  and bool(fld - ins))
        if shaped and allow_drop:
            # During explicit recovery the occupied inner box identifies the
            # controller. A stale native owner would reverse siege sides,
            # applying the five-unit limit to the army outside the walls.
            from game_logic import is_fp_region
            from game_state import Side
            controller = next(iter(ins))
            native = Side.FREE_PEOPLES if is_fp_region(r) else Side.SHADOW
            for side, captures in state.enemy_controlled.items():
                should_hold = side is controller and controller is not native
                if should_hold and r not in captures:
                    captures.add(r)
                    changed.append('stronghold control adopted: %s by %s'
                                   % (r, side.name))
                elif not should_hold and r in captures:
                    captures.discard(r)
                    changed.append('stale stronghold control removed: %s by %s'
                                   % (r, side.name))
        if shaped and r not in bs:
            bs.add(r)
            changed.append('siege adopted: %s' % r)
        elif r in bs and not ins:
            if allow_drop:
                # nobody inside the walls: no siege (lifted / vacated)
                bs.discard(r)
                changed.append('siege dropped: %s (no garrison inside)'
                               % r)
            else:
                changed.append('siege kept: %s (no garrison piece in '
                               'the Stronghold box) - owed: seat the '
                               'garrison in the %s Stronghold box '
                               '(BL-76); a manual resync drops the '
                               'flag' % (r, r))
        elif not shaped and (ins & fld):
            changed.append('siege NOT adopted at %s: %s units both '
                           'inside and outside (messy shape, left '
                           'as-is)' % (r, '/'.join(sorted(
                               s.name for s in (ins & fld)))))
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


def adopt_tracks(state, dump, owed=None):
    """The TRACK markers the client shows, taken as given: the
    Fellowship progress (FSP 247/248 on 'Track N') and the corruption
    marker (252).  Lenient-mode companion of the piece adopters (USER
    2026-08-22: 'takes the new game state as given after each
    action').

    POLITICAL boxes are NOT adopted (USER trust ruling 2026-08-24:
    "trust engine for political track, client otherwise") - the
    engine is the political authority and the driver's political
    reconciler walks the CLIENT markers to match it, never the
    reverse.

    `owed` (BL-2, 2026-08-25): {pid: target} for engine-side track
    changes still awaiting their rendered marker move.  While a
    track is owed and the client marker has not yet reached the
    target, that track is HELD, never adopted - fpgame1's tile-1r
    corruption was silently erased by adopting the untouched client
    marker over the engine's rules-derived damage."""
    import re as _re
    out = []
    for pid, zone in (dump or {}).items():
        z = str(zone)
        m = _re.match(r'^Track (\d+)$', z)
        if m and str(pid) in ('247', '248', '252'):
            n = int(m.group(1))
            fs = state.fellowship
            tgt = (owed or {}).get(str(pid))
            if tgt is not None and n != int(tgt):
                out.append('track pid %s adoption HELD: engine change '
                           'to %s awaits its marker move (client '
                           'shows %d)' % (pid, tgt, n))
                continue
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


def adopt_fellowship_position(state, dump, owed=None, manual=False):
    """Adopt the Fellowship PAWN's position from the client (USER bug
    2026-08-24: 'the fellowship pawn was not tracked during resyncs' -
    a manual edit moving the pawn never reached the engine).  The pawn
    is pid 247: on a MAP REGION zone the engine fellowship moves
    there; on a 'Mordor N' box the engine adopts the Mordor step
    (entry is physical truth on the client per the resync direction
    ruling 2026-08-21).  'Track N' zones are the PROGRESS marker's
    home and are handled by adopt_tracks, not here."""
    import re as _re
    import board as bd
    out = []
    zone = None
    for pid, z in (dump or {}).items():
        if str(pid) == '247':
            zone = str(z)
            break
    if zone is None:
        return out
    fs = state.fellowship
    m = _re.match(r'^Mordor (\d)', zone)
    if m:
        step = int(m.group(1))
        if not getattr(fs, 'mordor_entered', False) \
                or getattr(fs, 'mordor_step', None) != step:
            out.append('fellowship: Mordor step %s -> %d (pawn box)'
                       % (getattr(fs, 'mordor_step', None), step))
            fs.mordor_entered = True
            fs.mordor_step = step
            fs.progress = 0
        return out
    base = cb._strip_sh(_norm(zone))
    if base not in bd.regions:
        return out                 # a track/other box: nothing to do
    if owed and base != _norm(str(owed)):
        # F1: an engine-side pawn move awaits its rendered
        # walk - the stale client pawn is NOT adoptable until it
        # lands (the BL-2 owed-guard pattern for pid 247)
        out.append('fellowship pawn adoption HELD: engine move to '
                   '%s awaits its walk (client shows %s)'
                   % (owed, base))
        return out
    import game_logic as gl
    cur = _norm(str(gl.fellowship_region(state)))
    if manual and getattr(fs, 'mordor_entered', False):
        fs.mordor_entered = False
        fs.mordor_step = 0
        fs.region = state.board.region(base)
        fs.progress = 0
        return ['fellowship: Mordor -> %s (explicit board recovery)' % base]
    if cur == base or getattr(fs, 'mordor_entered', False):
        return out
    try:
        fs.region = state.board.region(base)
        fs.progress = 0
        out.append('fellowship pawn: %s -> %s (progress reset)'
                   % (cur, base))
    except Exception as _fe:
        out.append('fellowship pawn NOT adopted (%s -> %s): %r'
                   % (cur, base, _fe))
    return out


def capture_coherence(state):
    """Revert inconsistent captures during an authorized board adoption.

    Use the same garrison distinction as the read-only witness: the original
    owner's army outside a besieged captured stronghold is legal. For actual
    conflicts, VP derives from the corrected enemy_controlled bookkeeping.
    """
    out = []
    for cap_side, owner, rkey, n_owner in cb.captured_garrison_conflicts(state):
        state.enemy_controlled[cap_side].discard(rkey)
        out.append('capture INCOHERENT at %s: %d %s unit(s) '
                   'standing - the %s capture is REVERTED '
                   '(client truth; VP re-derives)'
                   % (rkey, n_owner, owner.name, cap_side.name))
    return out


def adopt_control_markers(state, dump):
    """Explicit recovery reads ownership from physical control markers.

    With every marker deployed, an existing unmarked capture may exceed the
    component supply. Keep it until there is contrary physical evidence.
    """
    import board
    from game_state import Side
    from game_logic import is_fp_region
    observed = {}
    for side, ids in ((Side.FREE_PEOPLES, range(405, 411)),
                      (Side.SHADOW, range(411, 421))):
        positions = [cb._strip_sh(_norm(dump.get('%03d' % pid, '')))
                     for pid in ids]
        regions = [region for region in positions if region in board.regions]
        if len(regions) != len(set(regions)):
            raise AdoptionRefused('duplicate %s control markers' % side.name)
        for region in regions:
            native = Side.FREE_PEOPLES if is_fp_region(region) else Side.SHADOW
            if native is side:
                raise AdoptionRefused('control marker at own settlement: %s' % region)
        observed[side] = set(regions)
        if len(regions) == len(ids):
            observed[side].update(state.enemy_controlled.get(side, ()))
    notes = []
    for side, regions in observed.items():
        previous = state.enemy_controlled.setdefault(side, set())
        if previous != regions:
            notes.append('control markers adopted: %s %s -> %s' %
                         (side.name, sorted(previous), sorted(regions)))
            previous.clear()
            previous.update(regions)
    return notes


def client_fp_casualties(dump, dynamic_types=None):
    """{(Nation, kind_str): count} of FREE-PEOPLES pieces sitting in the
    client's casualty / dead zones - the physical dead pile the dump
    shows.  kind in regular / elite / leader (nazgul are Shadow).  Same
    per-pid decode (cb.unit_type) the region adopters use, so the dead
    count cannot drift from the deployed count."""
    import game_state as gs
    try:
        from combat import side_of_nation as _son
    except Exception:
        return {}
    out = {}
    for pid, zone in (dump or {}).items():
        if _norm(zone) not in _DEAD_ZONES:
            continue
        dec = _decode_piece(pid, dynamic_types)
        if dec is None:
            continue
        nat, ut = dec
        if ut not in ('regular', 'elite', 'leader'):
            continue
        if _son(nat) is not gs.Side.FREE_PEOPLES:
            continue
        out[(nat, ut)] = out.get((nat, ut), 0) + 1
    return out


def reconcile_fp_casualties(state, dump, dynamic_types=None, manual=False,
                            resurrected=None):
    """BL-21(5) (2026-08-25): the resync adopters rebuild every DEPLOYED
    piece to client truth (adopt_units / adopt_leaders) but nothing
    touched `fp_casualties`.  When an adoption resurrects a battle-dead
    FP garrison the client kept standing (the BL-21(1) held-choice
    family), the dead pieces re-enter the deployed count while their
    deaths stay on the ledger - conservation breaks (deployed +
    casualties > components) and the shrunken `avail = max - deployed -
    casualties` UNDER-OFFERS legal musters.  Both are invisible to the
    piece census (engine == client on every figure).

    MANUAL takes the client's physical dead pile EXACTLY (client AS IS,
    BL-147 doctrine).  AUTO (BRCMP-4, 2026-08-29) NEVER lowers the
    ledger on the dead-pile count alone - a dump whose dead pile LAGS
    a fresh kill, or whose dead piece went back to Reinforcements
    (fpgame15: a dead Elf re-entered the muster pool), keeps the
    engine's count.  It lowers only
      (a) by the pieces the SAME adoption resurrected per nation/kind
          (`resurrected`, the deployed delta), never below the
          client's dead count - a resurrected piece is not dead; and
      (b) to restore the PHYSICAL COMPONENT LIMIT (deployed +
          casualties <= pieces; USER ruling 2026-08-18) when an
          adoption broke it.
    Loud per change; never mutates anything else."""
    fpc = getattr(state, 'fp_casualties', None)
    if fpc is None:
        return []
    import ai as _ai
    client = client_fp_casualties(dump, dynamic_types)
    dep = _ai._count_deployed_all(state)
    keys = set(client)
    for nat, kinds in fpc.items():
        for k in (kinds or {}):
            keys.add((nat, k))
    notes = []
    for nat, kind in sorted(keys, key=lambda x: (str(x[0]), str(x[1]))):
        cur = int((fpc.get(nat) or {}).get(kind, 0))
        want = int(client.get((nat, kind), 0))
        if manual:
            new, why = want, 'manual client-as-is'
        else:
            new, why = cur, ''
            res_k = int((resurrected or {}).get((nat, kind), 0) or 0)
            if res_k > 0 and cur > want:
                new = max(want, cur - res_k)
                why = '%d resurrected by this adoption' % res_k
            mx = (_ai.NATION_MAX_FIGURES.get(nat) or {}).get(kind)
            d = int((dep.get(nat) or {}).get(kind, 0))
            if mx is not None and d + new > int(mx):
                new = max(0, int(mx) - d)
                why = ('conservation: %d deployed + %d casualties > '
                       '%d pieces' % (d, cur, int(mx)))
        if new != cur:
            fpc.setdefault(nat, {})[kind] = new
            notes.append(
                'fp_casualties %s %s: %d -> %d (%s)'
                % (getattr(nat, 'name', nat), kind, cur, new, why))
    return notes


def adopt_political(state, dump, flags=None, marker_states=None):
    """Manual adoption: positions from zones, activation from native H.k().

    The fourth mask column does not encode political activation. Legacy
    dumps without marker states preserve existing activation with a note.
    Elves and Shadow nations have permanently active single-face pieces.
    """
    import re as _re
    import log_replay as lr
    out = []
    pt = getattr(state, 'political', None)
    if pt is None:
        return out
    for pid, nation in lr.POLITICAL_MARKERS.items():
        pid = str(pid)
        zone = str((dump or {}).get(pid, '') or '')
        match = _re.match(r'^Political (\d)$', zone)
        if not match:
            continue
        box = 3 - int(match.group(1))
        if pt.boxes.get(nation) != box:
            out.append('political %s: box %s -> %d (client as-is)'
                       % (getattr(nation, 'value', nation), pt.boxes.get(nation), box))
            pt.boxes[nation] = box
        active = cb.political_marker_active(pid, marker_states)
        if active is None:
            out.append('political %s: native face unavailable; activation unchanged'
                       % getattr(nation, 'value', nation))
            continue
        if bool(pt.active.get(nation, False)) != active:
            out.append('political %s: %s (client as-is)'
                       % (getattr(nation, 'value', nation),
                          'ACTIVE' if active else 'passive'))
            pt.active[nation] = active
    return out


def _deck_attr(side, deck):
    from game_state import Side
    return '%s_%s_deck' % ('fp' if side is Side.FREE_PEOPLES else 'sh',
                           'char' if deck == 'character' else 'strat')


def adopt_hands(state, dump, pid_to_cid, mint, human_side, unknown_pids=None):
    """MANUAL RESYNC ONLY: both hands become the client's hand zones,
    identity by the commitment decode (pid -> cid); an undecodable
    human pid becomes a placeholder (mint), an undecodable AI pid is
    REPORTED (the AI's cards are always decoded).  A card the engine
    held but the client does not left the hand physically (played /
    discarded) -> its deck's face-up discard, so counts conserve.  An
    adopted cid never lingers in a deck list.  Hand-limit flag
    recomputed."""
    from board import sanitize
    from game_state import Side
    import game_logic as gl
    out = []
    retained_unknown = {}
    zones = {'fp hand': Side.FREE_PEOPLES, 'sa hand': Side.SHADOW}
    by_zone = {}
    for p, z in (dump or {}).items():
        k = sanitize(str(z)).strip().lower()
        if k in zones:
            by_zone.setdefault(k, []).append(str(p))
    for zname, side in zones.items():
        pids = sorted(by_zone.get(zname, []))
        hand_obj = state.hands.get(side)
        if hand_obj is None:
            continue
        old = list(hand_obj.hand)
        new, unknown = [], []
        for p in pids:
            cid = (pid_to_cid or {}).get(str(p))
            if cid is not None:
                new.append(str(cid))
                continue
            unknown.append(p)
            if side is human_side and mint is not None:
                deck = 'strategy'
                try:
                    import card_decoder as _cd
                    ip = int(p)
                    fp_t = getattr(_cd, '_FP_PID_TYPES', ())
                    sa_t = getattr(_cd, '_SA_PID_TYPES', ())
                    letter = None
                    if 106 <= ip < 106 + len(fp_t):
                        letter = fp_t[ip - 106]
                    elif 158 <= ip < 158 + len(sa_t):
                        letter = sa_t[ip - 158]
                    if letter == 'C':
                        deck = 'character'
                except Exception:
                    pass
                previous = (unknown_pids or {}).get(p)
                if (previous in old and str(previous).startswith('HU_')
                        and previous not in new and gl.card_deck_of(previous) == deck):
                    placeholder = previous
                else:
                    placeholder = mint(deck)
                new.append(placeholder)
                retained_unknown[p] = placeholder
        if unknown and side is not human_side:
            out.append('hand %s: %d AI pid(s) undecodable (%s) - not '
                       'adopted' % (side.name, len(unknown),
                                    ' '.join(unknown[:4])))
            continue
        if sorted(new) != sorted(str(c) for c in old):
            hand_obj.hand[:] = new
            out.append('hand %s: %d -> %d card(s) (client as-is%s)'
                       % (side.name, len(old), len(new),
                          '; %d placeholder(s)' % len(unknown)
                          if unknown else ''))
            for c in old:
                if c in new or str(c).startswith('HU_'):
                    continue
                try:
                    deck = gl.card_deck_of(c) or 'strategy'
                except Exception:
                    deck = 'strategy'
                key = '%s_%s' % ('fp' if side is Side.FREE_PEOPLES
                                 else 'sh', deck)
                pile = state.discard_face_up.setdefault(key, [])
                if c not in pile:
                    pile.append(c)
                    out.append('hand %s: %s left the hand -> %s '
                               'discard' % (side.name, c, key))
        for c in new:
            for attr in ('fp_char_deck', 'fp_strat_deck',
                         'sh_char_deck', 'sh_strat_deck'):
                lst = getattr(state, attr, None)
                if lst and c in lst:
                    lst.remove(c)
        hand_obj.needs_discard = len(new) > 6
    if unknown_pids is not None:
        unknown_pids.clear()
        unknown_pids.update(retained_unknown)
    return out


def adopt_decks(state, dump, mint, pid_to_cid=None):
    """MANUAL RESYNC ONLY: each engine deck's LENGTH becomes the count
    of pieces in the client's deck zone (identities stay engine-side;
    a shortfall is padded with placeholders, an excess drops
    placeholders first, then real cids from the top)."""
    from board import sanitize
    from game_state import Side
    deckz = {'fp char deck': ('fp_char_deck', 'character'),
             'fp strat deck': ('fp_strat_deck', 'strategy'),
             'sa char deck': ('sh_char_deck', 'character'),
             'sa strat deck': ('sh_strat_deck', 'strategy')}
    counts = {}
    for p, z in (dump or {}).items():
        k = sanitize(str(z)).strip().lower()
        if k in deckz:
            counts[k] = counts.get(k, 0) + 1
    out = []
    for k, (attr, deck) in deckz.items():
        lst = getattr(state, attr, None)
        if lst is None or k not in counts:
            continue
        want, have = counts[k], len(lst)
        known_here = {str(pid_to_cid[str(pid)]) for pid, zone in dump.items()
                      if pid_to_cid and str(pid) in pid_to_cid
                      and sanitize(str(zone)).strip().lower() == k}
        if len(known_here) > want:
            raise ValueError('deck %s has conflicting known card identities' % k)
        if have == want:
            continue
        if have > want:
            n = have - want
            for i in range(len(lst) - 1, -1, -1):
                if n <= 0:
                    break
                if str(lst[i]).startswith('HU_'):
                    del lst[i]
                    n -= 1
            for i in range(len(lst) - 1, -1, -1):
                if n <= 0:
                    break
                if lst[i] not in known_here:
                    del lst[i]
                    n -= 1
            if n:
                raise ValueError('deck %s cannot contract without dropping a known native card' % k)
        elif mint is not None:
            lst.extend(mint(deck) for _ in range(want - have))
        out.append('deck %s: %d -> %d (client as-is)' % (k, have,
                                                          len(lst)))
    return out


# the engine's open-decision fields (the driver's _PENDING_FIELDS twin;
# the driver passes its own tuple through `pending_fields`)
PENDING_FIELDS = (
    'pending_combat', 'pending_hunt', 'pending_guide_change',
    'pending_card_option', 'pending_card_followup',
    'pending_discard', '_pending_hunt_damage', '_declare_phase',
    '_eye_allocation_phase', '_reveal_place_due', '_hunt_reveal_pending',
    'pending_army_move_die', 'pending_muster_die',
)


def _disagreements(rows):
    """One key per DISAGREEMENT (family + region/box + side), not per
    row: the siege witness emits one row per unseated piece and the
    leader witness doubles companions/nazgul, which biased the count
    oracle toward whichever adoption deleted the most rows (BRCMP-4)."""
    return {str(r).split(':', 1)[0].strip() for r in (rows or ())}


def _deployed_delta(ai_module, before, state):
    """{(Nation, kind): n} pieces this adoption RESURRECTED (deployed
    after - deployed before, positive only); None without an ai
    module or a before-count."""
    if ai_module is None or before is None:
        return None
    after = ai_module._count_deployed_all(state)
    out = {}
    for nat in set(before) | set(after):
        b = before.get(nat) or {}
        a = after.get(nat) or {}
        for kind in set(b) | set(a):
            d = int(a.get(kind, 0)) - int(b.get(kind, 0))
            if d > 0:
                out[(nat, kind)] = d
    return out


def _revalidate_pendings(state, pending_fields=None):
    """After a committed adoption: an open decision whose option list
    the NEW board no longer supports (or that can no longer be read)
    is DROPPED WITH NOTICE - never left open against a board it does
    not describe, never cleared silently (BRCMP design 4)."""
    import ai as _ai
    try:
        dp = _ai.describe_pending(state)
    except Exception as e:
        dp = ('unreadable', repr(e)[:80])
    if dp is None:
        return []
    kind = None
    if isinstance(dp, dict):
        kind = str(dp.get('kind'))
        if dp.get('actions'):
            return []                # still answerable on the new board
        why = 'the adopted board offers it no legal option'
    else:
        why = 'it cannot be read on the adopted board (%s)' % dp[1]
    dropped = []
    for f in (pending_fields or PENDING_FIELDS):
        v = getattr(state, f, None)
        if v:
            setattr(state, f, 0 if f in ('_hunt_reveal_pending', '_pending_hunt_damage')
                    else False if isinstance(v, bool) else None)
            dropped.append(f)
    if not dropped:
        return []
    return ['open pending %s DROPPED after adoption: %s (cleared %s)'
            % (kind or 'decision', why, ', '.join(dropped))]


def _post_commit_coherence(state, pending_fields=None):
    """Derived state re-derived on the ADOPTED board, after the oracle
    accepted it (BRCMP design 4): the guide (rules-picked from the
    living Fellowship), a stale hand-limit flag, VP (derived on read
    from enemy_controlled - reported for the record), open pendings."""
    import game_logic as _gl
    out = []
    fs = state.fellowship
    comps = list(getattr(fs, 'companions', []) or [])
    g = getattr(fs, 'guide', None)
    if comps and (g is None or g not in comps):
        _gl.change_guide(state, allow_pending=True)
        out.append('guide re-derived on the adopted board: %s -> %s'
                   % (getattr(g, 'name', None),
                      getattr(getattr(fs, 'guide', None), 'name', None)))
    for s, h in (getattr(state, 'hands', None) or {}).items():
        if getattr(h, 'needs_discard', False) and len(h.hand) <= 6:
            h.needs_discard = False
            out.append('hand-limit flag cleared for %s (%d card(s))'
                       % (getattr(s, 'name', s), len(h.hand)))
    try:
        vp = _gl.calculate_victory_points(state)
        out.append('VP on the adopted board: %s' % (vp,))
    except Exception as e:
        out.append('VP re-derivation unavailable: %r' % (e,))
    out.extend(_revalidate_pendings(state, pending_fields))
    return out


def adopt_rings(state, dump):
    """Take all three physical ring counters; do not invent another use.

    Positions expose ownership, not the turn of an earlier transfer. Preserve
    known turn flags and mark a newly observed expenditure as used this turn.
    """
    available = held = disposed = 0
    for pid in ('249', '250', '251'):
        zone = _norm(dump.get(pid, ''))
        if zone in ('vilya slot', 'nenya slot', 'narya slot'):
            available += 1
        elif zone.startswith(('used ring', 'dark ring')):
            held += 1
        elif zone in ('sa casualties', 'sa dead', 'dead'):
            disposed += 1
        else:
            raise AdoptionRefused('ring %s has no recognized client box: %r'
                                  % (pid, dump.get(pid)))
    old_fp = int(state.fp_elven_rings)
    old_sh = int(state.sh_elven_rings)
    if available < old_fp:
        state.fp_ring_used_this_turn = True
    if disposed > 3 - old_fp - old_sh:
        state.sh_ring_used_this_turn = True
    state.fp_elven_rings, state.sh_elven_rings = available, held
    return ([] if (available, held) == (old_fp, old_sh) else
            ['elven rings: FP %d -> %d, Shadow %d -> %d'
             % (old_fp, available, old_sh, held)])


def adopt_action_dice(state, dump, dice_pids, phase=None, faces=None):
    """Recover registered dice by physical zone and exported public face.

    The driver's identity map accounts for removed dice and bonus dice. Current
    Action is treated as paid by this explicit recovery operation. Normal
    reconciliation never calls this while a player's gesture is in flight.
    """
    from game_state import Side
    from .action_die_zones import zone as die_zone, temporary
    if dice_pids is None:
        return []
    changes = []
    hunt = []
    seen = set()
    for side, dice in state.dice.items():
        for index, die in enumerate(dice):
            pid = str(dice_pids[(side, index)])
            if pid in seen:
                raise AdoptionRefused('duplicate action-die identity: %s' % pid)
            seen.add(pid)
            zone = die_zone(_norm(dump.get(pid, '')))
            pool = 'fp dice' if side is Side.FREE_PEOPLES else 'sa dice'
            dead = ('fp casualties', 'fp dead') if side is Side.FREE_PEOPLES else ('sa casualties', 'sa dead')
            in_hunt = zone in ('the hunt for the ring', 'hunt box', 'hunt')
            if zone == pool:
                used = False
                # A recovered die awaits this turn's native roll. Keeping its
                # previous face would make adopt_rolled_faces skip that roll.
                if die.used and phase in ('eyes', 'roll'):
                    die.current = None
            elif zone in dead or zone == 'dead' or temporary(zone) or in_hunt:
                used = True
            else:
                raise AdoptionRefused('action die %s has no recognized client box: %r'
                                      % (pid, dump.get(pid)))
            # The action phase has already rolled. Native manual edits or
            # missed face changes must survive a board-based recovery. During
            # turn preparation the visible old faces are not this turn's roll.
            if phase == 'actions' and pid in (faces or {}):
                from .gesture_match import FACE_LETTER
                from game_state import ActionDieFace
                face = ActionDieFace[next(name for name, letter in FACE_LETTER.items()
                                         if letter == faces[pid])]
                if die.current != face:
                    changes.append('action die %s: face adopted as %s' % (pid, faces[pid]))
                    die.current = face
            if used != die.used or in_hunt != (die in state.hunt_box):
                changes.append('action die %s: %s (%s)' %
                               (pid, 'spent' if used else 'available', zone))
            die.used = used
            if in_hunt:
                hunt.append(die)
    state.hunt_box[:] = hunt
    return changes


def adopt_client_board(state, dump, dynamic_types=None, ai_module=None,
                       guide_hook=None, dump_path=None,
                       owed_tracks=None, owed_pawn=None,
                       manual=False, pid_to_cid=None, mint=None,
                       flags=None, human_side=None, skip_political=False,
                       consent=False, pending_fields=None, dice_pids=None,
                       dice_phase=None, pending_companions=()):
    """Adopt the client's physical board into the ENGINE.

    guide_hook: optional zero-arg callable returning a note string (the
    driver passes its reconcile_guide_from_client, which routes the
    guide through game_logic.change_guide so the RULES still validate
    the pick).
    dump_path: the client dump file; when given, the census diffs
    against an INDEPENDENT read that must agree with `dump`.
    manual: the BL-147 lever (client AS IS across ALL families).
    consent: the human said yes to the printed rows, or the lenient
    path (USER 2026-08-22 'as given') - the pieces AND the rules
    flags they imply (siege drop, capture revert, casualty ledger)
    are taken; tracks (rules-derived: hunt damage, guide abilities)
    follow too.  Neither: PIECES ONLY - a rules flag never changes on
    a plain adoption (BRCMP-4).

    ONE TRANSACTION (BRCMP-7): every leg runs inside the snapshot; a
    failing leg or a worse census rolls ALL of it back.  The oracle
    counts DISAGREEMENTS, not rows.  After the commit, derived state
    is re-derived and an open pending the new board cannot answer is
    dropped with notice.

    Returns (report_lines, n_before, n_after).  Refuses (AdoptionRefused,
    engine untouched) a dump below MIN_DUMP_PIECES or unstable across
    two reads.
    """
    report = []
    census_dump = _stable_dump(dump, dump_path)
    from .client_protocol import read_marker_states, read_action_die_faces
    marker_states = read_marker_states(dump_path) if dump_path else None
    dice_faces = read_action_die_faces(dump_path) if dump_path else None
    census = (lambda st: cb.diff_engine_vs_dump(
        st, census_dump, dynamic_types=dynamic_types,
        ai_module=ai_module, marker_states=marker_states,
        pid_to_cid=pid_to_cid))
    before = census(state)
    if before is None:
        raise AdoptionRefused(
            'ADOPTION REFUSED: the census cannot judge this dump '
            '(diff_engine_vs_dump returned None); engine left untouched')
    as_given = bool(manual or consent)
    snap = _snapshot_armies(state)
    # Native PID -> opaque slot belongs to bridge recovery, not engine state.
    # Keep it provisional until every adoption leg and census check accepts.
    hand_model = getattr(mint, '__self__', None)
    unknown_pids = dict(getattr(hand_model, '_recovery_unknown_pids', {}))
    for receipt in getattr(hand_model, '_draw_receipts', ()):
        observation = receipt.get('observation')
        if (observation and observation.get('pid') is not None
                and observation.get('cid') is None and receipt.get('placeholder')
                and not receipt.get('cancelled') and not receipt.get('undone')):
            unknown_pids[str(observation['pid'])] = receipt['placeholder']
    dep0 = (ai_module._count_deployed_all(state)
            if ai_module is not None else None)
    try:
        report.extend(adopt_units(state, census_dump, dynamic_types))
        report.extend(adopt_leaders(state, census_dump, dynamic_types))
        if manual:
            from .promotion_recovery import adopt_public_characters
            report.extend(adopt_public_characters(state, census_dump, marker_states))
        # companions BEFORE the guide: the guide must be picked from a
        # Fellowship that already reflects who is still in it
        report.extend(adopt_companions(state, census_dump, dynamic_types,
                                       pending_companions=pending_companions))
        # named minions relocate by object (never fabricated)
        report.extend(adopt_minions(state, census_dump, manual=manual))
        if manual:
            report.extend(adopt_control_markers(state, census_dump))
        # sieges BEFORE the guide/census: the flag gates siege-only
        # rules and the census's own siege-zone check reads it; a
        # DROP is a rules-flag change - consent forms only
        report.extend(adopt_sieges(state, census_dump, dynamic_types,
                                   allow_drop=as_given))
        # the pawn's REGION/Mordor box (USER 2026-08-24: manual-edit
        # resyncs never reached the fellowship position)
        report.extend(adopt_fellowship_position(state, census_dump,
                                                owed=None if manual else owed_pawn,
                                                manual=manual))
        if manual:
            hidden = cb.fellowship_marker_hidden(marker_states)
            if hidden is not None and state.fellowship.hidden != hidden:
                state.fellowship.hidden = hidden
                report.append('fellowship visibility adopted: %s' %
                              ('hidden' if hidden else 'revealed'))
        if guide_hook is not None:
            note = guide_hook()
            if note:
                report.append(note)
        if as_given:
            # BL-2 (2026-08-25): the STABLE census dump, not the raw
            # one, and the owed-track hold so an engine change never
            # loses to a lagging marker.  Tracks are rules-derived:
            # taken only when the human asked (manual / consent /
            # lenient); the plain path leaves them as census rows.
            for _tl in adopt_tracks(state, census_dump,
                                    owed=None if manual else owed_tracks):
                report.append('track adopted: %s' % _tl)
            report.extend(adopt_rings(state, census_dump))
            report.extend(adopt_action_dice(state, census_dump, dice_pids, dice_phase, dice_faces))
            # BL-21(3) (2026-08-25, the Woodland Realm chimera): an
            # adoption can resurrect a garrison the engine had
            # eliminated while the capture + VP stand - a rules-
            # impossible state nothing else can see.  Reverted under
            # a consent form, loudly.
            report.extend(capture_coherence(state))
        else:
            for _iv in cb.rules_invariant_rows(state):
                if 'captured-with-garrison' in str(_iv):
                    report.append('capture INCOHERENT (report-only, a '
                                  'manual resync or your consent '
                                  'reverts it): %s' % (_iv,))
        # BL-21(5): the FP casualty ledger - lowered only by what THIS
        # adoption resurrected / to restore conservation (auto), or
        # taken exactly (manual); never on the dead pile alone
        report.extend(reconcile_fp_casualties(
            state, census_dump, dynamic_types, manual=manual,
            resurrected=(_deployed_delta(ai_module, dep0, state)
                         if as_given else None)))
        if manual:
            # USER 2026-08-25: the MANUAL lever takes the client AS
            # IS - politics (boxes + activation), both hands, deck
            # counts - the families the auto-reconcile deliberately
            # leaves engine-side.  A failing family rolls the whole
            # transaction back (never a silent partial adoption).
            # self-repair (USER 2026-08-28): everything client-as-is
            # EXCEPT the political track, which stays engine truth and
            # is pushed to the client by _political_recon afterwards
            if not skip_political:
                report.extend(adopt_political(state, census_dump,
                                              marker_states=marker_states))
            from .table_recovery import adopt as adopt_table_cards
            report.extend(adopt_table_cards(state, census_dump, pid_to_cid))
            report.extend(adopt_hands(state, census_dump, pid_to_cid,
                                      mint, human_side, unknown_pids))
            report.extend(adopt_decks(state, census_dump, mint, pid_to_cid))
    except Exception as e:
        _restore_armies(state, snap)
        report.append('unit adoption failed, rolled back: %r' % (e,))
        return report, len(before), len(before)
    after = census(state) or []
    kb, ka = _disagreements(before), _disagreements(after)
    if len(ka) > len(kb):
        _restore_armies(state, snap)
        report.append('ADOPTION DECLINED: disagreements would have gone '
                      '%d -> %d (rows %d -> %d), so nothing was written'
                      % (len(kb), len(ka), len(before), len(after)))
        return report, len(before), len(before)
    # committed: derived state re-derived on the NEW board
    report.extend(_post_commit_coherence(state, pending_fields))
    if manual and hand_model is not None:
        hand_model._recovery_unknown_pids = unknown_pids
    report.append('board adopted: divergences %d -> %d'
                  % (len(before), len(after)))
    return report, len(before), len(after)
