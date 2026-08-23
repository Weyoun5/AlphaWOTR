"""Client board-dump reader + client-vs-engine witnesses (LIFT-B).

LIFTED from wotr_bridge/ai_move_server.py (read-only donor; nothing is
imported from it):
  read_board            <- _read_client_board          (:10946-10967)
  diff_engine_vs_dump   <- verify_client_vs_engine     (:11042-11510)
  card_census           <- _card_census + hand-zone detail
                                                       (:12952-13042)
  proj_board            <- _proj_board                 (:3135-3179)
  board_delta           <- _board_delta                (:3181-3281)
  unit_type             <- emitter._unit_type shape, decoupled from the
                           replayer object (A17: dynamic_types threaded
                           through log_replay.get_unit_type_from_piece_id;
                           PIECE_NAME_ALIASES honored via names_for_pid)

DESIGN RULES (REBUILD_PLAN.md) THAT BIND THIS MODULE:
  * Witnesses NOTICE, the engine OWNS: every function here READS state
    and dump and returns rows/strings.  Nothing here mutates the engine
    state, the dump, or any caller structure.
  * Decoupled: explicit arguments only.  The monolith's god-object
    fields (self.eng / self.em / self._census_notes / self._revoked_pids
    / self._client_initial) became parameters; driver-state-dependent
    checks run only when their inputs are passed.
  * Client fact #14: a full board dump is ~428 pieces (measured 428 on
    the blind_run specimen).  <400 pieces or two reads that disagree =
    torn mid-write; read_board re-reads and returns None rather than a
    torn dict.  (The donor _read_client_board had NO torn guard.)
  * Client fact #3: nothing here branches on a nick; the dump carries
    zones and pids only.
  * Client fact #10: card IDENTITY never comes from the dump or any
    alphabetical pid map.  card_census compares hand SIZES from the
    dump; identity rows appear only when the caller passes
    commitment-decoded cids (hand_cids).

DEVIATIONS from the donor (each documented at its site):
  * dice count check accepts an optional granted_pids set so a
    never-granted bonus die parked in a Dead box is not miscounted as
    'used' (the donor only shielded the ROSTER check this way).
  * the census-window classifier (_hand_window) is NOT lifted: it reads
    driver flow state (chain-open plays, turn-start draws).  Rows are
    always emitted; window tagging is the driver's business.
  * proj_board imports wotr_bridge/parity_test (summarize) - the donor
    did the same; re-transcribing summarize here would create a second
    answerer for the projection (defect-9 shape).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BRIDGE = os.path.abspath(os.path.join(_HERE, '..'))
_SRC = os.path.abspath(os.path.join(_HERE, '..', '..', 'src'))
for _p in (_SRC, _BRIDGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Client fact #14: full dump ~428 pieces; below this it is torn.
MIN_DUMP_PIECES = 400

# named minion pids (log_replay.MINION_PIECES keys; kept as the
# engine-name -> pid mapping the donor used at :11177-11179)
_MINION_PIDS = (('253', 'witch'), ('254', 'saruman'), ('255', 'mouth'))

_UNIT_KINDS = ('regular', 'elite', 'leader', 'nazgul', 'companion',
               'minion')


def _nv(x):
    return str(getattr(x, 'value', x)).lower()


def _strip_sh(z):
    return z[:-11] if z.endswith(' stronghold') else z


def unit_type(pid, dynamic_types=None, side_hint=None):
    """(nation, kind) for a client piece id, kind lowercased.

    A17: fungible pids resolve through
    log_replay.get_unit_type_from_piece_id WITH the caller's
    dynamic_types (runtime swap assignments); named figures resolve
    through the COMPANIONS/MINION_PIECES tables (PIECE_NAME_ALIASES is
    honored by names_for_pid wherever names are compared).  Mirrors
    emitter._unit_type (:693-713) minus the replayer object."""
    import log_replay as lr
    pid = str(pid)
    if pid in lr.COMPANIONS:
        return lr.COMPANIONS[pid], 'companion'
    if pid in lr.MINION_PIECES:
        return lr.MINION_PIECES[pid], 'minion'
    nat, ut = lr.get_unit_type_from_piece_id(pid, dynamic_types)
    if ut is None and side_hint:
        nat, ut = lr.get_unit_type_from_piece_id(
            pid, dynamic_types, side_hint=side_hint)
    if ut is not None:
        ut = str(getattr(ut, 'value', ut)).lower()
    return nat, ut


def _parse_dump(path):
    """One raw parse of the dump file -> {pid: decoded zone name} or
    None.  Donor: _read_client_board (:10946-10967), unchanged."""
    import log_replay as lr
    if not os.path.exists(path):
        return None
    out = {}
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            for line in f:
                if line.startswith('#HAND') \
                        and '<blinded>' not in line:
                    # WAVE-J (ledger F-06/A-23, rule 8 FAIL-CLOSED):
                    # an un-blinded whole-hand snapshot in a dump
                    # means the blinding flag is missing - no reader
                    # may consume this file.  Raise, never skip: a
                    # silent None reads as 'no dump yet'.
                    raise RuntimeError(
                        'BLINDING VIOLATION: un-blinded #HAND '
                        'snapshot in the board dump (missing '
                        '-Dai.feed.aiside)')
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 3:
                    continue
                region = lr.decode_location(parts[1].strip('<~>'))
                rk = str(region) if region else parts[1]
                for pid in parts[2].split():
                    out[pid] = rk
    except RuntimeError:
        raise               # the blinding violation must stay LOUD
    except Exception:
        return None
    return out or None


def read_board(path, min_pieces=MIN_DUMP_PIECES, retries=3, delay=0.05,
               _parse=None):
    """Read the client's board dump -> {pid: zone name}, torn-safe.

    Client fact #14: a full dump is ~428 pieces.  A parse with fewer
    than min_pieces, or two consecutive parses that disagree, is a dump
    torn mid-write: re-read up to `retries` times, then return None
    (callers treat None exactly like 'no dump yet' - the donor's
    contract).  `_parse` is an injection point for the torn-logic
    tests; production callers never pass it."""
    import time
    parse = _parse or (lambda: _parse_dump(path))
    prev = None
    for attempt in range(max(1, retries)):
        cur = parse()
        if cur is not None and len(cur) >= min_pieces \
                and prev is not None and cur == prev:
            return cur
        prev = cur
        if attempt + 1 < max(1, retries) and delay:
            time.sleep(delay)
    return None


# ---------------------------------------------------------------------
# board projection / delta (donor :3135-3281, verbatim semantics)
# ---------------------------------------------------------------------
def proj_board(st):
    """Comparable board projection for gesture matching and the census.
    Donor: _proj_board (:3135-3179).  Carries leader IDENTITIES per
    region, DICE FACES, RINGS and HANDS (RED EX-2/EX-3); deck lengths
    stay OUT (a client-model's decks are fiction; draws are direct-
    mapped by pid)."""
    import parity_test as pt
    from game_state import Side as _S
    from board import sanitize
    d = pt.summarize(st)
    named = {}
    for k, a in st.armies.items():
        nm = tuple(sorted(
            str(getattr(l9, 'name', '')) for l9 in a.leaders
            if getattr(l9, 'name', None)
            and type(l9).__name__ in ('Companion', 'Minion')))
        if nm:
            named[sanitize(k)] = nm
    dice = {}
    for sd in (_S.FREE_PEOPLES, _S.SHADOW):
        ds = st.dice.get(sd, [])
        dice[sd.name] = (
            tuple(sorted(str(getattr(x.current, 'name', None))
                         for x in ds if not x.used)),
            sum(1 for x in ds if x.used))
    hands = {}
    for sd in (_S.FREE_PEOPLES, _S.SHADOW):
        hands[sd.name] = tuple(sorted(
            str(c) for c in st.hands[sd].hand
            if isinstance(c, str)))
    return {'armies': dict(d.get('armies') or ()),
            'political': d.get('political'),
            'fellowship': d.get('fellowship'),
            'named_leaders': named,
            'dice': dice,
            'rings': (int(getattr(st, 'fp_elven_rings', 3)),
                      int(getattr(st, 'sh_elven_rings', 0))),
            'hands': hands,
            'table': d.get('table')}


def board_delta(pre, post):
    """{region: (before, after)} for changed army rows + per-nation
    political INCREMENTS + identity-aware fellowship/leader/dice/ring/
    hand/table deltas.  Donor: _board_delta (:3181-3281), verbatim
    semantics (increment-based so a one-tick view offset cannot refuse
    a legal action)."""
    a0, a1 = proj_board(pre), proj_board(post)
    out = {}
    keys = set(a0['armies']) | set(a1['armies'])
    for k in keys:
        b, a = a0['armies'].get(k), a1['armies'].get(k)
        if b == a:
            continue
        bu = dict(b[0]) if b else {}
        au = dict(a[0]) if a else {}
        ud = tuple(sorted(
            (t, au.get(t, 0) - bu.get(t, 0))
            for t in set(bu) | set(au)
            if au.get(t, 0) != bu.get(t, 0)))
        ld = (a[1] if a else 0) - (b[1] if b else 0)
        if ud or ld:
            out[k] = (ud, ld)
    p0 = dict(a0['political'] or ())
    p1 = dict(a1['political'] or ())
    pd = tuple(sorted((n, p1.get(n, 0) - p0.get(n, 0))
                      for n in set(p0) | set(p1)
                      if p1.get(n, 0) != p0.get(n, 0)))
    if pd:
        out['__political__'] = pd
    f0, f1 = a0.get('fellowship'), a1.get('fellowship')
    if f0 and f1 and f0 != f1:
        c0, c1 = set(f0[6]), set(f1[6])
        fd = {}
        if c0 - c1:
            fd['left'] = tuple(sorted(c0 - c1))
        if c1 - c0:
            fd['joined'] = tuple(sorted(c1 - c0))
        if f0[5] != f1[5]:
            fd['guide'] = f1[5]
        if f0[0] != f1[0]:
            fd['progress'] = f1[0] - f0[0]
        if f0[1] != f1[1]:
            fd['corruption'] = f1[1] - f0[1]
        if f0[2] != f1[2]:
            fd['hidden'] = f1[2]
        if len(f0) > 7 and len(f1) > 7 and f0[7] != f1[7]:
            fd['region'] = f1[7]
        if fd:
            out['__fellowship__'] = tuple(sorted(fd.items()))
    n0 = a0.get('named_leaders') or {}
    n1 = a1.get('named_leaders') or {}
    nd = tuple(sorted(
        (k, n0.get(k, ()), n1.get(k, ()))
        for k in set(n0) | set(n1) if n0.get(k) != n1.get(k)))
    if nd:
        out['__named__'] = nd
    d0, d1 = a0.get('dice') or {}, a1.get('dice') or {}
    dd = tuple(sorted((k, d0.get(k), d1.get(k))
                      for k in set(d0) | set(d1)
                      if d0.get(k) != d1.get(k)))
    if dd:
        out['__dice__'] = dd
    r0, r1 = a0.get('rings'), a1.get('rings')
    if r0 is not None and r1 is not None and r0 != r1:
        out['__rings__'] = (r1[0] - r0[0], r1[1] - r0[1])
    h0, h1 = a0.get('hands') or {}, a1.get('hands') or {}
    hd = {}
    for k in set(h0) | set(h1):
        s0, s1 = set(h0.get(k, ())), set(h1.get(k, ()))
        if s0 != s1:
            hd[k] = (tuple(sorted(s0 - s1)),
                     tuple(sorted(s1 - s0)))
    if hd:
        out['__hands__'] = tuple(sorted(hd.items()))
    t0, t1 = a0.get('table'), a1.get('table')
    if t0 is not None and t1 is not None and t0 != t1:
        out['__table__'] = (t0, t1)
    return out


# ---------------------------------------------------------------------
# the direct engine<->client witness (donor :11042-11510)
# ---------------------------------------------------------------------
def _leader_side(leader):
    """'fp' | 'shadow' of a leader object (donor :9976-9984)."""
    from game_state import Nazgul
    import emitter as em
    if isinstance(leader, Nazgul):
        return 'shadow'
    nat = getattr(leader, 'nation', None)
    if nat is not None:
        return em._side_of_nation(nat)
    return 'fp'      # companions carry no nation


def diff_engine_vs_dump(state, dump, dynamic_types=None, ai_module=None,
                        initial_dump=None, granted_pids=None,
                        revoked_pids=None):
    """DIRECT engine<->client verification (no parser in between):
    armies per region, leaders per region+side, nazgul counts + named
    minions, dice zones, deck sizes, siege sub-boxes, companions,
    control markers, elven rings; optional per-nation FORCE POOLS
    (needs ai_module) and PIECE CONSERVATION (needs initial_dump).

    Returns a list of issue strings; empty = the client board IS the
    engine state.  Never mutates anything (witnesses notice, the
    engine owns).  Donor: verify_client_vs_engine (:11042-11510),
    decoupled: dynamic_types replaces the replayer, ai_module replaces
    self.ai, initial_dump replaces self._client_initial, granted/
    revoked pid sets replace the die-registration driver fields."""
    if not dump or state is None:
        return None
    from board import sanitize
    from game_state import Side
    import board as bd
    import emitter as em
    st = state
    regions = bd.regions
    issues = []
    cl_units, cl_leaders, cl_pool = {}, {}, {}
    cl_naz, cl_minion = {}, {}
    cl_dice, cl_decks = {}, {}
    fp_dice, sh_dice = set(em._FP_DIE_IDS), set(em._SH_DIE_IDS)
    deckz = {'fp char deck', 'fp strat deck',
             'sa char deck', 'sa strat deck'}
    rvd = set(revoked_pids or ())
    for pid, zone in dump.items():
        z = sanitize(str(zone))
        if pid in fp_dice or pid in sh_dice:
            if pid in rvd:
                # revocation-display window: the revoked die's PIECE
                # rides the dead pile until the next recovery while the
                # engine dropped the die entirely (donor :11077-11085)
                continue
            if granted_pids is not None and int(pid) >= 300 \
                    and pid not in granted_pids:
                # DEVIATION (documented): a never-granted BONUS die
                # parks in a Dead/Reinf box from setup; counting it as
                # 'used' misreads the box.  The donor shielded only the
                # roster check this way (:11249-11259); with a granted
                # set available the count check gets the same shield.
                continue
            s = 'fp' if pid in fp_dice else 'shadow'
            if z in ('fp dice', 'sa dice'):
                box = 'pool'
            elif 'hunt' in z:
                box = 'hunt'
            elif z in ('used', 'fp dead', 'sa dead'):
                # the client's kill routes a spent die to the side's
                # Dead zone - same meaning as the Used box
                box = 'used'
            else:
                continue   # bonus die not yet in play (reinf box)
            cl_dice[(s, box)] = cl_dice.get((s, box), 0) + 1
            continue
        if z in deckz:
            cl_decks[z] = cl_decks.get(z, 0) + 1
            continue
        try:
            nat, ut = unit_type(pid, dynamic_types)
        except Exception:
            continue
        natv = _nv(nat)
        base = _strip_sh(z)
        if base in regions:
            if ut in ('regular', 'elite'):
                k = (base, (natv, ut))
                cl_units[k] = cl_units.get(k, 0) + 1
            elif ut in ('leader', 'nazgul', 'minion', 'companion'):
                s = ('shadow' if ut in ('nazgul', 'minion')
                     else 'fp' if ut == 'companion'
                     else em._side_of_nation(nat))
                cl_leaders[(base, s)] = cl_leaders.get((base, s), 0) + 1
                if ut == 'nazgul':
                    cl_naz[base] = cl_naz.get(base, 0) + 1
                elif ut == 'minion':
                    cl_minion[pid] = base
        elif z == 'fp reinf' and ut in ('regular', 'elite'):
            k = ('fp', (natv, ut))
            cl_pool[k] = cl_pool.get(k, 0) + 1
        elif z in ('sa reinf', 'sa dead') and ut in ('regular', 'elite'):
            k = ('shadow', (natv, ut))
            cl_pool[k] = cl_pool.get(k, 0) + 1
    # engine views
    en_units, en_leaders = {}, {}
    for key, army in st.armies.items():
        k = sanitize(str(key))
        for u in army.units:
            kk = (k, (_nv(u.nation), _nv(u.kind)))
            en_units[kk] = en_units.get(kk, 0) + 1
        for ld in army.leaders:
            s = _leader_side(ld)
            en_leaders[(k, s)] = en_leaders.get((k, s), 0) + 1
    for kk in sorted(set(en_units) | set(cl_units)):
        a, b = en_units.get(kk, 0), cl_units.get(kk, 0)
        if a != b:
            issues.append('units %s %s/%s: engine=%d client=%d'
                          % (kk[0], kk[1][0], kk[1][1], a, b))
    for kk in sorted(set(en_leaders) | set(cl_leaders)):
        a, b = en_leaders.get(kk, 0), cl_leaders.get(kk, 0)
        if a != b:
            issues.append('leaders %s (%s): engine=%d client=%d'
                          % (kk[0], kk[1], a, b))
    # nazgul per-region counts + named minions (wrong-piece renders)
    try:
        en_naz, en_min = {}, {}
        for k9, a9 in st.armies.items():
            rg = sanitize(str(k9))
            for ld in getattr(a9, 'leaders', []) or []:
                t9 = type(ld).__name__
                nm = str(getattr(ld, 'name', '')).lower()
                if t9 == 'Nazgul':
                    en_naz[rg] = en_naz.get(rg, 0) + 1
                elif t9 == 'Minion' or nm:
                    if 'witch' in nm:
                        en_min['253'] = rg
                    elif 'saruman' in nm:
                        en_min['254'] = rg
                    elif 'mouth' in nm:
                        en_min['255'] = rg
        for rg in sorted(set(en_naz) | set(cl_naz)):
            a, b = en_naz.get(rg, 0), cl_naz.get(rg, 0)
            if a != b:
                issues.append('nazgul %s: engine=%d client=%d'
                              % (rg, a, b))
        for mp, mn in (('253', 'witch-king'), ('254', 'saruman'),
                       ('255', 'mouth of sauron')):
            er, cr = en_min.get(mp), cl_minion.get(mp)
            if er != cr and (er or cr):
                issues.append('minion %s (%s): engine@%s client@%s'
                              % (mn, mp, er or '-', cr or '-'))
    except Exception as e:
        issues.append('character check unavailable: %r' % (e,))
    # dice zones
    for s, tag in ((Side.FREE_PEOPLES, 'fp'), (Side.SHADOW, 'shadow')):
        total = len(st.dice.get(s, []))
        hunt = sum(1 for d in st.hunt_box
                   if getattr(d, 'side', None) is s)
        used = sum(1 for d in st.dice.get(s, []) if d.used) - hunt
        pool = total - used - hunt
        for box, want in (('pool', pool), ('used', used),
                          ('hunt', hunt)):
            got = cl_dice.get((tag, box), 0)
            if got != want:
                issues.append('dice %s %s: engine=%d client=%d'
                              % (tag, box, want, got))
    # decks
    for name, lst in (('fp char deck', st.fp_char_deck),
                      ('fp strat deck', st.fp_strat_deck),
                      ('sa char deck', st.sh_char_deck),
                      ('sa strat deck', st.sh_strat_deck)):
        a, b = len(lst or []), cl_decks.get(name, 0)
        if a != b:
            issues.append('deck %s: engine=%d client=%d' % (name, a, b))
    # force pools per nation+kind (max - deployed - FP casualties);
    # only PHANTOM reserve is a failure (client has MORE pool pieces
    # than the engine allows).  Runs only with an ai module in hand.
    if ai_module is not None:
        try:
            dep = ai_module._count_deployed_all(st)
            fp_cas = getattr(st, 'fp_casualties', None) or {}
            for nat, maxes in ai_module.NATION_MAX_FIGURES.items():
                sside = ('fp' if ai_module._side_of_nation(nat)
                         == Side.FREE_PEOPLES else 'shadow')
                d0 = dep.get(nat, {})
                cas = fp_cas.get(nat, {})
                for kind in ('regular', 'elite'):
                    avail = maxes.get(kind, 0) - d0.get(kind, 0)
                    if sside == 'fp':
                        avail -= cas.get(kind, 0)
                    got = cl_pool.get((sside, (_nv(nat), kind)), 0)
                    if got > avail:
                        issues.append(
                            'pool %s %s/%s: engine_avail=%d client=%d'
                            ' (PHANTOM)'
                            % (sside, _nv(nat), kind, avail, got))
        except Exception as e:
            issues.append('pool check unavailable: %r' % (e,))
    # SIEGE ZONES: inside-vs-outside a besieged settlement are
    # DIFFERENT client zones of one region; garrison side INSIDE the
    # stronghold box, besieger OUTSIDE (donor :11270-11311)
    try:
        for breg in sorted(
                {sanitize(str(b9)) for b9 in
                 (getattr(st, 'besieged_strongholds', set()) or set())}):
            from combat import garrison_side_for
            g = garrison_side_for(st, breg)
            gs_side = 'fp' if g.name == 'FREE_PEOPLES' else 'shadow'
            for pid9, z9 in dump.items():
                zz = sanitize(str(z9))
                base9 = _strip_sh(zz)
                if base9 != breg:
                    continue
                try:
                    n9, u9 = unit_type(pid9, dynamic_types)
                except Exception:
                    continue
                if u9 not in _UNIT_KINDS:
                    continue
                ps = ('shadow' if u9 in ('nazgul', 'minion')
                      else 'fp' if u9 == 'companion'
                      else em._side_of_nation(n9))
                inside = zz.endswith(' stronghold')
                if ps == gs_side and not inside:
                    issues.append(
                        'siege %s: garrison-side piece %s OUTSIDE the '
                        'stronghold' % (breg, pid9))
                elif ps != gs_side and inside:
                    issues.append(
                        'siege %s: besieger piece %s INSIDE the '
                        'stronghold' % (breg, pid9))
    except Exception as e:
        issues.append('siege-zone check unavailable: %r' % (e,))
    # PIECE CONSERVATION vs the game-start dump (a vanished pid is a
    # DELETED piece, a new pid is a phantom mint)
    if initial_dump:
        interesting = set()
        for pid in initial_dump:
            try:
                _n9, u9 = unit_type(pid, dynamic_types)
            except Exception:
                continue
            if u9 in _UNIT_KINDS:
                interesting.add(pid)
        gone = sorted(p for p in interesting if p not in dump)
        if gone:
            issues.append(
                'CONSERVATION: %d piece(s) VANISHED from the client '
                'board: %s' % (len(gone), ' '.join(gone[:12])))
    # companion zone classes: in-fellowship -> Fellowship/Guide box;
    # separated -> its map region; dead/out -> casualty or reinf zones
    try:
        import log_replay as lr
        fs = st.fellowship
        in_fs = {str(c.name) for c in
                 getattr(fs, 'companions', []) or []}
        g9 = getattr(fs, 'guide', None)
        if g9 is not None:
            in_fs.add(str(g9.name))
        sep = {}
        for k9, army9 in st.armies.items():
            for ld9 in getattr(army9, 'leaders', []) or []:
                nm9 = str(getattr(ld9, 'name', ''))
                if nm9:
                    sep[nm9] = sanitize(str(k9))
        deadish = {'fp casualties', 'sa casualties', 'dead',
                   'fp dead', 'sa dead'}
        fsbox = {'fellowship of the ring', 'guide of the fellowship',
                 'fellowship box', 'guide box'}
        for pid9, nm_t9 in sorted(lr.COMPANIONS.items()):
            z9 = dump.get(str(pid9))
            if z9 is None:
                continue
            zs = sanitize(str(z9))
            names9 = set(lr.names_for_pid(pid9) or (nm_t9,))
            if names9 & in_fs:
                if zs not in fsbox:
                    issues.append(
                        'companion %s: engine=in fellowship, client '
                        'zone=%s' % (nm_t9, zs))
            elif names9 & set(sep):
                want = next(sep[n] for n in names9 if n in sep)
                base9 = _strip_sh(zs)
                if base9 != want:
                    issues.append(
                        'companion %s: engine=separated at %s, client '
                        'zone=%s' % (nm_t9, want, zs))
            else:
                if zs not in deadish and zs not in (
                        'fp reinf', 'fp reinforcements',
                        'sa reinf', 'sa reinforcements'):
                    issues.append(
                        'companion %s: engine=dead/out, client zone=%s'
                        % (nm_t9, zs))
    except Exception as e:
        issues.append('companion check unavailable: %r' % (e,))
    # CONTROL MARKERS vs the engine's enemy_controlled sets.  Marker
    # supply is PHYSICAL (10 SH / 6 FP): controlling more regions than
    # markers is correct board state, so a shortfall of that size is
    # tolerated, never an issue row (donor :11400-1474).
    try:
        for key9, rng9 in (('sh', range(411, 421)),
                           ('fp', range(405, 411))):
            sd9 = Side.SHADOW if key9 == 'sh' else Side.FREE_PEOPLES
            want9 = {sanitize(str(k)) for k in
                     st.enemy_controlled.get(sd9, set())}
            have9, cnt9, off9 = set(), {}, []
            for i9 in rng9:
                z9 = sanitize(str(dump.get('%03d' % i9, '')))
                b9 = _strip_sh(z9)
                if b9 in regions:
                    have9.add(b9)
                    cnt9[b9] = cnt9.get(b9, 0) + 1
                elif z9 not in ('fp reinf', 'fp reinforcements',
                                'sa reinf', 'sa reinforcements', ''):
                    off9.append('%03d@%s' % (i9, z9))
            supply9 = len(list(rng9))
            missing9 = sorted(want9 - have9)
            short9 = max(0, len(want9) - supply9)
            for reg9 in missing9[short9:]:
                issues.append('control %s: engine holds %s, no client '
                              'marker' % (key9, reg9))
            for reg9 in sorted(have9 - want9):
                issues.append('control %s: client marker at %s, engine '
                              'does not' % (key9, reg9))
            for reg9, n9 in sorted(cnt9.items()):
                if n9 > 1:
                    issues.append('control %s: %d DUPLICATE markers at '
                                  '%s' % (key9, n9, reg9))
            if off9:
                issues.append('control %s: marker(s) outside pool/map: '
                              '%s' % (key9, ' '.join(off9)))
    except Exception as e:
        issues.append('control check unavailable: %r' % (e,))
    # WAVE-J: the CLIENT'S OWN TURN COUNTER (fact 15, '<~Game.7~>N').
    # The hand-built recover bypassed the client's menu handler and
    # the marker NEVER ticked (wjg7: client 1 vs engine 6, invisible
    # to every witness).  One contract: after both sides recover,
    # client turn == engine turn.
    try:
        import re as _re9
        tm9 = None
        for _pid9, _z9 in dump.items():
            _m9 = _re9.search(r'<~Game\.7~>(\d+)', str(_z9))
            if _m9:
                tm9 = int(_m9.group(1))
                break
        if tm9 is not None \
                and int(getattr(st, 'turn', 0) or 0) != tm9:
            issues.append('turn counter: engine=%s client=%s'
                          % (getattr(st, 'turn', '?'), tm9))
    except Exception:
        pass
    # ELVEN-RING BOXES: pids 249-251 in the Vilya/Nenya/Narya Slots =
    # available, Used Ring / Dark Ring = Sauron-held, SA casualties =
    # disposed; conservation closes the face (donor :11477-11509)
    try:
        av = us = dis = 0
        for rp in ('249', '250', '251'):
            z9 = sanitize(str(dump.get(rp, '')))
            if z9 in ('vilya slot', 'nenya slot', 'narya slot'):
                av += 1
            elif z9.startswith(('used ring', 'dark ring')):
                us += 1
            elif z9 in ('sa casualties', 'sa dead', 'dead'):
                dis += 1
        efp = int(getattr(st, 'fp_elven_rings', 3))
        esh = int(getattr(st, 'sh_elven_rings', 0))
        if av != efp:
            issues.append('elven rings fp available: engine=%d '
                          'client=%d' % (efp, av))
        if us != esh:
            issues.append('elven rings sh held: engine=%d client=%d'
                          % (esh, us))
        if av + us + dis != 3:
            issues.append('elven rings conservation: %d avail + %d '
                          'held + %d disposed != 3' % (av, us, dis))
    except Exception as e:
        issues.append('elven-ring check unavailable: %r' % (e,))
    return issues


def card_census(state, dump, hand_cids=None):
    """Engine hands vs the client's physical hand zones (SIZE) and,
    when the caller supplies commitment-decoded identities, vs those
    cids (IDENTITY).  Returns issue strings.

    Client fact #10 binds this witness: live card IDENTITY comes ONLY
    from commitment decode.  The dump gives zone membership (how many
    card pieces sit in Controls.45/46), never names; identity rows
    appear only when hand_cids={Side: [cid, ...]} is passed by a caller
    that decoded them itself.  Donor: _card_census (:12952-13015) +
    the hand-zone membership from _card_census_detail (:13017-13042).

    DEVIATION (documented): the donor's window CLASSIFIER
    (_hand_window: chain-open play / turn-start draw tagging) is driver
    flow state and is not lifted; rows are always emitted and the
    driver decides persistence."""
    if state is None or not dump:
        return []
    from board import sanitize
    from game_state import Side
    issues = []
    hand_zone = {Side.FREE_PEOPLES: 'fp hand', Side.SHADOW: 'sa hand'}
    tag = {Side.FREE_PEOPLES: 'FP', Side.SHADOW: 'SH'}
    for s in (Side.FREE_PEOPLES, Side.SHADOW):
        in_hand = sorted(p for p, z in dump.items()
                         if sanitize(str(z)) == hand_zone[s])
        eng_n = len(state.hands[s].hand)
        if eng_n != len(in_hand):
            issues.append('hand size %s: engine=%d client=%d (pids %s)'
                          % (tag[s], eng_n, len(in_hand),
                             ' '.join(in_hand) or '(none)'))
        if hand_cids is not None and s in hand_cids:
            # WAVE-J (wjg5 false alarm): SIDE-SCOPED identity - a side
            # absent from hand_cids is size-only.  Under blinding the
            # HUMAN side's client identities are structurally unknown
            # (engine holds placeholders); comparing them produced
            # 'engine-only=[HU_FR_*]' rows that escalated a NON-desync.
            eng = sorted(c for c in state.hands[s].hand
                         if isinstance(c, str))
            cli = sorted(hand_cids.get(s, []))
            if eng != cli:
                only_e = [c for c in eng if c not in cli]
                only_c = [c for c in cli if c not in eng]
                issues.append('hand %s: engine-only=%s client-only=%s'
                              % (tag[s], only_e, only_c))
    return issues
