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
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BRIDGE = os.path.abspath(os.path.join(_HERE, '..'))
_SRC = os.path.abspath(os.path.join(_HERE, '..', '..', 'src'))
for _p in (_SRC, _BRIDGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Client fact #14: full dump ~428 pieces; below this it is torn.
MIN_DUMP_PIECES = 400

# CLIENT-SIDE zone names the DUMP reader keeps (BRCMP-2, 2026-08-29).
# The corpus parser's hand-tuned log_replay.LOCATION_CODES maps the
# client's Current Action slot (Game.2035) to 'Used' - right for the
# corpus kill grammar ('kills <die> in Current Action' = spent), WRONG
# for a board dump, where a die parked there is merely STAGED (the
# human's open window; the AI's die awaiting its chain-close kill).  A
# killed die shows in the side's Dead zone.  The dump reader consults
# this table FIRST so the client's own name survives (the authoritative
# messages.properties name).
LOCATION_CODES = {'Game.2035': 'Current Action'}

# Row classes (BRCMP design 2): the two ASYNC-written families - the
# action dice (pool / used / hunt / staged) and the deck counts - plus
# the hand SIZE are STAGING rows: published, never escalated, never a
# divergence.  Everything else is a BOARD row.  ONE classifier for the
# live lane (test_live_session_census keeps the offline twin).
_STAGING_ROW_RE = re.compile(r'^(dice |deck |hand size )')


def is_staging_row(row):
    return bool(_STAGING_ROW_RE.match(str(row)))


def classify_row(row):
    """'staging' | 'board' for one census row."""
    return 'staging' if is_staging_row(row) else 'board'


_UNIT_KINDS =('regular', 'elite', 'leader', 'nazgul', 'companion',
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
                    # design rule 8 FAIL-CLOSED: an un-blinded #HAND
                    # snapshot means the blinding flag is missing - no
                    # reader may consume this file (raise, never a
                    # silent None that reads as 'no dump yet')
                    raise RuntimeError(
                        'BLINDING VIOLATION: un-blinded #HAND '
                        'snapshot in the board dump (missing '
                        '-Dai.feed.aiside)')
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 3:
                    continue
                code = parts[1].strip('<~>')
                region = (LOCATION_CODES.get(code)
                          or lr.decode_location(code))
                rk = str(region) if region else parts[1]
                for pid in parts[2].split():
                    out[pid] = rk
    except RuntimeError:
        raise               # the blinding violation must stay LOUD
    except Exception:
        return None
    return out or None


def read_board_flags(path):
    """Read native fourth-column masks, aligned with piece IDs.

    Mask meanings depend on piece type. Political activation comes from
    the fifth-column H.k() state, not these masks. None when unreadable.
    """
    if not os.path.exists(path):
        return None
    out = {}
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 4:
                    continue
                pids = parts[2].split()
                flags = parts[3].split()
                for i, pid in enumerate(pids):
                    out[pid] = flags[i] if i < len(flags) else ''
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
# board projection / delta (increment semantics)
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
# the direct engine<->client witness
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


def fellowship_marker_hidden(marker_states):
    """Native bR.k(): 0 draws Fellowship.png, 1 Fellowship_revealed.png.

    This is the marker's public face, not the legacy selection/masking flag.
    Missing or unfamiliar exports cannot establish visibility.
    """
    value = (marker_states or {}).get('247')
    return value == 0 if type(value) is int and value in (0, 1) else None


def political_marker_active(pid, marker_states):
    """Native political face, or None when the dump cannot establish it."""
    if str(pid) in ('256', '261', '262', '263'):
        return True
    face = (marker_states or {}).get(str(pid))
    return bool(face) if face in (0, 1) else None


def diff_engine_vs_dump(state, dump, dynamic_types=None, ai_module=None,
                        initial_dump=None, granted_pids=None,
                        revoked_pids=None, client_turn=None, marker_states=None,
                        pid_to_cid=None):
    """DIRECT engine<->client verification (no parser in between):
    armies per region, leaders per region+side, nazgul counts + named
    minions, dice zones, deck sizes, siege sub-boxes, companions,
    fellowship tracks / pawn / political boxes and native activation faces,
    control markers, elven
    rings; optional per-nation FORCE POOLS (needs ai_module), PIECE
    CONSERVATION (needs initial_dump) and the turn counter (needs the
    caller's client_turn channel).

    Returns a list of issue strings; empty = the client board IS the
    engine state.  Never mutates anything (witnesses notice, the
    engine owns).  Donor: verify_client_vs_engine (:11042-11510),
    decoupled: dynamic_types replaces the replayer, ai_module replaces
    self.ai, initial_dump replaces self._client_initial, granted/
    revoked pid sets replace the die-registration driver fields.

    Rows name their client pids ('pids=<pid>@<zone> ...') so every
    surface can print WHO disagrees, not just how many (BRCMP-6)."""
    if not dump or state is None:
        return None
    from board import sanitize
    from game_state import Side
    import board as bd
    import emitter as em
    st = state
    regions = bd.regions
    issues = []
    from .shadow_siege_slots import GENERIC_SLOTS
    for pid, zone in dump.items():
        if sanitize(str(zone)).lower() in GENERIC_SLOTS:
            _, kind = unit_type(pid, dynamic_types)
            if kind in _UNIT_KINDS:
                issues.append('unbound generic stronghold: piece %s in %s requires its observed origin' % (pid, zone))
    cl_units, cl_leaders, cl_pool = {}, {}, {}
    cl_leader_nations, cl_leader_nation_pids = {}, {}
    cl_naz, cl_minion = {}, {}
    cl_dice, cl_decks = {}, {}
    cl_dice_pids, cl_unit_pids, cl_leader_pids = {}, {}, {}
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
                # engine dropped the die entirely
                continue
            if granted_pids is not None and int(pid) >= 300 \
                    and pid not in granted_pids:
                # a never-granted BONUS die parks in a Dead/Reinf box
                # from setup; counting it as 'used' misreads the box
                # (the granted set shields the count check as it
                # shields the roster check)
                continue
            s = 'fp' if pid in fp_dice else 'shadow'
            if z in ('fp dice', 'sa dice'):
                box = 'pool'
            elif 'hunt' in z:
                box = 'hunt'
            elif z == 'current action':
                # BRCMP-2: the client's FOURTH die state - STAGED (a
                # human window open, or the AI's die awaiting its
                # chain-close kill).  Compatible with engine pool OR
                # used; never a disagreement by itself.
                box = 'staged'
            elif z in ('used', 'fp dead', 'sa dead'):
                # the client's kill routes a spent die to the side's
                # Dead zone - same meaning as the Used box
                box = 'used'
            else:
                continue   # bonus die not yet in play (reinf box)
            cl_dice[(s, box)] = cl_dice.get((s, box), 0) + 1
            cl_dice_pids.setdefault((s, box), []).append(
                '%s@%s' % (pid, z))
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
                cl_unit_pids.setdefault(k, []).append(pid)
            elif ut in ('leader', 'nazgul', 'minion', 'companion'):
                s = ('shadow' if ut in ('nazgul', 'minion')
                     else 'fp' if ut == 'companion'
                     else em._side_of_nation(nat))
                cl_leaders[(base, s)] = cl_leaders.get((base, s), 0) + 1
                cl_leader_pids.setdefault((base, s), []).append(pid)
                if ut == 'leader':
                    nk = (base, natv)
                    cl_leader_nations[nk] = cl_leader_nations.get(nk, 0) + 1
                    cl_leader_nation_pids.setdefault(nk, []).append(pid)
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
    en_units, en_leaders, en_leader_nations = {}, {}, {}
    for key, army in st.armies.items():
        k = sanitize(str(key))
        for u in army.units:
            kk = (k, (_nv(u.nation), _nv(u.kind)))
            en_units[kk] = en_units.get(kk, 0) + 1
        for ld in army.leaders:
            s = _leader_side(ld)
            en_leaders[(k, s)] = en_leaders.get((k, s), 0) + 1
            if type(ld).__name__ == 'Leader':
                nk = (k, _nv(ld.nation))
                en_leader_nations[nk] = en_leader_nations.get(nk, 0) + 1
    for kk in sorted(set(en_units) | set(cl_units)):
        a, b = en_units.get(kk, 0), cl_units.get(kk, 0)
        if a != b:
            issues.append('units %s %s/%s: engine=%d client=%d pids=%s'
                          % (kk[0], kk[1][0], kk[1][1], a, b,
                             ' '.join(cl_unit_pids.get(kk, ())) or '-'))
    for kk in sorted(set(en_leaders) | set(cl_leaders)):
        a, b = en_leaders.get(kk, 0), cl_leaders.get(kk, 0)
        if a != b:
            issues.append('leaders %s (%s): engine=%d client=%d pids=%s'
                          % (kk[0], kk[1], a, b,
                             ' '.join(cl_leader_pids.get(kk, ())) or '-'))
    for nk in sorted(set(en_leader_nations) | set(cl_leader_nations)):
        a, b = en_leader_nations.get(nk, 0), cl_leader_nations.get(nk, 0)
        if a != b:
            issues.append('leader nation %s %s: engine=%d client=%d pids=%s'
                          % (nk[0], nk[1], a, b,
                             ' '.join(cl_leader_nation_pids.get(nk, ())) or '-'))
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
    # dice zones.  BRCMP-2: the client has FOUR die states - pool,
    # hunt, used (killed -> Dead zone) and STAGED (Current Action).  A
    # staged die is an unspent pool die (human window open) or a spent
    # die not yet killed (the AI's chain-close kill), so the engine's
    # pool/used counts may each exceed the client's by up to the
    # staged count.  Rows print pid@zone (BRCMP-6).
    for s, tag in ((Side.FREE_PEOPLES, 'fp'), (Side.SHADOW, 'shadow')):
        total = len(st.dice.get(s, []))
        hunt = sum(1 for d in st.hunt_box
                   if getattr(d, 'side', None) is s)
        used = sum(1 for d in st.dice.get(s, []) if d.used) - hunt
        pool = total - used - hunt
        c_pool = cl_dice.get((tag, 'pool'), 0)
        c_used = cl_dice.get((tag, 'used'), 0)
        c_hunt = cl_dice.get((tag, 'hunt'), 0)
        c_stg = cl_dice.get((tag, 'staged'), 0)
        # Players put the payment die in the Hunt Box before resolving the
        # Hunt. The rules engine queues that addition until damage, reveal
        # and Guide choices finish, so it cannot affect this Hunt's modifier.
        # This is a physical-zone projection only; never alter hunt_box.
        if s is Side.FREE_PEOPLES and c_hunt == hunt + 1:
            from fellowship_action_timing import pending_die
            if pending_die(st) is not None:
                hunt += 1
                used -= 1

        def _dpids(*boxes, _tag=tag):
            return ' '.join(p for b in boxes
                            for p in cl_dice_pids.get((_tag, b), ())) \
                or '-'
        if c_hunt != hunt:
            issues.append('dice %s hunt: engine=%d client=%d pids=%s'
                          % (tag, hunt, c_hunt, _dpids('hunt')))
        pool_ok = c_pool <= pool <= c_pool + c_stg
        used_ok = c_used <= used <= c_used + c_stg
        if not pool_ok:
            issues.append('dice %s pool: engine=%d client=%d (+%d '
                          'staged) pids=%s'
                          % (tag, pool, c_pool, c_stg,
                             _dpids('pool', 'staged')))
        if not used_ok:
            issues.append('dice %s used: engine=%d client=%d (+%d '
                          'staged) pids=%s'
                          % (tag, used, c_used, c_stg,
                             _dpids('used', 'staged')))
        if pool_ok and used_ok \
                and c_pool + c_used + c_stg != pool + used:
            issues.append('dice %s count: engine=%d client=%d '
                          '(pool+used+staged) pids=%s'
                          % (tag, pool + used, c_pool + c_used + c_stg,
                             _dpids('pool', 'used', 'staged')))
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
    # stronghold box, besieger OUTSIDE
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
    from .promotion_recovery import identity_rows
    issues.extend(identity_rows(st, dump, marker_states))
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
        # BRCMP-5: the client's PARKING slots count as dead/out-
        # compatible - the client's own random-companion draw ('draws
        # Boromir') parks the drawn piece in the drawer's Card Play
        # zone; the corpus parser's _COMPANION_DEATH_AREAS calls that
        # a death area (log_replay), the comparator must too
        deadish = {'fp casualties', 'sa casualties', 'dead',
                   'fp dead', 'sa dead',
                   'fp card play', 'sa card play', 'current action',
                   'current fp card', 'current sa card'}
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
    # FELLOWSHIP TRACKS / PAWN / POLITICAL BOXES (BRCMP-7): the fields
    # the adopters WRITE were never compared, so an adoption triggered
    # by an unrelated row rewrote them with no row before and none
    # after.  Rows here make them visible on every surface (and count
    # for the adoption oracle); the driver's owed-track / owed-pawn
    # holds keep an in-flight marker walk out of the census.
    try:
        import game_logic as _gl7
        import log_replay as _lr7
        fs7 = st.fellowship
        physical_hidden = fellowship_marker_hidden(marker_states)
        if physical_hidden is not None and bool(fs7.hidden) != physical_hidden:
            issues.append('fellowship visibility: engine=%s client=%s pids=247'
                          % ('hidden' if fs7.hidden else 'revealed',
                             'hidden' if physical_hidden else 'revealed'))
        in_mordor = bool(getattr(fs7, 'mordor_entered', False))
        z248 = str(dump.get('248', '') or '')
        m248 = re.match(r'^Track (\d+)$', z248)
        if m248 and not in_mordor:
            cp7 = int(m248.group(1))
            ep7 = int(getattr(fs7, 'progress', 0) or 0)
            if cp7 != ep7:
                issues.append('track progress: engine=%d client=%d '
                              'pids=248@%s' % (ep7, cp7, sanitize(z248)))
        z252 = str(dump.get('252', '') or '')
        m252 = re.match(r'^Track (\d+)$', z252)
        if m252:
            cc7 = int(m252.group(1))
            ec7 = int(getattr(fs7, 'corruption', 0) or 0)
            if cc7 != ec7:
                issues.append('track corruption: engine=%d client=%d '
                              'pids=252@%s' % (ec7, cc7, sanitize(z252)))
        z247 = str(dump.get('247', '') or '')
        zs247 = sanitize(z247)
        m247 = re.match(r'^Mordor (\d)', z247)
        if m247:
            cstep = int(m247.group(1))
            estep = getattr(fs7, 'mordor_step', None)
            if not in_mordor or estep != cstep:
                issues.append('pawn mordor step: engine=%s client=%d '
                              'pids=247@%s'
                              % (estep if in_mordor else 'not in mordor',
                                 cstep, zs247))
        elif z247 and _strip_sh(zs247) in regions and in_mordor:
            issues.append('pawn mordor exit: engine=Mordor client=%s pids=247@%s'
                          % (_strip_sh(zs247), zs247))
        elif z247 and _strip_sh(zs247) in regions and not in_mordor:
            ereg = sanitize(str(_gl7.fellowship_region(st)))
            if ereg != _strip_sh(zs247):
                issues.append('pawn region: engine=%s client=%s '
                              'pids=247@%s' % (ereg, _strip_sh(zs247),
                                               zs247))
        pol7 = getattr(st, 'political', None)
        if pol7 is not None:
            for pid7, nation7 in _lr7.POLITICAL_MARKERS.items():
                zp7 = str(dump.get(str(pid7), '') or '')
                mp7 = re.match(r'^Political (\d)$', zp7)
                if not mp7:
                    continue
                cbox7 = 3 - int(mp7.group(1))
                ebox7 = pol7.boxes.get(nation7)
                if ebox7 is not None and int(ebox7) != cbox7:
                    issues.append('political %s: engine=%d client=%d '
                                  'pids=%s@%s'
                                  % (_nv(nation7), int(ebox7), cbox7,
                                     pid7, sanitize(zp7)))
                active7 = (political_marker_active(pid7, marker_states)
                           if marker_states is not None else None)
                if active7 is not None and bool(pol7.active.get(nation7, False)) != active7:
                    issues.append('political activation %s: engine=%s client=%s '
                                  'pids=%s@%s'
                                  % (_nv(nation7), bool(pol7.active.get(nation7, False)),
                                     active7, pid7, sanitize(zp7)))
    except Exception as e:
        issues.append('track/pawn/political check unavailable: %r'
                      % (e,))
    # CONTROL MARKERS vs the engine's enemy_controlled sets.  Marker
    # supply is PHYSICAL (10 SH / 6 FP): controlling more regions than
    # markers is correct board state, so a shortfall of that size is
    # tolerated, never an issue row.
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
    # the CLIENT'S OWN TURN COUNTER (fact 15): after both sides recover,
    # client turn == engine turn.  BRCMP-8: the dump carries NO Game.7
    # row (the client does not export the turn marker), so the leg
    # reads the CALLER's client-turn channel (`client_turn`) and is
    # NOT-MEASURED (no row) without one - never a dead check posing as
    # a passing one.
    if client_turn is not None \
            and int(getattr(st, 'turn', 0) or 0) != int(client_turn):
        issues.append('turn counter: engine=%s client=%s'
                      % (getattr(st, 'turn', '?'), int(client_turn)))
    # ELVEN-RING BOXES: pids 249-251 in the Vilya/Nenya/Narya Slots =
    # available, Used Ring / Dark Ring = Sauron-held, SA casualties =
    # disposed; conservation closes the face
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
    from .table_recovery import rows as table_card_rows
    issues.extend(table_card_rows(state, dump, pid_to_cid))
    return issues


def captured_garrison_conflicts(state):
    """Captured settlements still occupied by their original garrison.

    Siege armies share one engine region entry. At a captured stronghold,
    the original owner's units may legally besiege the capturing side's
    garrison. Require the stronghold, siege flag and both unit sides before
    treating those units as besiegers; a stale flag alone proves nothing.
    """
    import game_state as gs
    from board import regions, sanitize
    from combat import side_of_nation
    armies = {sanitize(str(k)): army for k, army in
              getattr(state, 'armies', {}).items()}
    sieges = {sanitize(str(k)) for k in
              (getattr(state, 'besieged_strongholds', None) or ())}
    conflicts = []
    for captor in (gs.Side.SHADOW, gs.Side.FREE_PEOPLES):
        owner = (gs.Side.FREE_PEOPLES if captor is gs.Side.SHADOW
                 else gs.Side.SHADOW)
        captured = (getattr(state, 'enemy_controlled', None) or {}).get(captor)
        for key in list(captured or ()):
            region = sanitize(str(key))
            army = armies.get(region)
            if army is None or not army.units:
                continue
            sides = [side_of_nation(unit.nation) for unit in army.units]
            count = sides.count(owner)
            if not count:
                continue
            if (region in sieges and regions.get(region, {}).get('is_stronghold')
                    and captor in sides):
                continue
            conflicts.append((captor, owner, key, count))
    return conflicts


def rules_invariant_rows(state):
    """BL-21(4) (2026-08-25): agreed-wrong states the piece diff can
    NEVER see (engine == client on the same impossible board).  Rows
    ride the witness cadence into ai_census.jsonl, so the tester's
    DIVERGENCE=STOP rule fires on a rules-impossible state exactly
    like on a desync.  First invariant: a captured settlement whose
    original owner's units still stand (the Woodland Realm chimera)."""
    import game_state as gs
    rows = []
    for cap_side, owner, rkey, n in captured_garrison_conflicts(state):
        rows.append('INVARIANT captured-with-garrison: %d %s '
                    'unit(s) stand in %s-captured %s'
                    % (n, owner.name, cap_side.name, rkey))
    # BL-21(5) (2026-08-25): a broken figure conservation (deployed +
    # casualties > the physical components) is another agreed-wrong
    # state the piece diff never sees - a resync that resurrected a
    # dead FP garrison without rolling back fp_casualties.  It rides
    # the witness cadence so the tester STOPs on it live.
    try:
        import ai as _ai
        dep = _ai._count_deployed_all(state)
        fcas = getattr(state, 'fp_casualties', None) or {}
        for nat, maxes in _ai.NATION_MAX_FIGURES.items():
            fp = _ai._side_of_nation(nat) == gs.Side.FREE_PEOPLES
            d = dep.get(nat, {})
            c = fcas.get(nat, {}) if fp else {}
            for kind, mx in maxes.items():
                tot = d.get(kind, 0) + c.get(kind, 0)
                if tot > mx:
                    rows.append('INVARIANT conservation %s %s: %d '
                                'deployed + %d casualties > %d'
                                % (getattr(nat, 'name', nat), kind,
                                   d.get(kind, 0), c.get(kind, 0), mx))
    except Exception as _ce:
        rows.append('conservation check unavailable: %r' % (_ce,))
    return rows


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
            # SIDE-SCOPED identity: a side absent from hand_cids is
            # size-only (under blinding the HUMAN side's client
            # identities are structurally unknown - the engine holds
            # placeholders - and comparing them would escalate a
            # NON-desync)
            eng = sorted(c for c in state.hands[s].hand
                         if isinstance(c, str))
            cli = sorted(hand_cids.get(s, []))
            if eng != cli:
                only_e = [c for c in eng if c not in cli]
                only_c = [c for c in cli if c not in eng]
                issues.append('hand %s: engine-only=%s client-only=%s'
                              % (tag[s], only_e, only_c))
    return issues
