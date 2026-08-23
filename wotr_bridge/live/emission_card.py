"""Card-effect row renderer: the SANCTIONED, BOUNDED exception to
design rule 6 (REBUILD_PLAN.md), decided by the coordinator 2026-08-19.

WHY THIS DIFF-SCOPED RENDERER IS ALLOWED WHILE _emit_board_diff WAS NOT
(provenance for the future reader):

  The retired `_emit_board_diff` (ai_move_server.py :8940-9759) diffed
  WHOLE parser-state snapshots after the fact, then PAIRED pluses to
  minuses with distance/adjacency heuristics whenever the counts did
  not line up -- and those guessed pairings bred the phantom-piece
  defect family (pieces invented, resurrected, or teleported to make a
  diff balance).  Rule 6 therefore bans emission-by-diff.

  Optioned event cards are the one action class where the engine's
  ACTION carries no board effect at all (the chosen option's effect is
  computed inside the card handler), so rule 6's normal form -- render
  the action's own params -- has nothing to render.  The bounded
  exception:

    * scope: EXACTLY ONE pending card_option decision, between the two
      engine states the DRIVER itself held around the apply.  Never a
      running reconciliation, never a parser state.
    * rows: client_board.board_delta(pre, post) -- both sides of the
      diff are the ENGINE's own states (I1), the same projection the
      witnesses use.
    * pairing: a row renders ONLY when it matches one LITERAL rule
      below (single unambiguous counterpart).  There are NO pairing
      heuristics, no distance scoring, no cross-type matching.  Every
      ambiguity class that fed the old differ's guesses -- multiple
      sources for one destination, split counts, a removal that could
      be either a casualty or a return -- is a LOUD HOLD naming the
      rows.  Holds are acceptable; phantom pairings are the disease
      this replaces.

ROW-MATCHING RULES (per (nation, kind) unit type; leaders refined from
the engine armies into nazgul / per-nation plain-leader rows; named
figures ride the __named__ identity rows):

  RECRUIT   +n with NO matching -n anywhere for the type: each unit is
            one pool drag via emitter.plan_muster (FP pool = FP Reinf
            ONLY -- FP dead are permanent; SA pool includes the SA dead
            pile -- SA casualties recycle, rulebook p.30; a physical
            shortage raises and becomes a HOLD, pieces are never
            invented).
  MOVE      exactly ONE (-n) region and exactly ONE (+n) region with
            EQUAL counts and an unmoved casualty ledger: one group drag,
            arrival zone via the garrison question.
  CASUALTY  (-n) rows fully covered by the engine's OWN fp_casualties
            ledger increment: kill drags (hand_exit_b) out of the
            region; any (+n) rows of the type then recruit from the
            pool (the downgrade shape).  For SHADOW units a bare (-n)
            with no (+n) is rendered as the kill drag: the SA dead pile
            RECYCLES into the reinforcement pool (rulebook p.30, and
            client_board's census counts 'sa dead' with 'sa reinf'), so
            'casualty' and 'return' converge physically -- this is the
            one removal that is NOT ambiguous.  An FP removal the
            ledger does not fully account for stays ambiguous
            (casualty-or-return differ materially: FP dead are
            permanent) and HOLDS.
  NAMED     a name leaving exactly one region and joining exactly one
            (or joining from off-board / the fellowship): one identity
            drag of its pid from where the client holds it.  A named
            FP figure sourced from the FP dead pile is a resurrection
            and HOLDS.  A name that only LEAVES holds (casualty or
            return -- cannot tell).
  POLITICAL per-nation +k increments walk the nation's track marker k
            steps via emitter.plan_advance_political (the political
            planner emission.py already dispatches); a backward step
            has no grammar and HOLDS.
  FELLOWSHIP corruption delta = one drag of marker 252 on the shared
            track (the hunt_take_corruption grammar); hidden flip =
            flip of pawn 247; a guide change is IMPLICIT in the client
            (the parser auto-reassigns) and is consumed without lines;
            companions leaving the fellowship are consumed by their
            __named__ join row; everything else fellowship HOLDS.
  DICE      pure pool GROWTH renders via emission._pool_growth_lines
            (bonus dice from card musters); any other dice delta HOLDS.
  EXPECTED  bookkeeping rows of the play itself are consumed silently:
            the pending card's own cid leaving the owner's hand and its
            arrival on the table / face-up discard (the '!CALL
            play_card' client handler performed that physical move at
            play time), and AI-hand GAINS whose cids are in
            view.card_pid_map (the client itself drew them via the
            '!CALL draw_*' randomness funnel).  Any other hand / table
            / ring row HOLDS.

Anything the rules above do not consume -> ONE loud HOLD naming every
unconsumed row.  Piece picks run on a SCRATCH copy of the client model
(with incremental retrack so two rows never pick the same pid); the
REAL view is retracked only when the whole render succeeds, so a HOLD
leaves the client model untouched.

ASCII only.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BRIDGE = os.path.abspath(os.path.join(_HERE, '..'))
_SRC = os.path.abspath(os.path.join(_HERE, '..', '..', 'src'))
for _p in (_SRC, _BRIDGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import emitter

from . import client_board


# ---------------------------------------------------------------------
# scratch client model: planners pick on a COPY so a HOLD never leaves
# provisional retracks behind on the real view
# ---------------------------------------------------------------------
class _Scratch(object):
    def __init__(self, view, pre):
        self.state = pre                       # the PRE engine state
        self.piece_region = dict(view.piece_region)
        # same object on purpose: A17 swap-awareness must stay current
        self.piece_types = getattr(view, 'piece_types', None)
        self.die_id_map = dict(getattr(view, 'die_id_map', {}) or {})
        self.ai_drawn_pids = set(getattr(view, 'ai_drawn_pids', ())
                                 or ())
        self.fellowship_location = getattr(view, 'fellowship_location',
                                           None)


def _retrack_scratch(scratch, ops):
    """The same provisional client-model update emission._retrack_ops
    performs, applied to the scratch so the NEXT pick in this render
    cannot re-select a piece an earlier row already moved (the
    muster-double-pick lesson, donor _retrack :10005-10011)."""
    from .emission import _retrack_ops
    _retrack_ops(scratch, ops)


# ---------------------------------------------------------------------
# small readers (engine states only)
# ---------------------------------------------------------------------
def _nv(x):
    return str(getattr(x, 'value', x) or '').lower()


def _cas_increments(pre, post):
    """{'nation:kind': +n} increments of the engine's OWN FP-casualty
    ledger across the apply (combat._track_fp_casualties keys)."""
    out = {}
    c0 = getattr(pre, 'fp_casualties', None) or {}
    c1 = getattr(post, 'fp_casualties', None) or {}
    for nat in set(c0) | set(c1):
        for kind in ('regular', 'elite', 'leader'):
            d = ((c1.get(nat, {}) or {}).get(kind, 0) or 0) \
                - ((c0.get(nat, {}) or {}).get(kind, 0) or 0)
            if d:
                out['%s:%s' % (_nv(nat), kind)] = \
                    out.get('%s:%s' % (_nv(nat), kind), 0) + d
    return out


def _leader_comp(state, region_san):
    """(nazgul_count, {nation_value: plain_leader_count}) at a region,
    EXCLUDING named figures (Companion/Minion ride __named__)."""
    from board import sanitize
    naz, plain = 0, {}
    for k, a in state.armies.items():
        if sanitize(str(k)) != region_san:
            continue
        for ld in getattr(a, 'leaders', []) or []:
            tn = type(ld).__name__
            if tn == 'Nazgul':
                naz += 1
            elif tn in ('Companion', 'Minion'):
                continue
            else:
                nv = _nv(getattr(ld, 'nation', ''))
                plain[nv] = plain.get(nv, 0) + 1
    return naz, plain


def _named_pid_side(name):
    """(client pid, canonical side name) for a named figure, or
    (None, None) when the tables do not know it."""
    import log_replay as lr
    try:
        return emitter._companion_pid(name), 'FREE_PEOPLES'
    except LookupError:
        pass
    want = str(name).strip().lower()
    for pid, nm in lr.MINION_PIECES.items():
        if str(nm).strip().lower() == want:
            return str(pid), 'SHADOW'
    return None, None


def _table_cids_changed(t0, t1):
    """Every cid that differs between two summarize 'table' terms
    (fp_cards, sh_cards, discard_face_up items)."""
    changed = set()
    for a, b in ((t0[0], t1[0]), (t0[1], t1[1])):
        changed |= set(a) ^ set(b)
    d0 = {k: set(v) for k, v in (t0[2] if len(t0) > 2 else ())}
    d1 = {k: set(v) for k, v in (t1[2] if len(t1) > 2 else ())}
    for k in set(d0) | set(d1):
        changed |= d0.get(k, set()) ^ d1.get(k, set())
    return changed


# ---------------------------------------------------------------------
# the renderer
# ---------------------------------------------------------------------
def render_card_effect(view, nick, side, kind, action, pre, post):
    """Render the board effect of ONE applied card_option/card_followup
    decision from the engine's pre/post states.  See the module
    docstring for the row-matching rules and the rule-6 rationale.

    Returns an emission.Emission; HOLD (loud, naming the rows) on any
    ambiguity or physical shortage -- never a guessed pairing."""
    from .emission import Emission, _ops_to_emission, _pool_growth_lines
    from board import sanitize

    at = getattr(action, 'action_type', 'card_option')
    if pre is None:
        pre = view.state
    if post is None:
        return Emission(hold=(
            '%s (kind %r): the card-effect row renderer needs BOTH '
            'states around the apply and post is missing -- refusing '
            'to guess the board effect' % (at, kind)))

    pco = getattr(pre, 'pending_card_option', None) \
        or getattr(pre, 'pending_card_followup', None) or {}
    cname = str(pco.get('card_name', '') or
                (action.params or {}).get('card_name', '') or '?')
    expected_cid = str(pco.get('card_id', '') or
                       (action.params or {}).get('card_id', ''))

    delta = client_board.board_delta(pre, post)
    if not delta:
        return Emission()      # a choice with no board effect

    holds = []
    scratch = _Scratch(view, pre)
    kill_ops, move_ops, recruit_ops = [], [], []
    tail_ops, extra_lines = [], []

    # ---- expected play bookkeeping (consumed, never rendered) --------
    from game_state import Side as _S
    hands = dict(delta.pop('__hands__', ()) or ())
    for skey, (lost, gained) in sorted(hands.items()):
        for cid in lost:
            if str(cid) == expected_cid:
                continue       # the play_card client handler did this
            holds.append('hand row: %s lost %r (not the pending card; '
                         'no discard renderer here)' % (skey, cid))
        pid_map = getattr(view, 'card_pid_map', None) or {}
        for cid in gained:
            if str(cid) in pid_map:
                continue       # the client itself drew it (!CALL)
            holds.append('hand row: %s gained %r with no client-drawn '
                         'pid -- cannot have been a funnel draw'
                         % (skey, cid))
    table = delta.pop('__table__', None)
    if table is not None:
        cids = _table_cids_changed(table[0], table[1])
        stray = sorted(c for c in cids if str(c) != expected_cid)
        if stray:
            holds.append('table row: cards other than the pending one '
                         'changed: %r' % (stray,))
    if '__rings__' in delta:
        holds.append('rings row: %r has no card-effect grammar'
                     % (delta.pop('__rings__'),))

    # ---- dice: pure pool growth only ---------------------------------
    if '__dice__' in delta:
        dd = delta.pop('__dice__')
        grew = any(len(post.dice.get(s, [])) > len(pre.dice.get(s, []))
                   for s in (_S.FREE_PEOPLES, _S.SHADOW))
        shrank = any(len(post.dice.get(s, []))
                     < len(pre.dice.get(s, []))
                     for s in (_S.FREE_PEOPLES, _S.SHADOW))
        if grew and not shrank:
            try:
                extra_lines += _pool_growth_lines(view, nick, pre, post)
            except LookupError as e:
                holds.append('dice row: pool growth unrenderable: %s'
                             % e)
        else:
            # WAVE-K K6 (wkf1 LIVE: Mirror of Galadriel wedged the
            # first AI-as-FP game): a pure FACE CHANGE of unused dice
            # (same roster length, same used flags, only .current
            # differs) renders as the corpus die-change line per die
            # ('$nick changes <pid>  to <letter>' - emitter op
            # 'die_change', the elven-ring planner's grammar); the
            # Shadow's Mouth-of-Sauron class rides the same lane.  Any
            # other dice delta (usage flips, shrink) still HOLDS.
            changed = []
            pure = not grew and not shrank
            if pure:
                for s2 in (_S.FREE_PEOPLES, _S.SHADOW):
                    d0 = pre.dice.get(s2, [])
                    d1 = post.dice.get(s2, [])
                    for i, (a0, a1) in enumerate(zip(d0, d1)):
                        if bool(a0.used) != bool(a1.used):
                            pure = False
                            break
                        if a0.current != a1.current:
                            if a0.used or a1.current is None:
                                pure = False
                                break
                            changed.append((s2, i, a1.current))
                    if not pure:
                        break
            if pure and changed:
                for s2, i, face in changed:
                    fname = str(getattr(face, 'name', face))
                    letter = emitter._FACE_LETTER.get(fname)
                    try:
                        dp = emitter.die_piece_for(view, s2.name, i)
                    except LookupError as e:
                        holds.append('dice row: face change die %s/%d '
                                     'has no client piece: %s'
                                     % (s2.name, i, e))
                        continue
                    if letter is None:
                        holds.append('dice row: no die-change letter '
                                     'for face %r' % (fname,))
                        continue
                    tail_ops.append(('die_change', {
                        'nick': nick, 'die': str(dp), 'face': letter}))
            else:
                holds.append('dice row: %r is not pure pool growth '
                             'nor a pure face change (no card-effect '
                             'grammar for usage changes)' % (dd,))

    # ---- fellowship ---------------------------------------------------
    fell = dict(delta.pop('__fellowship__', ()) or ())
    fell_left = set(fell.pop('left', ()) or ())
    fell.pop('guide', None)          # implicit: parser auto-reassigns
    if 'corruption' in fell:
        fell.pop('corruption')
        c0 = max(0, min(12, int(pre.fellowship.corruption)))
        c1 = max(0, min(12, int(post.fellowship.corruption)))
        if c0 != c1:
            tail_ops.append(('move', {
                'nick': nick, 'pieces': ['252'],
                'frm': 'Game.%d' % (2014 + c0),
                'to': 'Game.%d' % (2014 + c1)}))
    if 'hidden' in fell:
        fell.pop('hidden')
        tail_ops.append(('flip', {'nick': nick, 'pieces': ['247']}))
    # 'joined'/'progress'/'region'/anything else: no grammar here
    for k in sorted(fell):
        holds.append('fellowship row: %r=%r has no card-effect grammar'
                     % (k, fell[k]))

    # ---- named figures (identity rows) --------------------------------
    named = delta.pop('__named__', ()) or ()
    named_delta = {}                     # region -> named count delta
    leaves, joins = {}, {}
    for region, before, after in named:
        b, a2 = set(before), set(after)
        named_delta[region] = named_delta.get(region, 0) \
            + len(after) - len(before)
        for nm in b - a2:
            leaves.setdefault(nm, []).append(region)
        for nm in a2 - b:
            joins.setdefault(nm, []).append(region)
    for nm in sorted(set(leaves) | set(joins)):
        ls, js = leaves.get(nm, []), joins.get(nm, [])
        if len(js) == 1 and len(ls) <= 1:
            pid, pside = _named_pid_side(nm)
            if pid is None:
                holds.append('named row: unknown figure %r' % (nm,))
                continue
            frm = None
            if ls:
                try:
                    frm = emitter._piece_zone_code(scratch, pid)
                except LookupError:
                    frm = None
            else:
                frm = emitter._piece_zone_code(scratch, pid,
                                               default=None)
                if frm is None and nm in fell_left:
                    frm = emitter.FELLOWSHIP_BOX
                if frm is None and pside == 'SHADOW':
                    frm = emitter.SH_POOL
            if frm is None:
                holds.append('named row: no client zone for %r (pid '
                             '%s)' % (nm, pid))
                continue
            if frm == emitter.FP_DEAD:
                holds.append('named row: %r would rise from the FP '
                             'dead pile (FP dead are PERMANENT)'
                             % (nm,))
                continue
            if nm in fell_left:
                fell_left.discard(nm)
            op = ('move', {'nick': nick, 'pieces': [pid], 'frm': frm,
                           'to': emitter._garrison_dest(
                               scratch, sanitize(str(js[0])), pside)})
            move_ops.append(op)
            _retrack_scratch(scratch, [op])
        elif ls and not js:
            holds.append('named row: %r left %r with no destination '
                         '(casualty or return -- cannot tell)'
                         % (nm, ls))
        else:
            holds.append('named row: %r ambiguous (leaves=%r joins=%r)'
                         % (nm, ls, js))
    for nm in sorted(fell_left):
        holds.append('fellowship row: %r left the fellowship with no '
                     'named board arrival' % (nm,))

    # ---- army rows: units + refined leader rows -----------------------
    cas_inc = _cas_increments(pre, post)
    typed = {}     # 'nation:kind' -> {'plus': [(r,n)], 'minus': [(r,n)]}
    naz_rows = {'plus': [], 'minus': []}
    for region in sorted(k for k in delta if not str(k).startswith('__')):
        ud, ld = delta.pop(region)
        for tkey, n in ud:
            slot = typed.setdefault(str(tkey), {'plus': [], 'minus': []})
            if n > 0:
                slot['plus'].append((region, int(n)))
            else:
                slot['minus'].append((region, -int(n)))
        if ld:
            naz0, pl0 = _leader_comp(pre, region)
            naz1, pl1 = _leader_comp(post, region)
            naz_d = naz1 - naz0
            plain_d, explained = {}, naz_d + named_delta.get(region, 0)
            for nv in set(pl0) | set(pl1):
                d = pl1.get(nv, 0) - pl0.get(nv, 0)
                if d:
                    plain_d[nv] = d
                    explained += d
            if explained != ld:
                holds.append('leader row: %s count delta %+d not '
                             'explained by named%+d/nazgul%+d/plain%r'
                             % (region, ld, named_delta.get(region, 0),
                                naz_d, plain_d))
                continue
            if naz_d > 0:
                naz_rows['plus'].append((region, naz_d))
            elif naz_d < 0:
                naz_rows['minus'].append((region, -naz_d))
            for nv, d in sorted(plain_d.items()):
                slot = typed.setdefault('%s:leader' % nv,
                                        {'plus': [], 'minus': []})
                if d > 0:
                    slot['plus'].append((region, d))
                else:
                    slot['minus'].append((region, -d))

    def _pick(region_key, side_tag, nation, unit_kind, n):
        kw = {'n_regular': 0, 'n_elite': 0, 'n_leader': 0,
              'n_nazgul': 0}
        kw['n_' + unit_kind] = n
        return emitter.pick_units(scratch, region_key, side_tag,
                                  nation=nation, side_hint=side_tag,
                                  **kw)

    def _emit_kills(region, n, side_tag, nation, unit_kind, tname):
        try:
            pids = _pick(sanitize(str(region)), side_tag, nation,
                         unit_kind, n)
            op = ('hand_exit_b', {
                'nick': nick, 'pieces': pids,
                'frm': emitter.garrison_arrival_zone(
                    pre, sanitize(str(region)), side_tag)})
            kill_ops.append(op)
            _retrack_scratch(scratch, [op])
        except LookupError as e:
            holds.append('%s row: cannot supply %d casualt%s at %s: %s'
                         % (tname, n, 'y' if n == 1 else 'ies',
                            region, e))

    def _emit_recruits(region, n, side_name9, nation, unit_kind,
                       tname):
        params = {'nation': nation, 'region': sanitize(str(region))}
        if emitter.garrison_side_of(pre, params['region']) \
                == side_name9:
            params['_besieged_garrison'] = True
        for _ in range(n):
            try:
                op = emitter.plan_muster(scratch, nick, side_name9,
                                         params, unit_kind)
                recruit_ops.append(op)
                _retrack_scratch(scratch, [op])
            except LookupError as e:
                holds.append('%s row: pool cannot supply a %s at %s: '
                             '%s' % (tname, unit_kind, region, e))
                return

    def _emit_move(frm, to, n, side_tag, side_name9, nation, unit_kind,
                   tname):
        try:
            pids = _pick(sanitize(str(frm)), side_tag, nation,
                         unit_kind, n)
            op = ('move', {
                'nick': nick, 'pieces': pids,
                'frm': emitter.zone_code_for(sanitize(str(frm))),
                'to': emitter._garrison_dest(scratch,
                                             sanitize(str(to)),
                                             side_name9)})
            move_ops.append(op)
            _retrack_scratch(scratch, [op])
        except LookupError as e:
            holds.append('%s row: cannot supply the %d-piece move '
                         '%s -> %s: %s' % (tname, n, frm, to, e))

    for tkey in sorted(typed):
        P, M = typed[tkey]['plus'], typed[tkey]['minus']
        nation, unit_kind = tkey.rsplit(':', 1)
        side_tag = emitter._side_of_nation(nation)
        side_name9 = emitter.canon_side(side_tag)
        cas = int(cas_inc.get(tkey, 0))
        tm = sum(n for _, n in M)
        if not M:
            for r, n in P:
                _emit_recruits(r, n, side_name9, nation, unit_kind,
                               tkey)
        elif side_tag == 'fp' and cas == tm:
            # the engine's OWN ledger says every removed piece is a
            # casualty; any pluses are then pool recruits (downgrades)
            for r, n in M:
                _emit_kills(r, n, side_tag, nation, unit_kind, tkey)
            for r, n in P:
                _emit_recruits(r, n, side_name9, nation, unit_kind,
                               tkey)
        elif not P:
            if side_tag == 'fp':
                holds.append(
                    '%s row: %r removed with the FP casualty ledger '
                    'moving %+d of %d -- casualty or return, cannot '
                    'tell (FP dead are permanent)' % (tkey, M, cas, tm))
            else:
                # SHADOW: the dead pile recycles into the pool, so
                # casualty and return converge on one physical gesture
                for r, n in M:
                    _emit_kills(r, n, side_tag, nation, unit_kind,
                                tkey)
        elif len(M) == 1 and len(P) == 1 and M[0][1] == P[0][1] \
                and cas == 0:
            _emit_move(M[0][0], P[0][0], M[0][1], side_tag, side_name9,
                       nation, unit_kind, tkey)
        elif cas == 0 and sum(n for _, n in M) == sum(n
                                                      for _, n in P):
            # WAVE-J (wjg8 LIVE wedge: Rage of the Dunlendings moved
            # Isengard regulars North+South Dunland -> Moria and this
            # branch held 'AMBIGUOUS' forever): a COUNT-CONSERVING
            # pairing of same-(nation,kind) unit rows with NO casualty
            # increment has NOTHING to guess - the units are fungible,
            # so every assignment of sources to destinations yields
            # the IDENTICAL client board (pick_units already owns pid
            # fungibility).  Deterministic greedy pour, sorted row
            # order.
            pi = [[r, n] for r, n in P]
            for r, n in M:
                left = n
                while left > 0 and pi:
                    pr, pn = pi[0]
                    take = min(left, pn)
                    _emit_move(r, pr, take, side_tag, side_name9,
                               nation, unit_kind, tkey)
                    left -= take
                    if pn - take == 0:
                        pi.pop(0)
                    else:
                        pi[0][1] = pn - take
        else:
            holds.append('%s row: AMBIGUOUS pairing (minus=%r plus=%r '
                         'fp_cas%+d) -- refusing to guess'
                         % (tkey, M, P, cas))

    # nazgul rows (fungible shadow leaders)
    P, M = naz_rows['plus'], naz_rows['minus']
    if M or P:
        if not M:
            for r, n in P:
                _emit_recruits(r, n, 'SHADOW', None, 'leader',
                               'nazgul')
        elif not P:
            for r, n in M:
                _emit_kills(r, n, 'shadow', None, 'nazgul', 'nazgul')
        elif len(M) == 1 and len(P) == 1 and M[0][1] == P[0][1]:
            _emit_move(M[0][0], P[0][0], M[0][1], 'shadow', 'SHADOW',
                       None, 'nazgul', 'nazgul')
        else:
            holds.append('nazgul row: AMBIGUOUS pairing (minus=%r '
                         'plus=%r) -- refusing to guess' % (M, P))

    # ---- political rows (the planner emission.py already dispatches)
    for nat, inc in (delta.pop('__political__', ()) or ()):
        if inc <= 0:
            holds.append('political row: %s %+d has no backward '
                         'grammar' % (nat, inc))
            continue
        box0 = next((int(v) for k, v in pre.political.boxes.items()
                     if _nv(k) == str(nat)), None)
        if box0 is None:
            holds.append('political row: unknown nation %r' % (nat,))
            continue
        for step in range(int(inc)):
            try:
                tail_ops.append(emitter.plan_advance_political(
                    scratch, nick, emitter.canon_side(side),
                    {'nation': nat, 'from': box0 + step}))
            except LookupError as e:
                holds.append('political row: %s step %d: %s'
                             % (nat, box0 + step, e))
                break

    # ---- leftovers = rows with no rule --------------------------------
    for k in sorted(delta, key=str):
        holds.append('unclassified row: %r = %r' % (k, delta[k]))

    if holds:
        return Emission(hold=(
            '%s %r (kind %r) unrenderable/ambiguous rows -- HOLDING, '
            'never guessing: %s' % (at, cname, kind, ' | '.join(holds))))

    ops = kill_ops + move_ops + recruit_ops + tail_ops
    em = _ops_to_emission(view, nick, at, ops)   # retracks the REAL view
    em.lines += extra_lines
    return em
