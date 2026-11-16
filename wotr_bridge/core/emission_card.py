"""Card-effect row renderer: the bounded exception to "render the action's own parameters".

The action of a card choice carries no board effect (the card's handler computes it), so its
effect is rendered from the two ENGINE states around the apply: board_delta(pre, post), for
exactly ONE card choice.  Every row must be reproduced with real pieces - same nation and
figure type, a named figure by identity; a piece is never invented to make a row balance.

Row rules, per (nation, kind) unit type; Leaders as Nazgul / per-nation Leader rows; named
figures in their own identity rows:
  RECRUIT    +n with no -n of the type: one pool drag each (emitter.plan_muster; the Free
             Peoples pool is the reinforcements only, the Shadow's includes its dead pile).
  MOVE       removals and arrivals of one type that balance, casualty ledger unchanged:
             fixed transfers from any sources to any destinations (field or stronghold box
             as the arrival owns it).
  CASUALTY   -n covered by the engine's own casualty ledger: kill drags; a +n of the type
             then recruits (the downgrade).  A bare Shadow -n is the kill drag (its dead
             pile and reserve are one place); a Free Peoples removal the ledger does not
             account for is ambiguous (dead are permanent) and HOLDS.
  NAMED      leaves one region and joins one (or comes from off the board / the Fellowship):
             one drag of its piece.  From the Free Peoples dead pile: HOLDS.  A name that
             only leaves goes to the dead pile only when the state after proves it out of play.
  POLITICAL  +k per nation walks its marker k steps; a step back is one plain move.
  FELLOWSHIP Corruption: one drag of its marker; hidden: the flip of the figure (none when a
             reveal off the Track leaves its placement due: the placement flips); a reveal
             placement forced inside the choice takes the hidden / progress / region rows as
             the placement's own lines; a Guide change shows no line; a Companion leaving is
             its NAMED row; off the Mordor Track a new region is one move of the figure and a
             new progress one move of its marker; anything else HOLDS.
  DICE       new dice enter the pool; every other die by where it lies and what it shows
             (_die_ops).  A die that left the game, or a used die that came back, HOLDS.
  TABLE      the card the choice itself takes off the table goes to the player's Current
             Card slot; one the rules discard is killed where it lies, as a consequence.  A
             card that came TO the table and is not the pending or chained card HOLDS.
  EXPECTED   consumed silently: the pending card leaving its hand and arriving on the table
             or the discards (the play showed it), and AI hand gains the client itself drew.
             Any other hand / ring row HOLDS.
OWN MOVES (own_move, OWN_MOVES): what a choice moves by hand before any roll or draw of it
exists - the plan of a person's answer while the state after is not known.

What no rule consumes is ONE hold naming every such row.  Pieces are picked on a SCRATCH copy
of the client model (retracked row by row, so two rows never take one piece); the real view
is retracked only when the whole render succeeds.
"""
import collections

from . import emitter


# ---------------------------------------------------------------------
# the rows: what a card's effect changed between two engine states
# ---------------------------------------------------------------------
def _proj_board(st):
    """The comparable projection of an engine state: armies by region and type with their
    named Leaders, the Fellowship (with its declared region), politics, the table cards by
    identity, the dice faces, the rings and the hands (deck lengths stay out: a draw is shown
    by its own piece)."""
    from game_state import Side
    from board import sanitize
    import game_logic
    low = lambda x: str(getattr(x, 'value', x)).lower()
    armies, named = {}, {}
    for key, army in st.armies.items():
        reg = {}
        for un in army.units:
            t = '%s:%s' % (low(getattr(un, 'nation', '?')), low(getattr(un, 'kind', getattr(un, 'unit_type', '?'))))
            reg[t] = reg.get(t, 0) + 1
        if reg or army.leaders:
            armies[sanitize(key)] = (tuple(sorted(reg.items())), len(army.leaders))
        nm = tuple(sorted(str(getattr(ld, 'name', '')) for ld in army.leaders
                          if getattr(ld, 'name', None) and type(ld).__name__ in ('Companion', 'Minion')))
        if nm:
            named[sanitize(key)] = nm
    fs = st.fellowship
    try:
        region = str(game_logic.fellowship_region(st))
    except Exception:
        region = ''
    table, pol = getattr(st, 'persistent_cards', None), getattr(st, 'political', None)
    dice, hands = {}, {}
    for sd in (Side.FREE_PEOPLES, Side.SHADOW):
        ds = st.dice.get(sd, [])
        dice[sd.name] = (tuple(sorted(str(getattr(x.current, 'name', None)) for x in ds if not x.used)),
                         sum(1 for x in ds if x.used))
        hands[sd.name] = tuple(sorted(str(c) for c in st.hands[sd].hand if isinstance(c, str)))
    return {'armies': armies, 'named_leaders': named, 'dice': dice, 'hands': hands,
            'political': None if pol is None else tuple(sorted(
                (low(n), int(b)) for n, b in getattr(pol, 'boxes', {}).items())),
            'fellowship': (
                int(getattr(fs, 'progress', 0)), int(getattr(fs, 'corruption', 0)), bool(getattr(fs, 'hidden', True)),
                bool(getattr(fs, 'mordor_entered', False)), int(getattr(fs, 'mordor_step', 0)),
                str(getattr(getattr(fs, 'guide', None), 'name', '')),
                tuple(sorted(str(c.name) for c in getattr(fs, 'companions', []))), region),
            'rings': (int(getattr(st, 'fp_elven_rings', 3)), int(getattr(st, 'sh_elven_rings', 0))),
            'table': (
                tuple(sorted(str(getattr(c, 'card_id', c)) for c in getattr(table, 'fp_cards', []) or [])),
                tuple(sorted(str(getattr(c, 'card_id', c)) for c in getattr(table, 'sh_cards', []) or [])),
                tuple(sorted((str(k), tuple(str(x) for x in v))
                             for k, v in (getattr(st, 'discard_face_up', None) or {}).items())))}


def board_delta(pre, post):
    """{region: (before, after)} for the army rows that changed, the political steps per nation
    as INCREMENTS (a view that lags one tick cannot refuse a legal action), and the Fellowship,
    named Leader, dice, ring, hand and table differences by identity."""
    a0, a1 = _proj_board(pre), _proj_board(post)
    out = {}
    for k in set(a0['armies']) | set(a1['armies']):
        b, a = a0['armies'].get(k), a1['armies'].get(k)
        if b == a:
            continue
        bu = dict(b[0]) if b else {}
        au = dict(a[0]) if a else {}
        ud = tuple(sorted((t, au.get(t, 0) - bu.get(t, 0)) for t in set(bu) | set(au)
                          if au.get(t, 0) != bu.get(t, 0)))
        ld = (a[1] if a else 0) - (b[1] if b else 0)
        if ud or ld:
            out[k] = (ud, ld)
    p0, p1 = dict(a0['political'] or ()), dict(a1['political'] or ())
    pd = tuple(sorted((n, p1.get(n, 0) - p0.get(n, 0)) for n in set(p0) | set(p1) if p1.get(n, 0) != p0.get(n, 0)))
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
    n0, n1 = a0.get('named_leaders') or {}, a1.get('named_leaders') or {}
    nd = tuple(sorted((k, n0.get(k, ()), n1.get(k, ())) for k in set(n0) | set(n1) if n0.get(k) != n1.get(k)))
    if nd:
        out['__named__'] = nd
    d0, d1 = a0.get('dice') or {}, a1.get('dice') or {}
    dd = tuple(sorted((k, d0.get(k), d1.get(k)) for k in set(d0) | set(d1) if d0.get(k) != d1.get(k)))
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
            hd[k] = (tuple(sorted(s0 - s1)), tuple(sorted(s1 - s0)))
    if hd:
        out['__hands__'] = tuple(sorted(hd.items()))
    t0, t1 = a0.get('table'), a1.get('table')
    if t0 is not None and t1 is not None and t0 != t1:
        out['__table__'] = (t0, t1)
    return out


def balanced_transfers(minus, plus):
    """The physical transfers of one fungible figure type between regions, in a fixed order -
    or None when the counts do not balance.  Asked only after the engine has committed the
    decision and ruled out casualties of this type: it picks pieces, not game moves."""
    from board import sanitize
    delta = collections.Counter()
    for rows, sign in ((minus, -1), (plus, 1)):
        for region, count in rows:
            if int(count) != count or count < 0:
                return None
            delta[sanitize(str(region))] += sign * int(count)
    if sum(delta.values()):
        return None
    destinations = iter(sorted((region, n) for region, n in delta.items() if n > 0))
    target, remaining = next(destinations, (None, 0))
    transfers = []
    for source, count in sorted((region, -n) for region, n in delta.items() if n < 0):
        while count:
            take = min(count, remaining)
            transfers.append((source, target, take))
            count -= take
            remaining -= take
            if not remaining:
                target, remaining = next(destinations, (None, 0))
    return transfers


def physical_move_ops(view, nick, pids, destination):
    """One move op per zone the chosen pieces really stand in (the client's command names it)."""
    groups = {}
    for pid in pids:
        groups.setdefault(emitter._piece_zone_code(view, pid), []).append(pid)
    return [('move', {'nick': nick, 'pieces': pieces, 'frm': source, 'to': destination})
            for source, pieces in groups.items()]


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
            elif tn in ('Companion', 'Minion', 'ShadowMinion'):
                # FLA-3 (par5 g3#6): the engine's minion class is
                # ShadowMinion - the old 'Minion' check matched
                # nothing, so minions leaked into the plain-leader
                # buckets (nation '') and their identity rows vanished
                continue
            else:
                nv = _nv(getattr(ld, 'nation', ''))
                plain[nv] = plain.get(nv, 0) + 1
    return naz, plain


def _named_pid_side(name):
    """(client pid, canonical side name) for a named figure, or
    (None, None) when the tables do not know it.  FLA-6: reads the
    parser tables DIRECTLY - _companion_pid resolves minions too now
    (the seat's echo performer needs it), so its success no longer
    implies FREE_PEOPLES."""
    import log_replay as lr
    want = str(name).strip().lower()
    for p, n in lr.COMPANIONS.items():
        if str(n).strip().lower() == want:
            return str(p), 'FREE_PEOPLES'
    for p, faces in getattr(lr, 'PIECE_NAME_ALIASES', {}).items():
        for f in faces:
            if str(f).strip().lower() == want:
                return str(p), 'FREE_PEOPLES'
    for pid, nm in lr.MINION_PIECES.items():
        if str(nm).strip().lower() == want:
            return str(pid), 'SHADOW'
    return None, None


def _named_identity(name):
    """A promotion changes a piece's face, not its physical identity."""
    import log_replay
    pid, _side = _named_pid_side(name)
    return str(log_replay.COMPANIONS.get(str(pid), name))


def _named_elimination_destination(post, pid, side):
    """Use the completed rule state, retaining live/deferred named figures."""
    if post is None or pid is None:
        return None
    if side == 'FREE_PEOPLES':
        from .emission import _companion_out_of_play
        return emitter.FP_DEAD if _companion_out_of_play(post, pid) else None
    if side == 'SHADOW':
        import log_replay
        names = set(log_replay.names_for_pid(pid) or ())
        living = {str(getattr(figure, 'name', ''))
                  for army in post.armies.values() for figure in army.leaders}
        if names and not names.intersection(living):
            return emitter.SH_DEAD
    return None


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


def _die_ops(view, nick, side, index, before, after, post):
    """One action die across a card choice, by where it lies and what it shows: turned in the
    pool, put into the Hunt Box (the Shadow's as an Eye; the die of a Fellowship move the choice
    let go on), or used up.  A die that was used before stays where its own action put it."""
    import fellowship_action_timing
    if before.used:
        if after.used and after.current == before.current:
            return []
        raise LookupError('a used die came back')
    waiting = fellowship_action_timing.pending_die(post)    # the Hunt Box takes it when the move settles
    takes = emitter.hunt_takes_fellowship_die(post) if after is waiting else False
    hunt = any(after is d for d in post.hunt_box) or takes
    if takes is None or (not after.used and after.current == before.current):      # (not known yet where it goes)
        return []
    pid = str(emitter.die_piece_for(view, side.name, index))
    letter = emitter._FACE_LETTER.get(str(getattr(after.current, 'name', after.current)))
    if letter is None:
        raise LookupError('a die shows no face (%r)' % (after.current,))
    ops = [('die_change', {'nick': nick, 'die': pid, 'face': letter})] if after.current != before.current else []
    if hunt:
        return ops + [emitter.plan_eye_bank(view, nick, pid)]
    return ops + ([('kill', {'nick': nick, 'pieces': [pid], 'to': emitter.DICE_USED})] if after.used else [])


def chained_card(action):
    """The card id a card option PLAYS from the hand as part of the card's own
    effect (The Ents Awake chained into another card), else None."""
    choices = (getattr(action, 'params', None) or {}).get('choices')
    cid = choices.get('ents_chain_play') if isinstance(choices, dict) else None
    return str(cid) if cid else None


def _table_piece(view, cid):
    """(piece, Board slot) of a card that lies on the table, else None."""
    pid = (getattr(view, 'card_pid_map', None) or {}).get(str(cid))
    slot = emitter._piece_zone_code(view, pid, default=None) if pid else None
    return (str(pid), slot) if slot in emitter._TABLE_SLOTS else None


# the choices that take a card off the table as their own move: (choice key, its value - True:
# any true value -, the field of the choice that names the card)
_USES_TABLE_CARD = (
    ('mithril_redraw', True, 'card_id'), ('wizardstaff_prevent', True, 'card_id'),
    ('crebain_discard', True, 'card_id'), ('balrog_declare_draw', True, 'card_id'),
    ('use_hunt_reduction', True, 'card_id'), ('wwsat_source', 'table', 'card_id'),
    ('action', 'discard_table_card', 'target_card'))


def used_table_card(action):
    """The card id a card option takes off the table as its own move (the card it uses up or
    names: a row of _USES_TABLE_CARD), else None.  Every OTHER card that leaves the table
    during the choice is discarded by the rules: a consequence."""
    params = getattr(action, 'params', None) or {}
    choices = params.get('choices') if isinstance(params.get('choices'), dict) else {}
    for key, value, field in _USES_TABLE_CARD:
        if (choices.get(key) == value) if value is not True else choices.get(key):
            return str(choices.get(field) or params.get(field) or '') or None
    return None


def _used_card_op(view, nick, side_name, action):
    """The move of the table card a choice takes off the table (used_table_card): out of its
    Board slot into the player's Current Card slot, from where it is discarded; else None."""
    from .emission import _CARD_PLAY_ZONE
    seat = _table_piece(view, used_table_card(action))
    if seat is None or side_name not in _CARD_PLAY_ZONE:
        return None
    return ('move', {'nick': nick, 'pieces': [seat[0]], 'frm': seat[1], 'to': _CARD_PLAY_ZONE[side_name]})


# One card choice as its own-move rule reads it: the view and nick to plan with, the chooser,
# the state before, the pending card, and `said` - the choice in the engine's own words.
_Choice = collections.namedtuple('_Choice', 'view nick side_name pre pending said')


def _client(line):
    return ('directive', {'line': line})


def _own_gandalf_draw(c):
    return [_client(emitter.draw_call(c.side_name, c.pending['deck']))] if c.said.get('gandalf_draw') else []


def _own_palantir_draw(c):
    return [_client(emitter.draw_call(c.side_name, c.said['palantir_deck']))]


def _own_random_discard(c):
    return [_client('!CALL wwsat')] if c.said.get('wwsat_source') == 'hand' else []


def _own_another_way(c):
    """Corruption one back; then the Fellowship one step on (no die: the card pays), or hidden."""
    had = int(c.pre.fellowship.corruption)
    ops = [('move', {'nick': c.nick, 'pieces': ['252'], 'frm': 'Game.%d' % (2014 + had),
                     'to': 'Game.%d' % (2013 + had)})] if had > 0 else []
    if c.said.get('action') == 'move':
        ops += emitter.plan('fellowship_move', {}, c.view, c.nick, c.side_name)
    if c.said.get('action') == 'hide':
        ops += emitter.flip_to(c.view, c.nick, '247', 0)
    return ops


def _own_stewards(c):
    """The Gondor unit of the kind he names, in Boromir's region."""
    if not c.said.get('unit_type'):
        return []
    return [emitter.plan_muster(c.view, c.nick, c.side_name, {
        'nation': 'gondor', 'region': c.said['boromir_region']}, c.said['unit_type'])]


def _own_dead_men(c):
    """The Companions he names walk into the target region (the card's later choices - where
    the beaten army goes, how many are recruited - name no walk)."""
    if not c.said.get('companions') or not c.said.get('target_region'):
        return []
    home = next((k for k, army in c.pre.armies.items() if any(
        str(leader.name) == c.said['companions'][0] for leader in army.leaders
        if type(leader).__name__ == 'Companion')), None)
    if home is None:
        raise LookupError('%s stands in no region' % (c.said['companions'][0],))
    return emitter.plan('companion_move', {'from': home, 'to': c.said['target_region'],
                                           'companions': c.said['companions']}, c.view, c.nick, c.side_name)


def _own_grey_company(c):
    """A Regular of the nation he names to the casualties, an Elite of it in its place."""
    from board import sanitize
    region = sanitize(str(c.said['strider_region']))
    gone = emitter.pick_units(c.view, region, 'fp', n_regular=1, nation=c.said['nation'], side_hint='fp')
    return [('hand_exit_b', {'nick': c.nick, 'pieces': gone, 'frm': emitter._piece_zone_code(c.view, gone[0])}),
            emitter.plan_muster(c.view, c.nick, c.side_name, {'nation': c.said['nation'], 'region': region},
                                'elite')]


# THE table of what a card's choice moves by itself - by hand, BEFORE any roll or draw of the
# choice exists: the pending card (its printed name, or the engine's name of a card step) ->
# rule(_Choice) -> [(op, fields)].  A card that is not here moves nothing of its own before its
# result: its choice is read from the board only once the engine could apply it (the rows).
OWN_MOVES = {
    'Gandalf guide draw': _own_gandalf_draw,
    'Palantir draw': _own_palantir_draw,
    '__wwsat_discard__': _own_random_discard,
    'There is Another Way': _own_another_way,
    'House of the Stewards': _own_stewards,
    'Dead Men of Dunharrow': _own_dead_men,
    'The Grey Company': _own_grey_company,
}


def own_move(view, nick, side_name, action, pre):
    """What a card option moves by itself, known WITHOUT the state after it
    (the client's dice, tile or random discard may be needed before the engine
    can apply it): an Emission, or None when the option has no such move.
      * a chained card play: the card is played like any card ('!CALL
        play_card', hand -> the player's Current Card slot);
      * the table card the choice takes off the table (_used_card_op);
      * what the card's own row of OWN_MOVES names.
    A piece the board cannot supply for it is a hold that names the card."""
    from .emission import Emission, card_announce
    params = getattr(action, 'params', None) or {}
    said = params.get('choices') if isinstance(params.get('choices'), dict) else {}
    pending = getattr(pre, 'pending_card_option', None) or {}
    out, ops = Emission(), []
    chained = chained_card(action)
    try:
        if chained is not None:
            if chained not in (getattr(view, 'card_pid_map', None) or {}):
                raise LookupError('no client piece for the chained card %s' % chained[:60])
            ops.append(_client('!CALL play_card %s' % view.card_pid_map[chained]))
            out.chat.append(card_announce(chained))
        ops += [op for op in (_used_card_op(view, nick, side_name, action),) if op is not None]
        if pending.get('card_name') in OWN_MOVES:
            ops += OWN_MOVES[pending['card_name']](_Choice(view, nick, side_name, pre, pending, said))
    except LookupError as e:
        return Emission(hold='card choice %r: %s' % (str(pending.get('card_name'))[:40], e))
    for op, f in ops:
        if op == 'directive':
            out.calls.append(f['line'])
        else:
            out.lines.append(emitter.render(op, f, local=False))
    return out if out.lines or out.calls else None


def _wwsat_client_random_discard(pre, post, action, skey, cid):
    """R3-WWSAT-HUMAN-SHADOW-DISCARD (live-server FU3): the Worn with Sorrow
    and Toil 'hand' source (1.211f) removes ONE Free Peoples Character card
    by the client's own native random discard ('!CALL wwsat' for the AI
    Shadow, the human Shadow's own menu action), whose identity the bridge
    fed to game_logic.pick_random_card from the native receipt
    (random_discard_identity).  The card already left the physical hand and
    went face-down (1.261), so this hand row needs no renderer: holding on
    it degraded the commit to a misleading 'complete any required piece
    changes' prompt.  Any other lost card still holds."""
    from game_state import Side
    pco = getattr(pre, 'pending_card_option', None) or {}
    choices = (getattr(action, 'params', None) or {}).get('choices') or {}
    if (pco.get('card_name') != '__wwsat_discard__'
            or choices.get('wwsat_source') != 'hand'
            or str(skey) != Side.FREE_PEOPLES.name):
        return False
    before = list(pre.hands[Side.FREE_PEOPLES].discarded_face_down)
    after = list(post.hands[Side.FREE_PEOPLES].discarded_face_down)
    return after.count(cid) == before.count(cid) + 1


# ---------------------------------------------------------------------
# the renderer
# ---------------------------------------------------------------------
def _render_card_effect_rows(view, nick, side, kind, action, pre, post,
                             fp_seat_ai=False):
    """Render the board effect of ONE applied card_option/card_followup
    decision from the engine's pre/post states.  See the module
    docstring for the row-matching rules and the rule-6 rationale.
    fp_seat_ai: R7-REVEAL-FLIP-RESIDUALS (a), the live caller's seat fact
    for the off-Track reveal-flip rule (emission._placement_owns_flip).

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
    chained = chained_card(action)      # played by the chain: emission.render_decision adds the play

    delta = board_delta(pre, post)
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
            if chained is not None and str(cid) == chained:  # L1104-W09
                continue       # L1104-W09: played below (!CALL play_card)
            if _wwsat_client_random_discard(pre, post, action, skey, cid):
                continue       # the client's own native random discard
            holds.append('hand row: %s lost %r (not the pending card; '
                         'no discard renderer here)' % (skey, cid))
        pid_map = getattr(view, 'card_pid_map', None) or {}
        for cid in gained:
            if str(cid) in pid_map:
                continue       # the client itself drew it (!CALL)
            if str(cid).startswith('HU_'):
                # SQF-9 (seqack_fp1): a HUMAN-side placeholder gain IS
                # the client's own blinded draw - the engine can mint
                # HU_ identities only from client randomness (the draw
                # funnel starves the apply otherwise), so the physical
                # card already exists client-side; nothing to render.
                continue
            holds.append('hand row: %s gained %r with no client-drawn '
                         'pid -- cannot have been a funnel draw'
                         % (skey, cid))
    table = delta.pop('__table__', None)
    if table is not None:
        cids = _table_cids_changed(table[0], table[1])
        # A card that left the table.  The one the choice itself takes off
        # (used_table_card) goes through the player's Current Card slot.
        # Every other one the RULES discarded during the choice (a
        # Companion's card when he leaves the Fellowship, Wormtongue when
        # Rohan wakes): a consequence, killed where it lies - the AI's
        # render shows it, a person may leave it to the server.
        gone = (set(table[0][0]) | set(table[0][1])) - (set(table[1][0]) | set(table[1][1]))
        for c in sorted(gone):
            seat = _table_piece(view, c)
            if c == used_table_card(action):
                tail_ops.extend(op for op in (_used_card_op(view, nick, emitter.canon_side(side), action),)
                                if op is not None)
            elif seat is not None:
                tail_ops.append(('kill', {'nick': nick, 'pieces': [seat[0]], 'to': seat[1], '_may': True}))
        stray = sorted(c for c in cids if c not in gone and str(c) not in (expected_cid, chained))
        if stray:
            holds.append('table row: cards other than the pending one '
                         'changed: %r' % (stray,))
    if '__rings__' in delta:
        holds.append('rings row: %r has no card-effect grammar'
                     % (delta.pop('__rings__'),))

    # ---- dice: new dice enter the pool; every other die by where it lies
    # and what it shows (_die_ops) ----------------------------------------
    if '__dice__' in delta:
        dd = delta.pop('__dice__')
        try:
            extra_lines += _pool_growth_lines(view, nick, pre, post)
            for s2 in (_S.FREE_PEOPLES, _S.SHADOW):
                d0, d1 = pre.dice.get(s2, []), post.dice.get(s2, [])
                if len(d1) < len(d0):
                    raise LookupError('a die left the game')
                for i, (a0, a1) in enumerate(zip(d0, d1)):
                    tail_ops.extend(_die_ops(view, nick, s2, i, a0, a1, post))
        except LookupError as e:
            holds.append('dice row: %s (%r)' % (e, dd))

    # ---- fellowship ---------------------------------------------------
    fell = dict(delta.pop('__fellowship__', ()) or ())
    fell_left = {_named_identity(name) for name in (fell.pop('left', ()) or ())}
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
    # R7-REVEAL-FLIP-RESIDUALS (b): a reveal placement the ENGINE resolved
    # inside this choice (a Hunt-reduction card that took the damage to zero
    # at Morannon / Minas Morgul, at progress 0 or with one legal region; a
    # Shadow card's reveal there) consumes the 'hidden' / 'progress' /
    # 'region' rows with the placement's choreography: 248 reset -> walk ->
    # ONE 247 flip (engine pre/post only; emission._forced_reveal_placement)
    from .emission import _forced_reveal_placement, _reveal_placement_follows
    forced = _forced_reveal_placement(pre, post, side, fp_seat_ai)
    if forced is not None:
        fell.pop('hidden', None)
        fell.pop('region', None)
        if forced['reset']:
            fell.pop('progress', None)
        try:
            tail_ops.extend(emitter.with_forced_reveal_placement(
                [], scratch, nick, emitter.canon_side(side), at, forced))
        except LookupError as e:
            holds.append('fellowship row: forced reveal placement: %s' % e)
    if 'hidden' in fell:
        fell.pop('hidden')
        # R6-OFFTRACK-DOUBLE-FLIP: an FP commit that revealed off the Mordor
        # Track with the reveal placement still due (a Hunt-reduction card
        # that took the damage to zero) is consumed WITHOUT a flip - the
        # FP's fellowship_reveal_place renders the one flip (engine pre/post
        # only; see emission._reveal_placement_follows).
        # R7-REVEAL-FLIP-RESIDUALS (a): an AI vs AI Shadow commit too
        # (fp_seat_ai: the AI-FP's placement renders the one flip).
        if not _reveal_placement_follows(pre, post, side, fp_seat_ai):
            tail_ops.extend(emitter.flip_to(
                view, nick, '247', 0 if post.fellowship.hidden else 1))
    if not getattr(post.fellowship, 'mordor_entered', False):
        # off the Mordor Track the figure and the progress marker follow the
        # engine: the figure stands in the region the choice put it in (Cruel
        # Weather), the marker on the count the choice left (a Fellowship
        # move the card makes, or lets go on)
        if 'region' in fell:
            import game_logic
            fell.pop('region')
            to = emitter.zone_code_for(sanitize(str(game_logic.fellowship_region(post))))
            frm = emitter._piece_zone_code(view, '247', default=to)
            if frm != to:
                tail_ops.append(('move', {'nick': nick, 'pieces': ['247'], 'frm': frm, 'to': to}))
        if 'progress' in fell:
            fell.pop('progress')
            was, now = (max(0, min(12, int(s.fellowship.progress))) for s in (pre, post))
            step = dict(nick=nick, frm='Game.%d' % (2014 + was), to='Game.%d' % (2014 + now))
            # (one step on is the client's own advance; anything else a plain drag of the marker)
            tail_ops.append(('track_marker', dict(step, piece='248')) if now == was + 1
                            else ('move', dict(step, pieces=['248'])))
    # 'joined'/anything else: no grammar here
    for k in sorted(fell):
        holds.append('fellowship row: %r=%r has no card-effect grammar'
                     % (k, fell[k]))

    # ---- named figures (identity rows) --------------------------------
    named = delta.pop('__named__', ()) or ()
    named_delta = {}                     # region -> named count delta
    leaves, joins = {}, {}
    for region, before, after in named:
        b = {_named_identity(name) for name in before}
        a2 = {_named_identity(name) for name in after}
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
            # R18-SIEGE C20: the arrival zone reads the POST state too, as
            # _emit_move does (R17-SIEGE-LIFT-SWEEP D5): a named Minion in a
            # Corsairs relief landing fights in the field beside the landed
            # army (1.281), not inside the walls with the frozen garrison
            op = ('move', {'nick': nick, 'pieces': [pid], 'frm': frm,
                           'to': emitter.garrison_arrival_zone(
                               scratch, sanitize(str(js[0])), pside,
                               after=post)})
            move_ops.append(op)
            _retrack_scratch(scratch, [op])
        elif ls and not js:
            pid, pside = _named_pid_side(nm)
            destination = _named_elimination_destination(post, pid, pside)
            if destination is None:
                holds.append('named row: %r left %r without a proved elimination '
                             'or rendered live destination' % (nm, ls))
                continue
            try:
                # Do not turn a missing physical identity into an initial-
                # setup source. The figure may occupy a native siege seat.
                frm = (emitter._piece_zone_code(scratch, pid)
                       if pid in scratch.piece_region else None)
            except LookupError:
                frm = None
            if frm is None or frm in (emitter.FP_DEAD, emitter.SH_DEAD,
                                       emitter.FP_POOL, emitter.SH_POOL):
                holds.append('named row: no live client source for eliminated %r' % (nm,))
                continue
            op = ('move', {'nick': nick, 'pieces': [pid], 'frm': frm,
                           'to': destination})
            kill_ops.append(op)
            _retrack_scratch(scratch, [op])
        else:
            holds.append('named row: %r ambiguous (leaves=%r joins=%r)'
                         % (nm, ls, js))
    for nm in sorted(fell_left):
        pid, pside = _named_pid_side(nm)
        if pid is None:
            holds.append('fellowship row: %r left the fellowship '
                         'with no named board arrival (unknown '
                         'figure)' % (nm,))
            continue
        destination = _named_elimination_destination(post, pid, pside)
        if destination is None:
            # Lure can remove a Hobbit from the Fellowship while its living
            # figure awaits a separate placement decision. That later action
            # owns the move; this choice must not kill or pre-place it.
            import log_replay
            names = set(log_replay.names_for_pid(pid) or ())
            pending = getattr(post, 'pending_hunt', None) or {}
            deferred = {str(getattr(c, 'name', c)) for c in
                        (getattr(post, '_hobbit_place_queue', None) or ())}
            if pending.get('phase') == 'hobbit_place':
                deferred.add(str(pending.get('companion', '')))
                deferred.add(str(getattr(pending.get('companion_obj'), 'name', '')))
            if names.intersection(deferred):
                continue
            holds.append('fellowship row: %r left without a proved elimination '
                         'or deferred placement' % nm)
            continue
        try:
            frm = (emitter._piece_zone_code(scratch, pid)
                   if pid in scratch.piece_region else None)
        except LookupError:
            frm = None
        if frm is None or frm in (emitter.FP_DEAD, emitter.SH_DEAD,
                                   emitter.FP_POOL, emitter.SH_POOL):
            holds.append('fellowship row: no live client source for eliminated %r' % nm)
            continue
        op = ('move', {'nick': nick, 'pieces': [pid],
                       'frm': frm, 'to': destination})
        move_ops.append(op)
        _retrack_scratch(scratch, [op])

    # ---- army rows: units + refined leader rows -----------------------
    cas_inc = _cas_increments(pre, post)
    replacement_seats = {}
    source_start = len(getattr(pre, '_dg_src_log', ()) or ())
    replacement_sources = [(_nv(nation), source) for nation, source in
        list(getattr(post, '_dg_src_log', ()) or ())[source_start:]
        if source is not None]
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
            for _move_op, fields in physical_move_ops(scratch, nick, pids, None):
                op = ('hand_exit_b', {'nick': nick, 'pieces': fields['pieces'],
                                      'frm': fields['frm']})
                if unit_kind == 'elite':
                    replacement_seats.setdefault((sanitize(str(region)), nation), []).extend(
                        (pid, fields['frm']) for pid in fields['pieces'])
                kill_ops.append(op)
                _retrack_scratch(scratch, [op])
        except LookupError as e:
            holds.append('%s row: cannot supply %d casualt%s at %s: %s'
                         % (tname, n, 'y' if n == 1 else 'ies',
                            region, e))

    def _emit_recruits(region, n, side_name9, nation, unit_kind,
                       tname):
        params = {'nation': nation, 'region': sanitize(str(region))}
        for _ in range(n):
            try:
                seats = replacement_seats.get((params['region'], nation), [])
                source_index = next((i for i, (nat, source) in enumerate(replacement_sources)
                    if unit_kind == 'regular' and seats and nat == nation), None)
                if source_index is None:
                    op = emitter.plan_muster(scratch, nick, side_name9,
                                             params, unit_kind)
                else:
                    _, source = replacement_sources.pop(source_index)
                    side_tag = emitter._side_of_nation(nation)
                    if source not in ('cas', 'pool'):
                        raise LookupError('invalid recorded downgrade source %r' % source)
                    zone = (('fp dead' if side_tag == 'fp' else 'sa dead')
                            if source == 'cas' else
                            ('fp reinf' if side_tag == 'fp' else 'sa reinf'))
                    pieces = emitter.pick_units(scratch, zone, side_tag,
                        nation=nation, n_regular=1, side_hint=side_tag)
                    op = ('move', {'nick': nick, 'pieces': pieces,
                        'frm': emitter._piece_zone_code(scratch, pieces[0]),
                        'to': emitter.garrison_arrival_zone(scratch, params['region'], side_tag)})
                if unit_kind == 'regular' and seats:
                    op[1]['to'] = seats.pop(0)[1]       # the reduced Elite's own seat
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
            # R17-SIEGE-LIFT-SWEEP D5: the arrival zone also reads the POST
            # state, so a Corsairs relief force lands in the field (1.281)
            transfers = physical_move_ops(scratch, nick, pids,
                emitter.garrison_arrival_zone(scratch, sanitize(str(to)),
                                              side_name9, after=post))
            move_ops.extend(transfers)
            _retrack_scratch(scratch, transfers)
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
        elif cas == 0 and balanced_transfers(M, P) is not None:
            for source, target, count in balanced_transfers(M, P):
                _emit_move(source, target, count, side_tag, side_name9,
                           nation, unit_kind, tkey)
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
        elif balanced_transfers(M, P) is not None:
            for source, target, count in balanced_transfers(M, P):
                _emit_move(source, target, count, 'shadow', 'SHADOW',
                           None, 'nazgul', 'nazgul')
        else:
            holds.append('nazgul row: AMBIGUOUS pairing (minus=%r '
                         'plus=%r) -- refusing to guess' % (M, P))

    # ---- political rows (the planner emission.py already dispatches)
    for nat, inc in (delta.pop('__political__', ()) or ()):
        box0 = next((int(v) for k, v in pre.political.boxes.items()
                     if _nv(k) == str(nat)), None)
        if box0 is None:
            holds.append('political row: unknown nation %r' % (nat,))
            continue
        if inc < 0:
            # a step back (Stormcrow): the walk button only goes forward, so
            # it is one plain move of the nation's marker to its box
            try:
                marker = emitter.plan_advance_political(scratch, nick, emitter.canon_side(side), {
                    'nation': nat, 'from': box0 + inc})[1]['piece']
                tail_ops.append(('move', {'nick': nick, 'pieces': [marker], 'frm': 'Game.%d' % (2000 + box0),
                                          'to': 'Game.%d' % (2000 + box0 + inc)}))
            except LookupError as e:
                holds.append('political row: %s %+d: %s' % (nat, inc, e))
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

_STAGED_ARMY_CARDS = frozenset((
    'SHADOW_Strategy_The_Shadow_Lengthens',
    'SHADOW_Strategy_The_Shadow_is_Moving',
))


def staged_army_card(state):
    pending = getattr(state, 'pending_card_option', None) or {}
    cid = pending.get('card_id')
    return cid if cid in _STAGED_ARMY_CARDS else None


def _staged_cargo_plan(view, nick, pre, witness):
    """Reserve every original physical group before rendering any arrival."""
    import ai
    import game_state as gs
    from board import sanitize
    projected = ai.clone_state(pre)
    scratch = _Scratch(view, pre)
    reserved, plans = set(), []

    def army_at(state, region):
        key = next((k for k in state.armies if sanitize(str(k)) == region), None)
        return key, state.armies.get(key)

    for source, target, units, leaders in witness[1]:
        source, target = sanitize(source), sanitize(target)
        key, army = army_at(projected, source)
        if army is None or source == target:
            raise LookupError('staged movement has no distinct original source: %s' % source)
        chosen_units, chosen_leaders, pids = [], [], []
        for nation, kind, count in units:
            if type(count) is not int or count <= 0:
                raise LookupError('invalid staged unit count')
            matches = [u for u in army.units if u.nation.name == nation and u.kind.name == kind]
            if len(matches) < count:
                raise LookupError('staged unit cargo exceeds original source at %s' % source)
            chosen_units.extend(matches[:count])
            kw = {'n_regular': 0, 'n_elite': 0}
            if kind not in ('REGULAR', 'ELITE'):
                raise LookupError('unsupported staged unit kind %s' % kind)
            kw['n_' + kind.lower()] = count
            pids.extend(emitter.pick_units(scratch, source, 'shadow',
                        nation=gs.Nation[nation].value, side_hint='shadow', **kw))
        for cls, nation, name, count in leaders:
            if type(count) is not int or count <= 0:
                raise LookupError('invalid staged leader count')
            matches = [l for l in army.leaders if
                       (type(l).__name__, getattr(getattr(l, 'nation', None), 'name', ''),
                        str(getattr(l, 'name', ''))) == (cls, nation, name)]
            if len(matches) < count:
                raise LookupError('staged leader cargo exceeds original source at %s' % source)
            chosen_leaders.extend(matches[:count])
            if name:
                pid, side = _named_pid_side(name)
                if count != 1 or pid is None or side != 'SHADOW':
                    raise LookupError('unknown staged named figure %s' % name)
                pid = str(pid)
                if sanitize(str(scratch.piece_region.get(pid, ''))) != source:
                    raise LookupError('staged named figure is not at original source: %s' % name)
                pids.append(pid)
            else:
                pids.extend(emitter.pick_units(scratch, source, 'shadow',
                            n_nazgul=count, side_hint='shadow'))
        pids = [str(pid) for pid in pids]
        if not pids or len(set(pids)) != len(pids) or reserved.intersection(pids):
            raise LookupError('staged movement reuses or lacks an original physical piece')
        reserved.update(pids)
        plans.append((key, source, target, chosen_units, chosen_leaders, pids))

    # This projection subtracts only already-rendered cargo from the residual
    # effect renderer. It never adopts the physical client board into gameplay.
    for key, source, target, units, leaders, pids in plans:
        army = projected.armies[key]
        for unit in units:
            army.units.remove(unit)
        for leader in leaders:
            army.leaders.remove(leader)
    for key, source, target, units, leaders, pids in plans:
        dest_key, army = army_at(projected, target)
        if army is None:
            army = gs.Army(region=projected.board.region(target))
            projected.armies[target] = army
        army.units.extend(units)
        army.leaders.extend(leaders)
    # R17-SIEGE-LIFT-SWEEP D6: each piece leaves its REAL client zone; a piece
    # outside the zone the engine implies is a loud zone_model_mismatch (noted
    # on the live view, which the driver files), never a silent regroup
    ops = []
    for key, source, target, units, leaders, pids in plans:
        ops.extend(emitter._by_real_zone(
            view, ('move', {'nick': nick, 'pieces': pids,
                            'frm': emitter.garrison_arrival_zone(scratch, source, 'shadow'),
                            'to': emitter.garrison_arrival_zone(scratch, target, 'shadow')}),
            'staged card move from %s' % source))
    return projected, ops




def render_card_effect(view, nick, side, kind, action, pre, post,
                       fp_seat_ai=False):
    # fp_seat_ai: R7-REVEAL-FLIP-RESIDUALS (a), the live caller's seat fact
    # (seats.seat_is_ai) for the off-Track reveal-flip rule
    from .emission import Emission, _ops_to_emission, _retrack_ops
    pre = pre if pre is not None else view.state
    cid = staged_army_card(pre)
    if cid is None:
        return _render_card_effect_rows(view, nick, side, kind, action, pre, post,
                                        fp_seat_ai=fp_seat_ai)
    if post is None:
        return Emission(hold='staged movement needs its committed post state')
    witness = getattr(post, '_card_army_move_witness', None)
    if witness is None or witness is getattr(pre, '_card_army_move_witness', None):
        delta = board_delta(pre, post)
        if staged_army_card(post) == cid and set(delta).issubset({'__hands__', '__table__'}):
            return _render_card_effect_rows(view, nick, side, kind, action, pre, post,
                                            fp_seat_ai=fp_seat_ai)
        return Emission(hold='staged movement completed without original-cargo provenance')
    if not isinstance(witness, tuple) or len(witness) != 2 or witness[0] != cid:
        return Emission(hold='staged movement provenance belongs to a different card')
    try:
        projected, ops = _staged_cargo_plan(view, nick, pre, witness)
        scratch = _Scratch(view, projected)
        _retrack_ops(scratch, ops)
        tail = _render_card_effect_rows(scratch, nick, side, kind, action, projected, post,
                                        fp_seat_ai=fp_seat_ai)
        if tail.hold:
            return tail
        # Validate the entire render on scratch before changing the live view.
        result = _ops_to_emission(view, nick, action.action_type, ops)
        _retrack_ops(view, tail.ops)
        result.ops.extend(tail.ops)
        result.lines.extend(tail.lines)
        result.calls.extend(tail.calls)
        result.chat.extend(tail.chat)
        return result
    except (LookupError, ValueError, KeyError, TypeError) as exc:
        return Emission(hold='staged movement: %s' % exc)
