"""Emission: engine action -> client gesture lines for the thin driver.

Emission renders an engine-approved decision from its action parameters
or its committed before/after states. The latter covers card effects and
automatic consequences. Equivalent groupings and source pairings of
interchangeable pieces reproduce the same final board. Physical piece
availability, identity, casualty destination and field/stronghold ownership
must still agree; rendering never invents pieces or legal game actions.

WHAT IS LIFTED (donor: wotr_bridge/ai_move_server.py, read-only; all
line refs are into that file):

  * play_event staging order          :15874-15900  (the play's own
    gestures go AHEAD of the effect's randomness: die-spend line,
    'plays <name>' chat request, '!CALL play_card <pid>')
  * RIG CLASS-B refund contract       :15903-15956  (a play the engine
    refunded renders NOTHING; the predicate tests ALL THREE conditions
    -- card kept, die intact, no chain -- exposed as the state-taking
    helper `play_was_refunded(pre, post, action)`; the DRIVER holds
    pre/post so the DRIVER applies it)
  * die-spend / die-kill bookkeeping  :16210-16216 (second-leg die-drag
    suppression), :16363-16375 (spent die killed to the Used box, kill
    deferred past a multi-step first leg), :15731-15758 (deferred kill
    when the chain closes without a die-carrying leg),
    :18150 (_DIE_TO_HUNT_ACTIONS)
  * bonus dice joining the pool       :13196-13232  (minus the driver's
    private ledgers -- the piece id table is the emitter's own)
  * dice recovery                     :13147-13166  (the bulk
    Used-box -> pool move per side; the dump-driven stray-die sweep at
    :13167-13193 stays with the driver, which owns the dump cadence)
  * engine-truth param threading      :16059-16206  (only the members
    that still need POST knowledge; the ones that existed because the
    old `r.state` was the lagging PARSER state die here, because
    ClientView.state IS the engine state)
  * combat-card commit                two-phase (K1): the raw
    'chooses a card' line (emitter op combat_card_pick) under the
    AI's OWN nick = the HIDDEN commit, then ONE '!CALL reveal_chosen'
    per round (render_combat_reveal) reveals every chosen card at
    once; the AI's card identity is never shown or named before the
    human has committed
  * combat protocol chats             :8607-8641 ('nc'/'press'/'stop'/
    'ffN'/'gandalf shines'), :6071-6099 (hit phrasing)
  * casualty physical supply          :9760-10028 semantics ride the
    emitter planner ('casualty_allocate'/'siege_press' in emitter.plan
    with the engine's `_dg_src_log` stamps threaded as `_dg_srcs`,
    :16191-16206): FP dead is permanent, SA dead recycles, and an
    elite downgrade with NO replacement regular is ELIMINATED OUTRIGHT
    (USER ruling 2026-08-18 / REBUILD_PLAN 'PHYSICAL COMPONENT LIMITS')
  * card-name announce decode         :10563-10607 (parser_card_name)

WHAT IS DELIBERATELY NOT LIFTED (per the rebuild plan):
  * `_emit_board_diff` and every consumer of it (emit-fallback,
    emit-completeness) -- rule 6.  `card_option` decisions render via
    emission_card.render_card_effect, the SANCTIONED bounded exception
    (coordinator, 2026-08-19): a delta between the DRIVER's own
    pre/post engine states around ONE decision, with count-conserving
    transfers for interchangeable figures. See emission_card's endpoint
    contract for casualty and physical-supply handling. Other
    `card_*` actions still HOLD from render_action (plays route
    through render_decision).
  * the after-the-fact delta strata (_track_delta_lines,
    _control_marker_lines, _vp_marker_lines, _emit_political_delta on
    diffs, _hand_delta_lines, _siege_transition_lines) -- diff-shaped.
  * the monolithic `_emit_combat` (:8329-8594) -- in the new driver
    every pending-chain decision emits separately via render_decision.
  * the AUTO chain-collapse machinery (hunt reveal resolved in one
    tick, `_flip_owed` double-flip suppression :16217-16223): the new
    driver emits per decision, so each flip has exactly one producer.

CHAT (design rule 7): this module NEVER speaks chat itself.  Chat is
returned as (kind, text) REQUESTS for the router's ChatOut; every kind
used here is a chat.ALLOWED_KINDS entry (the ONE vocabulary).

CONTRACT NOTES:
  * render_action/render_decision take the OPTIONAL committed post
    state.  Everything post-dependent (die-kill timing across
    multi-step dice, deferred kills, bonus-die pool growth, `_dg_srcs`
    threading, Mordor step advance, hunt corruption landing, alive
    hobbit separation region) is exact when `post` is passed; when it
    is None and the answer genuinely needs post, the render HOLDS
    loudly instead of guessing (rules 3 and 6: the engine answers, the
    emitter never re-derives).
  * a HOLD is Emission.hold != None; lines/calls/chat are then empty.
  * fungible pids (design rule 10) resolve inside the emitter planners
    from view.piece_region (+ the corpus setup table for pieces that
    never moved); a shortage raises LookupError there and becomes a
    HOLD here -- pieces are never invented.

ASCII only.
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

import emitter
from emitter import render

# die-carrying actions whose die goes to the HUNT BOX (via the
# planner's die_to_hunt / eye_alloc grammar), never to Used.
_DIE_TO_HUNT_ACTIONS = frozenset(('fellowship_move', 'eye_allocate'))

# actions whose FIRST die-carrying leg opens an engine multi-step
# (pending_army_move_die / pending_muster_die): the die kill is deferred
# until the step closes, so rendering them without `post` cannot answer
# "is the kill due" -- HOLD instead of guessing.
_MULTISTEP_TYPES = frozenset(('army_move', 'muster_recruit'))

# the chain-open predicate's pending fields.
_CHAIN_PENDINGS =('pending_combat', 'pending_hunt',
                   'pending_guide_change', 'pending_card_option',
                   'pending_card_followup', 'pending_discard')

# actions whose gestures depend on the CLIENT's randomness answer
# (the Mordor pawn walk on the hunt tile, the Dwarven Ring face on the
# d6): a pre-apply render (post None, the apply starved) emits
# NOTHING - the server's directive asks the client, and the resumed
# attempt renders the whole action from the post state.
_POST_ONLY_TYPES = frozenset(('dwarven_ring',))


class Emission(object):
    """One render result.

    lines  raw inject lines for the RESP file (client log grammar)
    calls  '!CALL ...' directives, in order (the driver writes lines
           then calls; a line that must FOLLOW a call -- e.g. the die
           kill after a play resolves -- is therefore never packed into
           the same Emission: it is owed, see render_die_kill)
    chat   (kind, text) REQUESTS for the router's ChatOut (rule 7)
    hold   loud reason string when unrenderable; None otherwise
    """

    def __init__(self, lines=None, calls=None, chat=None, hold=None):
        self.lines = list(lines or [])
        self.calls = list(calls or [])
        self.chat = list(chat or [])
        self.hold = hold
        # the planner ops behind self.lines (attached by
        # _ops_to_emission) - the forced-casualty sweep subtracts the
        # rows THIS emission already expressed before rendering the
        # residual, so a planner-rendered allocation is never
        # double-killed client-side.
        self.ops = []

    def __repr__(self):
        if self.hold:
            return 'Emission(HOLD: %s)' % self.hold
        return ('Emission(%d line(s), %d call(s), %d chat)' %
                (len(self.lines), len(self.calls), len(self.chat)))


def _hold(reason):
    return Emission(hold=reason)


# --------------------------------------------------------------------
# the refund predicate
# --------------------------------------------------------------------
def play_was_refunded(pre, post, action):
    """True iff the engine REFUNDED a play_event: the card is still in
    the hand, the action die was not consumed, and NO pending chain was
    opened.  The contract must test ALL THREE conditions (BRIDGE-7 /
    attempt-99: a chain-opening play keeps the card in hand while the
    die IS spent -- a card-only test called 100 of 100 refunds falsely).

    This is the DRIVER's knowledge (it holds pre/post): when it returns
    True the play renders NOTHING -- do not call render_action at all.
    Donor: ai_move_server.py :15903-15956."""
    if getattr(action, 'action_type', None) != 'play_event':
        return False
    from game_state import Side
    cid = (action.params or {}).get('card_id', '')
    if not cid:
        return False
    try:
        sd = (Side.FREE_PEOPLES
              if action.params.get('side', 'fp') == 'fp'
              else Side.SHADOW)
        if cid not in pre.hands[sd].hand or cid not in post.hands[sd].hand:
            return False
        u0 = [bool(d.used) for d in pre.dice.get(sd, [])]
        u1 = [bool(d.used) for d in post.dice.get(sd, [])]
        spent = (len(u0) != len(u1)
                 or any(b and not a for a, b in zip(u0, u1)))
        if spent:
            return False              # the die must be intact
        import ai
        if ai._pending_decision_side(post) is not None:
            return False              # and no chain may be open
        return True
    except Exception:
        return False


def chain_open(state):
    """True while the engine still owes a decision on the action in
    flight (donor :15787-15793)."""
    if state is None:
        return False
    return any(getattr(state, p, None) is not None
               for p in _CHAIN_PENDINGS)


# --------------------------------------------------------------------
# card identity plumbing
# --------------------------------------------------------------------
_parser_name_cache = None


def parser_card_name(cid):
    """The card name AS THE PARSER DECODES IT, for chat announces.
    Donor: :10563-10607 (six explicit aliases; the rest normalize
    1:1 through log_replay's card tables)."""
    global _parser_name_cache
    if _parser_name_cache is None:
        import unicodedata as _ud
        import log_replay as _lr

        def _n(s):
            s = _ud.normalize('NFKD', str(s)).encode(
                'ascii', 'ignore').decode()
            return re.sub(r'[^a-z0-9]', '', s.lower())

        alias = {
            'wisdomofelrond': 'The Wisdom of Elrond',
            'celebornsgaladhrim': "Celeborn's Galadrim",
            'grimbeorntheoldsonofbeorn': 'Grimbeorn the Old',
            'nazgulstrike': 'The Nazgul Strike!',
            'balrogofmoria': 'A Balrog is Come!',
            'musteringsoflongplannedwar':
                'Mustering of a Long-planned War',
        }
        pnames = {}
        for d in (_lr.FP_CHARACTER_CARDS, _lr.FP_STRATEGY_CARDS,
                  _lr.SA_CHARACTER_CARDS, _lr.SA_STRATEGY_CARDS):
            for nm in d.values():
                pnames.setdefault(_n(nm), nm)
        _parser_name_cache = (alias, pnames, _n)
    alias, pnames, _n = _parser_name_cache
    suffix = re.sub(r'^(?:FREE_PEOPLES|SHADOW)_'
                    r'(?:Character|Strategy)_', '', str(cid))
    k = _n(suffix)
    if k in alias:
        return alias[k]
    if k in pnames:
        return pnames[k]
    return suffix.replace('_', ' ')


def _card_pid(view, cid):
    """The client piece id for the AI's OWN card `cid`.

    Live card identity comes ONLY from commitment decode (client fact
    10); ClientView carries the map (card_pid() / card_pid_map, fed by
    the harvest in live_server) and this helper accepts either
    surface.  Returns None when unresolvable -- callers HOLD, they
    never guess a pid."""
    fn = getattr(view, 'card_pid', None)
    if callable(fn):
        try:
            pid = fn(cid)
            if pid:
                return str(pid)
        except Exception:
            return None
    m = getattr(view, 'card_pid_map', None)
    if isinstance(m, dict):
        pid = m.get(cid)
        if pid:
            return str(pid)
    return None


# --------------------------------------------------------------------
# chat-kind mapping (rule 7: requests, never spoken here)
# --------------------------------------------------------------------
def _chat_kind(action_type, msg):
    """Map an emission chat text to chat.py's ALLOWED_KINDS registry —
    the ONE chat vocabulary (a second kind list here was the
    two-vocabularies defect at a fresh seam; caught by the composed
    smoke 2026-08-19).  Unmapped text returns 'prompt', which ChatOut
    REFUSES for unregistered prose — loud, never silently mislabeled."""
    m = str(msg)
    if action_type == 'army_attack':
        return 'attack_declare'
    if m == 'nc':
        return 'nc'
    if m == '0e':
        return 'zero_eyes'
    if m == 'pass':
        return 'pass'
    if re.match(r'^ff\d+$', m):
        return 'ff'
    if m.startswith('cc: '):
        return 'cc'
    if m.startswith('plays '):
        return 'card_announce'
    if re.search(r': i choose \d+$', m):
        return 'magnitude'
    return 'prompt'


_DIE_PIDS = frozenset(emitter._FP_DIE_IDS) | frozenset(emitter._SH_DIE_IDS)

# ops that RELOCATE unit-family pieces to their 'to' zone vs ops that
# REMOVE them from the board (swap_place = Controls.304 removes the
# downgraded elites; hand_exit_b = the Controls.296 casualty kill).
_RETRACK_TO = frozenset(('move', 'card_move', 'move_637', 'move_662',
                         'marker_366', 'token_271', 'swap_remove',
                         'hand_exit'))
_RETRACK_DEAD = frozenset(('hand_exit_b', 'swap_place'))


def _retrack_ops(view, ops):
    """Reflect the emitted piece movement in the view's client model so
    later fungible picks in the SAME chain never re-select a piece this
    emission already moved (donor _retrack :10005-10011; without it the
    muster die's second recruit picked the SAME pool pid twice).  The
    next client dump refresh overwrites these entries with the client's
    own truth -- this is a provisional client-model update, never a
    rules authority (rule 1)."""
    import log_replay as _lr
    for op, f in ops:
        pids = list(f.get('pieces') or ())
        if not pids and f.get('piece'):
            pids = [f['piece']]
        pids = [str(p) for p in pids if str(p) not in _DIE_PIDS]
        if not pids:
            continue
        if op in _RETRACK_DEAD:
            for p in pids:
                view.piece_region[p] = emitter._native_kill_zone(view, p, default='dead')
        elif op in _RETRACK_TO and f.get('to'):
            name = _lr.decode_location(str(f['to'])) or str(f['to'])
            for p in pids:
                view.piece_region[p] = str(name)


def _ops_to_emission(view, nick, action_type, ops):
    """[(op, fields)] -> Emission: chat ops become (kind, text)
    requests, 'directive' ops become calls, the rest render to lines.
    Emitted piece movement is retracked into the view (see
    _retrack_ops)."""
    em = Emission()
    for op, f in ops:
        if op == 'chat':
            em.chat.append((_chat_kind(action_type, f.get('msg', '')),
                            str(f.get('msg', ''))))
        elif op == 'directive':
            em.calls.append(str(f.get('line', '')))
        else:
            em.lines.append(render(op, f, local=False))
    em.ops = list(ops)          # WAVE-I: expressed-row provenance
    _retrack_ops(view, ops)
    return em


# --------------------------------------------------------------------
# die bookkeeping helpers
# --------------------------------------------------------------------
def _die_kill_line(view, nick, side_name, die_idx, extra_pids=()):
    dp = emitter.die_piece_for(view, side_name, die_idx)
    return render('kill', {'nick': nick,
                           'pieces': [dp] + [str(p) for p in
                                             (extra_pids or ())],
                           'to': emitter.DICE_USED}, local=False)


def render_die_kill(view, nick, side, die_idx, extra_pids=()):
    """A spent die killed to the Used box, as its own Emission -- the
    DRIVER flushes this for the kills that must FOLLOW a '!CALL' or a
    chain close it owns (play_event, army_attack, draw_card: USER
    invariant 'die -> action -> resolve everything -> kill die',
    donor :15760-15829).

    extra_pids (WAVE-J slot cleanup): pieces that die WITH the die in
    one group kill -- the corpus E1 shape 'kills group (Free Action
    Die (P), <Card Title>) in Current Action.': a PLAYED EVENT CARD
    leaves the Current-Card slot at action close, together with its
    die.  The old die-only flush stranded every played card in the
    slot forever (wjg1/wjg2 live: pids 177/208 parked in SA Card
    Play; the witnesses are BLIND to non-region zones)."""
    side_name = emitter.canon_side(side)
    try:
        return Emission(lines=[_die_kill_line(view, nick, side_name,
                                              die_idx, extra_pids)])
    except LookupError as e:
        return _hold('die_kill %s idx %s: %s' % (side_name, die_idx, e))


def _pool_growth_lines(view, nick, pre, post):
    """Render a newly earned die using its observed spare, if already bound.

    An earned piece already in a playable/recoverable zone needs no move.
    """
    from game_state import Side
    out = []
    for s in (Side.FREE_PEOPLES, Side.SHADOW):
        n0 = len(pre.dice.get(s, []))
        n1 = len(post.dice.get(s, []))
        base = (emitter._FP_DIE_IDS if s is Side.FREE_PEOPLES
                else emitter._SH_DIE_IDS)
        for i in range(n0, n1):
            if i >= len(base):
                raise LookupError(
                    'bonus die index %d beyond known %s piece ids'
                    % (i, s.name))
            is_fp = s is Side.FREE_PEOPLES
            # The player may have taken a different interchangeable spare.
            # Use its observed identity, and do not create another die when
            # that earned piece is already on the table for recovery.
            by_index = {index: str(pid) for pid, (owner, index) in view.die_id_map.items()
                        if owner is s}
            pid = by_index.get(i, base[i])
            zone = str((getattr(view, 'piece_region', {}) or {}).get(pid, '')).lower()
            prefix = 'fp' if is_fp else 'sa'
            if zone in (prefix + ' dice', prefix + ' dead', prefix + ' casualties',
                        'hunt box', 'the hunt for the ring'):
                continue
            out.append(render('move', {
                'nick': nick, 'pieces': [pid],
                'frm': emitter.FP_POOL if is_fp else emitter.SH_POOL,
                'to': (emitter.FP_DICE_POOL if is_fp
                       else emitter.SH_DICE_POOL)}, local=False))
    return out


def _deferred_die_kill(view, nick, side_name, pre, post, action):
    """A multi-step die whose chain closed WITHOUT a die-carrying leg
    (a pass / die-less leg 2) still gets its kill, or it strands in
    Current Action forever (donor :15731-15758, the die-105 deadlock).
    Returns [] when nothing is owed."""
    out = []
    for pf in ('pending_army_move_die', 'pending_muster_die'):
        pidx = getattr(pre, pf, None)
        if pidx is None or getattr(post, pf, None) is not None:
            continue
        if action.die_idx is not None and action.die_idx >= 0:
            continue      # the closing leg carried the die itself
        out.append(_die_kill_line(view, nick, side_name, int(pidx)))
    return out


# --------------------------------------------------------------------
# engine-truth param threading (post-dependent members only:
# view.state IS the engine state, so no parser-lag member exists)
# --------------------------------------------------------------------
def _threaded_params(view, side_name, action, post):
    a = action
    at = a.action_type
    params = a.params
    st = view.state

    if at in ('muster_recruit', 'muster_elite', 'transform_gandalf'):
        # SIEGE musters join the garrison INSIDE (asked through the ONE
        # function, of the ENGINE's state).
        from board import sanitize
        reg = sanitize(str(params.get('region', '')))
        if emitter.garrison_side_of(st, reg) == \
                emitter.canon_side(side_name):
            params = dict(params)
            params['_besieged_garrison'] = True

    if at in ('army_move', 'victor_advance') and post is not None \
            and not params.get('moved_by_nation'):
        # K5: the ENGINE moves SPECIFIC units - the exact per-(nation,
        # kind) composition that LEFT the source between pre and post
        # rides the params; plan_army_move honours 'moved_by_nation'
        # (a nation-blind pick would ship the wrong nation's regular)
        from board import sanitize as _san
        _src = _san(str(params.get('from', '')))

        def _comp(state):
            for _k, _army in state.armies.items():
                if _san(str(_k)) != _src:
                    continue
                _d = {}
                for _u in _army.units:
                    _key = '%s:%s' % (
                        str(getattr(_u.nation, 'value',
                                    _u.nation)).lower(),
                        str(getattr(_u.kind, 'value', _u.kind)).lower())
                    _d[_key] = _d.get(_key, 0) + 1
                return _d
            return {}
        _c0, _c1 = _comp(st), _comp(post)
        _mv = {}
        for _k, _n in _c0.items():
            _dd = _n - _c1.get(_k, 0)
            if _dd > 0:
                _mv[_k] = _dd
        if _mv:
            params = dict(params)
            params['moved_by_nation'] = _mv

    if at == 'hunt_take_corruption' and post is not None \
            and 'corruption_target' not in params:
        # the ENGINE's true landing spot; without post the planner
        # reads the ENGINE pending_hunt damage, which is already the
        # rules answer
        params = dict(params)
        params['corruption_target'] = int(post.fellowship.corruption)

    if at == 'fellowship_move' \
            and getattr(st.fellowship, 'mordor_entered', False):
        # Mordor track: the pawn advances only when the ENGINE's step
        # advanced.  Needs post; callers HOLD when it is absent
        # (checked in render_action).
        params = dict(params)
        params['from_step'] = int(getattr(st.fellowship,
                                          'mordor_step', 0))
        if post is not None:
            params['mordor_post_step'] = int(
                getattr(post.fellowship, 'mordor_step', 0))

    if at == 'hunt_sacrifice_guide':
        # name the ENGINE's guide and, given post, the engine's true
        # ALIVE-SEPARATION placement region (USER ruling 2026-08-15).
        params = dict(params)
        g = getattr(st.fellowship, 'guide', None)
        if g is not None and not params.get('companion'):
            params['companion'] = g.name
        if post is not None and not params.get('to_region'):
            from game_state import Companion
            from board import sanitize
            nm = params.get('companion')
            for k, army in post.armies.items():
                for ld in (getattr(army, 'leaders', []) or []):
                    if isinstance(ld, Companion) and str(ld.name) == nm:
                        params['to_region'] = sanitize(str(k))
                        break

    if at == 'dwarven_ring' and post is not None \
            and 'new_face' not in params:
        # the rerolled die's face is the ENGINE's (landed from the
        # client's d6 through the randomness funnel)
        from game_state import Side
        rs = (Side.SHADOW if str(params.get('side', 'fp')).lower()
              == 'sh' else Side.FREE_PEOPLES)
        params = dict(params)
        params['new_face'] = post.dice[rs][int(params['die_idx'])].current

    if at == 'eye_allocate' and not params.get('die_indices'):
        # the engine allocates the FIRST n unused Shadow dice -- the
        # executor's own deterministic rule (ai.py :10727-10732),
        # mirrored so the emission names the same physical dice the
        # engine banked.  This is the engine event, not a diff.
        from game_state import Side
        n = params.get('count')
        n = 1 if n is None else int(n)
        if n > 0:
            idxs = [i for i, d in enumerate(st.dice.get(Side.SHADOW, []))
                    if not d.used][:n]
            params = dict(params)
            params['die_indices'] = idxs
    return params


# --------------------------------------------------------------------
# render_action: one engine ACTION -> Emission
# --------------------------------------------------------------------
def _norm_card_name(s):
    """Accent- and punctuation-insensitive card-name key ('The
    Palantir of Orthanc' == 'The Palantír of Orthanc')."""
    import re as _re
    import unicodedata as _ud
    s = _ud.normalize('NFKD', str(s))
    s = s.encode('ascii', 'ignore').decode('ascii')
    return _re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip()


# no static card-pid table (USER 2026-08-20): piece ids are the
# CLIENT's to state, never ours to hardcode - a table-kill target comes
# from driver.table_seats (the slot we watched the card go into)
# resolved against the client dump.
# The actor's own Current-Card zone (2036 renders as 'Current SA Card',
# 2037 as 'Current FP Card').
_CARD_PLAY_ZONE = {'SHADOW': 'Game.2036', 'FREE_PEOPLES': 'Game.2037'}


def _plan_discard_table_card(view, nick, side_name, a, die_idx, st,
                             post):
    """Ruling 1.149: spend a die to remove the OPPONENT's play-on-table
    card.  The engine offers it (ai._get_table_kill_actions) and the
    live stack had NO planner, so the AI's choice raised EMISSION HOLD
    -> DriverHalt on every cycle forever, with the action ALREADY
    applied engine-side (USER 2026-08-20: 'emission hold: no planner
    for "discard_table_card"').

    CORPUS GROUND TRUTH -- the Shadow kill of 'A Power too Great',
    2023-01-20_183326_FP_CircleDakkon_SP_GenericPie L583-591:

        moves Shadow Action Die (M) from SA Dice to Current Action.
        moves A Power Too Great/No Quarter from Board1 to Current SA Card.
        kills Shadow Character Card in Shadow Hand.
        kills Shadow Strategy Card(Army) in Shadow Hand.
        kills group (Shadow Action Die (M), A Power Too Great/No
                     Quarter) in Current Action.

    So: die spend, the target dragged out of its Board slot into the
    ACTOR's Current-Card zone, the payment discarded from the actor's
    hand as MASKED hand kills (the client renders a hand card
    generically -- identities are never revealed by a payment), and
    the die + card die together in ONE group.  That last line is the
    E1 group kill the driver already owes (_note_owed_die_kill /
    _note_owed_card_kill), so it is NOT emitted here.

    The payment cards are not in the action params -- only the target
    is -- so they are read from the POST state the way
    hunt_sacrifice_random reads its victim: whatever left the actor's
    hand paid for this."""
    params = a.params or {}
    key = _norm_card_name(params.get('card_name')
                          or params.get('card_id') or '')
    # THE PIECE COMES FROM WHERE WE WATCHED IT BE SEATED, never from a
    # static id table (USER 2026-08-20: "it should see where I placed
    # the power of tom bombadil from the rendered log, and delete it
    # there").  The static table said 145 was 'The Power of Tom
    # Bombadil'; this client renders 145 as 'Kindred of Glorfindel',
    # so the AI dragged THAT out of the FP Strategy Deck and killed it.
    # driver.table_seats is fed by the feed (name + slot) and resolves
    # the pid from the client's own dump - the client being the only
    # authority on where its pieces are.
    pid = None
    seats = getattr(view, 'table_seats', None)
    getter = getattr(view, 'table_seat_pid', None)
    if callable(getter):
        pid = getter(key)
    elif isinstance(seats, dict) and key in seats:
        pid = (seats[key] or {}).get('pid')
    if pid is None:
        return _hold(
            'discard_table_card %r: the client dump has not placed '
            'that card in a Board slot, so its piece id is unknown - '
            'refusing to guess one (a static id table shipped the '
            'WRONG card once already)'
            % (str(params.get('card_name'))[:48],))
    pid = str(pid)
    dest = _CARD_PLAY_ZONE.get(side_name)
    if dest is None:
        return _hold('discard_table_card: no Current-Card zone for '
                     'side %r' % (side_name,))
    em = Emission()
    if die_idx >= 0:
        try:
            op, f = emitter.plan_die_spend(view, nick, side_name,
                                           die_idx)
            em.lines.append(render(op, f, local=False))
        except LookupError as e:
            return _hold('discard_table_card die spend: %s' % e)
    # the target out of whatever Board slot it actually sits in
    try:
        frm = emitter._piece_zone_code(view, pid, default=None)
    except LookupError:
        return _hold(
            'discard_table_card: the client dump does not place table '
            'card pid %s (it should be in a Board slot) - refusing to '
            'guess its origin' % (pid,))
    em.lines.append(render('move', {
        'nick': nick, 'pieces': [pid], 'frm': frm, 'to': dest},
        local=False))
    # the FP ring payment: the engine moves an Elven Ring FP -> SH
    # (fp_elven_rings -= 1, sh_elven_rings += 1); the corpus gesture is
    # the ring marker's active-slot -> Dark Ring walk right before the
    # card leaves its slot ('moves Nenya from Elven Ring 2 to Dark Ring
    # 1' / 'moves The Palantir ... from Board1 to Current FP Card'),
    # the same Controls.366 walk the elven_ring planner emits.
    if str(params.get('payment')) == 'ring':
        try:
            rpid, rfrm, rto = emitter.pick_fp_ring_walk(view)
        except LookupError as e:
            return _hold('discard_table_card ring payment: %s' % e)
        em.lines.insert(len(em.lines) - 1, render('marker_366', {
            'nick': nick, 'piece': rpid, 'frm': rfrm, 'to': rto},
            local=False))
        import log_replay as _lr
        view.piece_region[rpid] = str(_lr.decode_location(rto) or rto)
    # the payment: whatever left the actor's hand (Shadow pays one
    # Army + one Character card; FP pays a Will die alone, or a die
    # plus the ring walk above -- neither touches the hand, so an
    # empty hand diff is CORRECT there, not a missing render).
    if post is not None:
        import game_state as _gs
        side_enum = (_gs.Side.SHADOW if side_name == 'SHADOW'
                     else _gs.Side.FREE_PEOPLES)

        def _hand(state):
            try:
                return list(state.hands[side_enum].hand)
            except Exception:
                return []
        before = list(_hand(st))
        for cid in _hand(post):
            if cid in before:
                before.remove(cid)
        hand_zone = ('Controls.46' if side_name == 'SHADOW'
                     else 'Controls.45')
        for cid in before:
            cpid = _card_pid(view, cid)
            if cpid is None:
                return _hold(
                    'discard_table_card payment: no client pid for the '
                    "AI's own card %r (commitment-decode gap) - "
                    'refusing to discard a guessed piece'
                    % (str(cid)[:48],))
            em.lines.append(render('hand_exit_b', {
                'nick': nick, 'pieces': [cpid], 'frm': hand_zone},
                local=False))
    return em


def _forced_guide_lines(view, nick, pre, post):
    """Seat an engine-resolved replacement after the previous Guide leaves.

    A remaining tie still has a pending choice and emits nothing here. The
    departing Guide's own action already emits its death/separation; only the
    forced surviving replacement needs this additional physical gesture.
    """
    if post is None or getattr(post, 'pending_guide_change', None):
        return []
    before = getattr(pre.fellowship, 'guide', None)
    after = getattr(post.fellowship, 'guide', None)
    old_name = getattr(before, 'name', None)
    new_name = getattr(after, 'name', None)
    # Gollum enters as Guide after the last Companion leaves. Unlike an
    # ordinary successor he is not a member of fellowship.companions, so the
    # living-companion guard below intentionally cannot handle this entry.
    if new_name == 'Gollum' and old_name != 'Gollum' and not post.fellowship.companions:
        pid = emitter._companion_pid('Gollum')
        source = emitter._piece_zone_code(view, pid)
        if source == emitter.GUIDE_BOX:
            return []
        if source not in (emitter.FP_POOL, emitter.FELLOWSHIP_BOX):
            raise LookupError('Entering Gollum is not in FP reinforcements or Fellowship box')
        return [render('move', {'nick': nick, 'pieces': [pid],
                                'frm': source, 'to': emitter.GUIDE_BOX}, local=False)]
    living = {c.name for c in post.fellowship.companions}
    if not old_name or not new_name or old_name == new_name \
            or old_name in living or new_name not in living:
        return []
    pid = emitter._companion_pid(new_name)
    source = emitter._piece_zone_code(view, pid,
                                     default=emitter.FELLOWSHIP_BOX)
    if source == emitter.GUIDE_BOX:
        return []
    if source != emitter.FELLOWSHIP_BOX:
        raise LookupError('forced Guide %s is not in the Fellowship box' % new_name)
    return [render('move', {'nick': nick, 'pieces': [pid],
                            'frm': source, 'to': emitter.GUIDE_BOX}, local=False)]


def render_action(view, nick, side, action, post=None, ring_paid=None):
    """Render the client gestures for ONE applied engine action.

    view   ClientView (engine state + client dump + die bindings)
    side   Side enum or name (the actor)
    action the engine Action (action_type, params, die_idx)
    post   the committed post-action engine state, when the driver has
           it (it always does after a successful apply).  Post-
           dependent bookkeeping HOLDS without it rather than guessing.
    ring_paid  optional set: the driver's FP elven-ring departure
           ledger (donor :16182-16190), threaded to pick_fp_ring_walk.

    A refunded play renders NOTHING -- the DRIVER tests
    play_was_refunded(pre, post, action) and skips this call.
    """
    side_name = emitter.canon_side(side)
    st = view.state
    a = action
    at = a.action_type
    die_idx = a.die_idx if a.die_idx is not None else -1
    pre_pending = (st.pending_army_move_die is not None
                   or getattr(st, 'pending_muster_die', None) is not None)

    # ---- pass: chat only + a possibly owed multi-step die kill ------
    if at == 'pass':
        # kind 'pass': a ('nc', 'pass') pair fails the 'nc' kind's
        # ^nc$ validator at the ChatOut exit
        em = Emission(chat=[('pass', 'pass')])
        if pre_pending:
            if post is None:
                return _hold('pass with an open multi-step die needs '
                             'the post state to settle the deferred '
                             'die kill (donor :15731-15758)')
            # a chain-closing pass CONSUMES the pending die (measured:
            # the engine marks it used) and a pass has no other kill
            # producer -- so unlike the generic deferred helper, the
            # kill lands here even when the pass action itself carries
            # the pending die's index.
            try:
                for pf in ('pending_army_move_die',
                           'pending_muster_die'):
                    pidx = getattr(st, pf, None)
                    if pidx is not None \
                            and getattr(post, pf, None) is None:
                        em.lines.append(_die_kill_line(
                            view, nick, side_name, int(pidx)))
            except LookupError as e:
                return _hold('pass deferred die kill: %s' % e)
        return em

    # ---- play_event: staged prologue ----------------------------------
    if at == 'play_event':
        cid = a.params.get('card_id', '')
        pid = _card_pid(view, cid)
        if pid is None:
            return _hold(
                'play_event %s: no client pid for the AI\'s own card '
                '(commitment-decode map not on ClientView -- reported '
                'gap); refusing to guess a piece' % (cid[:48],))
        em = Emission()
        if die_idx >= 0:
            try:
                op, f = emitter.plan_die_spend(view, nick, side_name,
                                               die_idx)
                em.lines.append(render(op, f, local=False))
            except LookupError as e:
                return _hold('play_event die spend: %s' % e)
        em.chat.append(('card_announce',
                        'plays %s' % parser_card_name(cid).lower()))
        em.calls.append('!CALL play_card %s' % pid)
        # NO die kill here: the kill must FOLLOW the play's resolution
        # (USER invariant 'die -> play -> resolve -> kill') and the
        # resolution rides the !CALL -- the driver flushes
        # render_die_kill when the chain closes.
        return em

    # ---- army_attack: declaration only (no board lines) -------------
    if at == 'army_attack':
        em = Emission()
        if die_idx >= 0:
            try:
                op, f = emitter.plan_die_spend(view, nick, side_name,
                                               die_idx)
                em.lines.append(render(op, f, local=False))
            except LookupError as e:
                return _hold('army_attack die spend: %s' % e)
        from board import sanitize
        frm = sanitize(str(a.params.get('from', '')))
        to = sanitize(str(a.params.get('to', '')))
        msg = to if frm == to else '%s from %s' % (to, frm)
        em.chat.append(('attack_declare', msg))
        # The battle die is killed at battle end. The shared commit
        # sweep emits declaration-triggered political activation/advance.
        return em

    # ---- draw_card: the CLIENT draws (directive), engine ingests ----
    if at == 'draw_card':
        em = Emission()
        if die_idx >= 0:
            try:
                op, f = emitter.plan_die_spend(view, nick, side_name,
                                               die_idx)
                em.lines.append(render(op, f, local=False))
            except LookupError as e:
                return _hold('draw_card die spend: %s' % e)
        tag = 'sa' if side_name == 'SHADOW' else 'fp'
        deck = ('strat' if str(a.params.get('deck', 'character')
                               ).lower().startswith('strat') else 'char')
        em.calls.append('!CALL draw_%s_%s' % (tag, deck))
        # die kill follows the draw's resolution: driver-owed.
        return em

    # ---- discard_card: the engine names the card ---------------------
    if at == 'discard_card':
        cid = a.params.get('card_id', '')
        pid = _card_pid(view, cid)
        if pid is None:
            return _hold('discard_card %s: no client pid for the '
                         'AI\'s own card (reported ClientView gap)'
                         % (cid[:48],))
        hand = ('Controls.46' if side_name == 'SHADOW'
                else 'Controls.45')
        # G19: discards ride the 296 kill-in-hand grammar (piece, hand).
        em = Emission(lines=[render('hand_exit_b', {
            'nick': nick, 'pieces': [pid], 'frm': hand}, local=False)])
        return em

    # ---- discard_table_card: the 1.149 OPPONENT table-card kill -----
    if at == 'discard_table_card':
        return _plan_discard_table_card(view, nick, side_name, a,
                                        die_idx, st, post)

    # ---- use_token with a draw half: directive, not a picked card ---
    if at == 'use_token' and str(a.params.get('token_type')) != 'muster':
        pid = '427'
        frm = emitter._piece_zone_code(view, pid,
                                       default=emitter.FP_POOL)
        em = Emission(lines=[render('move', {
            'nick': nick, 'pieces': [pid], 'frm': frm,
            'to': emitter.DICE_USED}, local=False)])
        tag = 'sa' if side_name == 'SHADOW' else 'fp'
        deck = ('strat' if str(a.params.get('deck', 'character')
                               ).lower().startswith('strat') else 'char')
        em.calls.append('!CALL draw_%s_%s' % (tag, deck))
        return em

    # ---- card_* effects: ACTION-path renders none (rule 6).  The
    # card_option DECISION renders via render_decision's sanctioned
    # delta-scoped path (emission_card), which needs pre/post -- an
    # action-path caller has neither, so this stays a loud HOLD.
    if at.startswith('card_'):
        return _hold(
            "no planner renders %r on the action path -- card_option "
            "decisions render via render_decision(pre, post) "
            "(emission_card); the retired _emit_board_diff differ is "
            "never the fallback" % (at,))

    # ---- generic planner dispatch ------------------------------------
    if post is None and die_idx >= 0 and at in _MULTISTEP_TYPES:
        return _hold(
            '%s with a die needs the post state: only the engine '
            'knows whether this leg opened a multi-step (pending die '
            'fields), and the die-kill timing follows that '
            '(donor :16363-16375)' % (at,))
    if post is None and pre_pending and die_idx < 0:
        return _hold(
            '%s while a multi-step die is open needs the post state '
            'to settle the deferred die kill (donor :15731-15758)'
            % (at,))

    params = _threaded_params(view, side_name, a, post)
    if post is None and (at in _POST_ONLY_TYPES or (
            at == 'fellowship_move'
            and getattr(st.fellowship, 'mordor_entered', False))):
        return Emission()
    if at == 'elven_ring' and ring_paid is not None:
        params = dict(params)
        params['_ring_paid'] = ring_paid

    try:
        ops = emitter.plan(at, params, view, nick, side_name,
                           die_idx=(die_idx if die_idx >= 0 else None),
                           rng=None)
    except NotImplementedError as e:
        return _hold('no planner for %r: %s' % (at, e))
    except LookupError as e:
        return _hold('%s: boards cannot supply the pieces: %s'
                     % (at, e))

    # second leg of a multi-step die: the die was already dragged on
    # the first leg
    if pre_pending:
        ops = [(op, f) for op, f in ops
               if not str(op).startswith('die_spend')]

    em = _ops_to_emission(view, nick, at, ops)

    # the LEARNED zero-eye choice is visible only in chat (D3: '0e';
    # absence-is-the-choice)
    if at == 'eye_allocate' and int(params.get('count') or 0) == 0:
        em.chat.append(('zero_eyes', '0e'))

    # ---- die kill to the Used box --------------------------------------
    if die_idx >= 0 and at not in _DIE_TO_HUNT_ACTIONS:
        due = True
        if post is not None:
            due = (post.pending_army_move_die is None
                   and getattr(post, 'pending_muster_die', None) is None)
        elif at in _MULTISTEP_TYPES:      # unreachable (held above)
            due = False
        if due:
            try:
                em.lines.append(_die_kill_line(view, nick, side_name,
                                               die_idx))
            except LookupError as e:
                return _hold('%s die kill: %s' % (at, e))

    if post is not None:
        try:
            em.lines += _deferred_die_kill(view, nick, side_name, st,
                                           post, a)
            em.lines += _pool_growth_lines(view, nick, st, post)
            em.lines += _forced_guide_lines(view, nick, st, post)
        except LookupError as e:
            return _hold('%s post bookkeeping: %s' % (at, e))
    return em


# --------------------------------------------------------------------
# render_decision: one pending-chain DECISION -> Emission
# --------------------------------------------------------------------
def dead_men_arrival_prologue(view, nick, side, action):
    """Render the validated initial Dead Men movement before requesting dice.

    Return the reached engine clone and its emission. Other card decisions
    return None. The live step retains this clone only to subtract the already
    acknowledged arrival from its eventual post-roll rendering.
    """
    import ai
    import event_effects
    from board import sanitize
    from game_state import Companion, Side
    from . import emission_card
    params = getattr(action, 'params', {}) or {}
    cid = 'FREE_PEOPLES_Character_Dead_Men_of_Dunharrow'
    choices = params.get('choices')
    if (getattr(action, 'action_type', None) != 'card_option'
            or params.get('card_id') != cid or not isinstance(choices, dict)
            or 'companions' not in choices or 'target_region' not in choices):
        return None
    pre = view.state
    pending = getattr(pre, 'pending_card_option', None) or {}
    if (side is not Side.FREE_PEOPLES or pending.get('card_id') != cid
            or choices not in pending.get('options', ())):
        raise LookupError('Dead Men arrival is not the current legal option')
    names = choices['companions']
    if (not isinstance(names, (list, tuple)) or not names
            or len(set(names)) != len(names)):
        raise LookupError('Dead Men arrival has an invalid companion group')
    sources = []
    for name in names:
        found = [region for region, army in pre.armies.items()
                 for companion in army.leaders
                 if isinstance(companion, Companion) and companion.name == name]
        if len(found) != 1:
            raise LookupError('Dead Men companion %r has no unique source' % name)
        sources.append(sanitize(str(found[0])))
    if len(set(sources)) != 1 or sources[0] == sanitize(choices['target_region']):
        raise LookupError('Dead Men companions do not share the departure region')
    # The explicit planner checks each selected identity against the actual
    # client source; the card delta renderer alone can otherwise use any seat.
    emitter.plan_companion_move(view, nick, 'FREE_PEOPLES', {
        'from': sources[0], 'to': choices['target_region'], 'companions': names})
    arrival = ai.clone_state(pre)
    if not event_effects.dead_men_arrival_before_roll(
            arrival, choices['target_region'], names):
        return None
    em = emission_card.render_card_effect(view, nick, side, 'card_option',
                                          action, pre, arrival)
    if em.hold:
        raise LookupError('Dead Men arrival: %s' % em.hold)
    return arrival, em


def render_decision(view, nick, side, kind, action, post=None,
                    pre=None):
    """Render one pending-chain decision (combat / hunt / casualty /
    guide / discard / card chains).  Each decision emits separately --
    the monolithic whole-battle emitter is NOT lifted.

    kind is ai.describe_pending()['kind'] (context for holds); dispatch
    keys on action.action_type.  `post` (the applied state) is needed
    for `_dg_srcs` threading on casualty decisions -- the engine's
    own downgrade-source stamps (combat.py `_dg_src_log`; donor
    :16191-16206) -- and, together with `pre` (the state the decision
    was applied TO; defaults to view.state, which IS that state while
    the driver has not yet committed), for the card-effect row renderer
    (emission_card.render_card_effect, the bounded rule-6 exception).
    """
    side_name = emitter.canon_side(side)
    st = view.state
    a = action
    at = a.action_type

    # ---- the AI's own hand-limit discard: the DECISION path renders
    # it exactly as render_action does ----------------------------------
    if at == 'discard_card':
        return render_action(view, nick, side, a, post=post)

    # ---- combat-card commit: HIDDEN (K1 step 2, R4) ------------------
    if at == 'combat_card_play':
        cid = a.params.get('card_id', '')
        pid = _card_pid(view, cid)
        if pid is None:
            return _hold(
                'combat_card_play %s: no client pid for the AI\'s own '
                'card (commitment-decode map not on ClientView -- '
                'reported gap)' % (str(cid)[:48],))
        # the hidden commit is the raw 'chooses a card' line under the
        # AI's OWN nick (emitter op combat_card_pick; the identity stays
        # hidden until render_combat_reveal, so the human commits
        # without seeing the AI's card.  No chat, no play_card).
        # Invariant: the client keeps ONE chosen card per NICK
        # (ay_0.l(): the first chooser's nick fills slot J, a second
        # nick fills K, the SAME nick replaces its own card), so the
        # F1 handler ('!CALL choose_card', local nick) would overwrite
        # the human's pending commit on a one-client board.
        return Emission(lines=[render('combat_card_pick',
                                      {'nick': nick, 'piece': str(pid)},
                                      local=False)])

    # combat protocol chats: each text under ITS OWN registered kind
    # (the ChatOut validators refuse a text under another kind's form)
    if at == 'siege_assault':
        # K8: the attacker_siege_decision -> assault has NO board
        # gesture of its own - the region was already declared by the
        # army_attack that OPENED the pending (corpus: after the
        # withdrawal into the Stronghold the attacker chats the region,
        # then the assault round rolls); the engine runs a combat round
        # here, whose d6 rolls the funnel asks and whose
        # hits/casualties get their own producers.  An empty emission
        # lets the round proceed.  siege_continue (maintain, no round)
        # likewise emits nothing new here.
        return Emission()
    if at == 'combat_card_none':
        return Emission(chat=[('nc', 'nc')])
    if at in ('combat_cease', 'siege_no_press'):
        return Emission(chat=[('stop', 'stop')])
    if at in ('combat_continue', 'siege_continue'):
        return Emission(chat=[('press', 'press')])
    if at == 'shine':
        return Emission(chat=[('shine', 'gandalf shines')])
    if at == 'shine_none':
        return Emission()
    if at == 'combat_effect_choice':
        # Count decisions announce a magnitude. Exact round payment choices
        # change pieces through the driver's existing post-commit forced rows;
        # they must not invent an additional zero-magnitude announcement.
        if all(key in a.params for key in ('nation', 'kind', 'operation')):
            return Emission()
        from .chat import combat_effect_choice_note
        return Emission(chat=[('prompt', combat_effect_choice_note(a.params))])

    # ---- EX-8 (2026-08-29): the Shadow's table-kill PAYMENT pair is a
    # learned card_option (ai._open_table_kill_payment -> the chosen
    # pair leaves the hand via ai.pay_table_card_kill).  The corpus
    # renders the payment as MASKED hand kills (one per card, the
    # identities never revealed) - the same lines the action path
    # rendered when the engine auto-paid.  Rendered by the DECISION,
    # never by a hand diff.
    _chx = (a.params or {}).get('choices')
    if at == 'card_option' and isinstance(_chx, dict) \
            and _chx.get('army_card') and _chx.get('character_card'):
        em = Emission()
        for cid in (_chx['army_card'], _chx['character_card']):
            cpid = _card_pid(view, cid)
            if cpid is None:
                return _hold(
                    'table-kill payment: no client pid for the AI\'s own '
                    'card %r (commitment-decode gap) - refusing to '
                    'discard a guessed piece' % (str(cid)[:48],))
            em.lines.append(render('hand_exit_b', {
                'nick': nick, 'pieces': [cpid],
                'frm': 'Controls.46' if side_name == 'SHADOW'
                else 'Controls.45'}, local=False))
        return em
    _card_casualty = getattr(st, 'pending_card_option', None) or {}
    if at == 'card_option' and _card_casualty.get('card_casualty'):
        params = emitter.casualty_action_params(a)
        if params is None or a.params.get('card_id') != _card_casualty.get('card_id'):
            return _hold('card casualty action does not identify this pending allocation')
        if post is not None:
            n0 = len(getattr(st, '_dg_src_log', None) or [])
            params['_dg_srcs'] = list(getattr(post, '_dg_src_log', None) or [])[n0:]
        try:
            ops = emitter.plan_casualty_operation(params, view, nick, side_name)
        except (LookupError, ValueError) as error:
            return _hold('card casualty operation: %s' % error)
        return _ops_to_emission(view, nick, at, ops)

    # ---- card_option: the SANCTIONED delta-scoped row renderer ------
    # (bounded exception to rule 6, coordinator 2026-08-19; provenance
    # + row rules in emission_card's docstring.  card_followup KIND
    # decisions carry real army_move/army_attack/followup_done actions
    # and ride their own planners below.)
    if at == 'card_option':
        from . import emission_card
        em = emission_card.render_card_effect(view, nick, side, kind,
                                              a, pre, post)
        if not em.hold:
            try:
                em.lines += _forced_guide_lines(view, nick, pre or st, post)
            except LookupError as e:
                return _hold('%s forced Guide: %s' % (at, e))
        return em
    if at.startswith('card_'):
        return _hold(
            "no planner renders %r (pending kind %r) -- the old "
            "driver rendered card chains via the retired board diff"
            % (at, kind))

    # ---- casualty family: thread the ENGINE's downgrade sources -----
    params = a.params
    if at in ('casualty_allocate', 'siege_press',
              'casualty_preserve_units', 'casualty_preserve_elites') \
            and post is not None and '_dg_srcs' not in params:
        n0 = len(getattr(st, '_dg_src_log', []) or [])
        dgs = list(getattr(post, '_dg_src_log', []) or [])[n0:]
        if dgs:
            params = dict(params)
            params['_dg_srcs'] = dgs

    if at == 'siege_trim' and post is not None \
            and '_trim_by_nation' not in params:
        # the ENGINE names the excess it returned to reinforcements:
        # the per-(nation, kind) rows that LEFT the garrison between
        # pre and post (the menu's k alone is nation-blind)
        from board import sanitize
        pc = getattr(st, 'pending_combat', None) or {}
        region = sanitize(str(pc.get('trim_region')
                              or pc.get('defender_region') or ''))
        owner = emitter.canon_side(pc.get('defender_side') or side_name)

        def _rows(state):
            out = {}
            for k, army in state.armies.items():
                if sanitize(str(k)) != region:
                    continue
                for u in army.units:
                    nat = str(getattr(u.nation, 'value', u.nation))
                    if emitter.canon_side(
                            'sh' if emitter._side_of_nation(nat)
                            == 'shadow' else 'fp') != owner:
                        continue
                    key = '%s:%s' % (nat.lower(), str(getattr(
                        u.kind, 'value', u.kind)).lower())
                    out[key] = out.get(key, 0) + 1
            return out
        r0, r1 = _rows(st), _rows(post)
        gone = {k: n - r1.get(k, 0) for k, n in r0.items()
                if n - r1.get(k, 0) > 0}
        if not gone:
            return _hold('siege_trim: the post state removed no %s '
                         'unit from %r' % (owner, region))
        params = dict(params)
        params['_trim_by_nation'] = gone

    if at == 'siege_press':
        try:
            ops = emitter.plan(at, params, view, nick, side_name,
                               die_idx=None, rng=None)
        except NotImplementedError as e:
            return _hold('no planner for %r: %s' % (at, e))
        except LookupError as e:
            return _hold('%s: boards cannot supply the pieces: %s'
                         % (at, e))
        em = _ops_to_emission(view, nick, at, ops)
        # the press announce disambiguates the swap
        em.chat.insert(0, ('press', 'press'))
        return em

    # ---- everything else rides the planner table ---------------------
    plan_at = at
    if at == 'hunt_sacrifice_random':
        # K7: the victim is the CLIENT's own random draw (randomcomp
        # funnel) - live_server defers this render until the apply
        # succeeded; the POST state names the companion that LEFT the
        # fellowship and the physical grammar is the guide-sacrifice
        # one (death drag to FP Casualties, or the ALIVE separation
        # onto the map - USER ruling 2026-08-16).
        if post is None:
            return _hold('hunt_sacrifice_random needs the post state '
                         '(the client draw names the victim)')
        from game_state import Companion
        from board import sanitize

        def _names(state):
            fs = getattr(state, 'fellowship', None)
            return set(str(getattr(c, 'name', c))
                       for c in (getattr(fs, 'companions', ()) or ()))
        gone = sorted(_names(st) - _names(post))
        if len(gone) != 1:
            return _hold('hunt_sacrifice_random: cannot name the victim '
                         'from the post state (left=%r)' % (gone,))
        params = dict(params)
        params['companion'] = gone[0]
        for k, army in post.armies.items():
            for ld in (getattr(army, 'leaders', []) or []):
                if isinstance(ld, Companion) \
                        and str(ld.name) == gone[0]:
                    params['to_region'] = sanitize(str(k))
                    break
        plan_at = 'hunt_sacrifice_guide'
    try:
        ops = emitter.plan(plan_at, params, view, nick, side_name,
                           die_idx=None, rng=None)
    except NotImplementedError as e:
        return _hold('no planner for decision %r (kind %r): %s'
                     % (at, kind, e))
    except LookupError as e:
        return _hold('%s: boards cannot supply the pieces: %s'
                     % (at, e))
    em40 = _ops_to_emission(view, nick, at, ops)
    if not em40.hold:
        try:
            em40.lines += _forced_guide_lines(view, nick, st, post)
        except LookupError as e:
            return _hold('%s forced Guide: %s' % (at, e))
    try:
        _ch40 = (a.params or {}).get('choices') or {}
        if str(at) == 'card_option' and isinstance(_ch40, dict) \
                and _ch40.get('action') == 'discard_table_card' \
                and not em40.hold:
            # BL-40: the AI discarding its OWN table card to cancel an
            # effect (Wizard's Staff eating a hunt hit) must be SAID,
            # or it reads as a broken hunt
            from .chat import table_discard_cancel_note
            em40.chat = list(em40.chat or ()) + [
                ('prompt', table_discard_cancel_note())]
    except Exception:
        pass
    return em40


# --------------------------------------------------------------------
# combat-card reveal (K1 step 2): ONE directive reveals every chosen
# card, both sides at once (the client's own F2 handler); the reveal
# lines '<auto> moves <TITLE> from <Side> Hand (chosen) to Current
# <X> Card.' speak for themselves - no chat.
# --------------------------------------------------------------------
def render_combat_reveal(view, nick, side):
    return Emission(calls=['!CALL reveal_chosen'])


# --------------------------------------------------------------------
# round-resolution chat helpers (requests only; rule 7)
# --------------------------------------------------------------------
def render_hits(atk_hits, def_hits, attacker_is_human):
    """Per-round hit announcement request in the sided phrasing
    ('N hit(s) for you, M hit(s) for me'), kind 'hit_count' = chat.py's
    registered G2 form.  attacker_is_human is a bool: the caller
    (round_hits_chat) always knows the seat."""

    def _p(n):
        return '%d hit%s' % (int(n), '' if int(n) == 1 else 's')

    if attacker_is_human:
        you, me = atk_hits, def_hits
    else:
        you, me = def_hits, atk_hits
    return Emission(chat=[('hit_count', '%s for you, %s for me'
                           % (_p(you), _p(me)))])


def round_hits_chat(post, human_side, pre=None):
    """G2 hit announcement for a combat round that resolved INSIDE the
    apply that produced `post` (WAVE-H wiring - nothing called
    render_hits).

    Detection is an ENGINE statement, not a diff: apply_action works on
    a fresh clone, clone_state does NOT carry `_last_battle_result`
    across, and game_logic.battle stamps it on the state it ran on - so
    the attribute EXISTS on `post` iff THIS apply fought a round.
    Returns [(kind, text)] chat requests ([] when no round fired)."""
    r = getattr(post, '_last_battle_result', None)
    if r is None:
        return []
    observed_rounds = getattr(r, '_bridge_ordinary_round_rolls', None)
    if observed_rounds is not None and not any(observed_rounds):
        # A live pre-combat wipe or retreat can finish the engine round before
        # normal dice. Its ordinary 0/0 totals would misleadingly deny the
        # already resolved additional attack. Actual rolled 0/0 stays visible.
        return []
    if getattr(r, 'round_paused', False):
        # A paused result's totals cover only completed rounds. Initiative-six
        # prevention has not produced this round's final casualty count yet.
        pending = getattr(r, 'round_pending', None) or {}
        if pending.get('effect') != 'heroic_death':
            return []
        resume = getattr(r, 'round_resume', None) or {}
        combat = getattr(resume.get('cursor'), 'combat', None)
        if combat is None:
            return []
        role = 'attacker' if pending.get('owner') == 'atk' else 'defender'
        enemy = 'defender' if role == 'attacker' else 'attacker'
        incoming = max(0, int(getattr(combat, enemy + '_hits', 0))
                       + int(getattr(combat, enemy + '_bonus_hits', 0))
                       - int(getattr(combat, role + '_hits_cancelled', 0)))
        from . import chat
        return [('prompt', chat.heroic_death_incoming_note(
            incoming, getattr(combat, role + '_side', None) is human_side))]
    call = getattr(post, '_last_round_call', None) or {}
    atk_side = call.get('atk_side')
    if atk_side is None and getattr(post, '_last_round_phase_call', False):
        # Resuming a saved cursor deliberately drops the old transient call.
        # The enclosing pre-commit combat still authoritatively names its side.
        prior_combat = getattr(pre, 'pending_combat', None) or {}
        atk_side = prior_combat.get('attacker_side')
    if atk_side is None or human_side is None:
        return []

    def _tot60(prefix):
        # BL-60: one answerer - the announce totals what the casualty
        # stage consumes (PRE-COMBAT hits included); absent fields
        # cost nothing
        return (int(getattr(r, prefix + '_total_hits', 0) or 0)
                + int(getattr(r, prefix + '_pre_combat_hits', 0)
                      or 0))
    em = render_hits(_tot60('attacker'), _tot60('defender'),
                     attacker_is_human=(atk_side is human_side))
    return list(em.chat)


# ====================================================================
# forced-allocation rows + settlement-capture markers.  Both are
# ENGINE EVENTS a committed apply resolved that NO gesture or
# decision-planner rendered:
#   * forced casualties: the engine auto-allocates a side's hits when
#     the allocation is no choice (unemitted, the client board would
#     keep dead units alive);
#   * settlement captures: enemy_controlled/VP move engine-side with
#     no marker lines (the client's control/VP markers would drift).
# Rendering follows emission_card's SANCTIONED bounded pattern: the
# delta between the driver's own pre/post states around ONE apply,
# Committed endpoint rows, preserving actual identity and physical supply.
# ====================================================================
def _row_side(nation_or_tkey):
    n = str(nation_or_tkey).split(':', 1)[0]
    return emitter._side_of_nation(n)


def removal_rows(pre, post, side_tag):
    """Side-filtered army-row deltas of ONE apply, in the bounded row
    space (per-(nation,kind) unit rows; leaders refined into nazgul /
    per-nation plain-leader rows; named figures as identity rows).

    side_tag: 'fp' | 'shadow' — only that side's rows are returned.
    Returns dict:
      units: {tkey: {'minus': [(region, n)], 'plus': [(region, n)]}}
      naz:   {'minus': [(region, n)], 'plus': [(region, n)]}  (shadow)
      named: {'leaves': {name: [region]}, 'joins': {name: [region]}}
      cas:   {tkey: n} — the engine's OWN fp_casualties increments
      unexplained: [(region, leader_delta)] rows that do not decompose
    """
    from . import emission_card as emc
    from . import client_board
    delta = client_board.board_delta(pre, post)
    out = {'units': {}, 'naz': {'minus': [], 'plus': []},
           'named': {'leaves': {}, 'joins': {}},
           'cas': {}, 'unexplained': []}
    if not delta:
        return out
    cas_inc = emc._cas_increments(pre, post)
    if side_tag == 'fp':
        # The casualty box is a stock, not a count of deaths in this apply.
        # An Elite reduction may borrow a dead Regular and automatic casualty
        # follow-through then kill that same replacement (net box delta zero).
        # Add only the engine's new, explicit casualty-box withdrawals so the
        # forced-row renderer can distinguish that genuine loss from a return.
        before_sources = list(getattr(pre, '_dg_src_log', None) or [])
        after_sources = list(getattr(post, '_dg_src_log', None) or [])
        if after_sources[:len(before_sources)] != before_sources:
            raise ValueError('Downgrade source log is not an append-only continuation')
        sources = []
        for nation, source in after_sources[len(before_sources):]:
            key = '%s:regular' % str(getattr(nation, 'value', nation)).lower()
            if source == 'cas' and _row_side(key) == 'fp':
                cas_inc[key] = cas_inc.get(key, 0) + 1
            if source in ('cas', 'pool') and _row_side(key) == 'fp':
                sources.append((key.rsplit(':', 1)[0], source))
        out['cas'] = dict(cas_inc)
        if sources:
            out['downgrade_sources'] = sources
    named = delta.get('__named__', ()) or ()
    named_delta = {}
    for region, before, after in named:
        b = {emc._named_identity(name) for name in before}
        a2 = {emc._named_identity(name) for name in after}
        named_delta[region] = named_delta.get(region, 0) \
            + len(after) - len(before)
        for nm in b - a2:
            pid, pside = emc._named_pid_side(nm)
            if (pside or '').upper().startswith(
                    'S' if side_tag == 'shadow' else 'F'):
                out['named']['leaves'].setdefault(nm, []).append(region)
        for nm in a2 - b:
            pid, pside = emc._named_pid_side(nm)
            if (pside or '').upper().startswith(
                    'S' if side_tag == 'shadow' else 'F'):
                out['named']['joins'].setdefault(nm, []).append(region)
    # FLA-3 (par5 g3#6): ShadowMinion figures are INVISIBLE in
    # __named__ (the projector's type check predates the class name),
    # so a minion relocation vanished from the row space entirely -
    # no demand, no render, no hold (the Return of the Witch-king
    # divergence).  Derive their identity rows directly from the
    # states; _leader_comp now excludes them so the leader-delta
    # decomposition stays consistent.
    if side_tag == 'shadow':
        from board import sanitize as _snMi

        def _minion_regions(st):
            out9 = {}
            for k9, a9 in st.armies.items():
                for l9 in (getattr(a9, 'leaders', None) or ()):
                    if type(l9).__name__ == 'ShadowMinion':
                        out9.setdefault(
                            str(getattr(l9, 'name', '')),
                            []).append(_snMi(str(k9)))
            return out9
        m0 = _minion_regions(pre)
        m1 = _minion_regions(post)
        for nm in sorted(set(m0) | set(m1)):
            r0, r1 = set(m0.get(nm, ())), set(m1.get(nm, ()))
            for region in sorted(r0 - r1):
                out['named']['leaves'].setdefault(nm,
                                                  []).append(region)
                named_delta[region] = named_delta.get(region, 0) - 1
            for region in sorted(r1 - r0):
                out['named']['joins'].setdefault(nm,
                                                 []).append(region)
                named_delta[region] = named_delta.get(region, 0) + 1
    for region in sorted({k for k in delta if not str(k).startswith('__')}
                         | set(named_delta)):
        ud, ld = delta.get(region, ((), 0))
        for tkey, n in ud:
            if _row_side(tkey) != side_tag:
                continue
            slot = out['units'].setdefault(str(tkey),
                                           {'minus': [], 'plus': []})
            if n > 0:
                slot['plus'].append((region, int(n)))
            else:
                slot['minus'].append((region, -int(n)))
        if ld or named_delta.get(region, 0):
            # Named arrivals can cancel generic losses in the aggregate
            # leader count (Dead Men companions replacing killed Nazgul).
            # Decompose that zero net change before emitting the aftermath.
            naz0, pl0 = emc._leader_comp(pre, region)
            naz1, pl1 = emc._leader_comp(post, region)
            naz_d = naz1 - naz0
            plain_d, explained = {}, naz_d + named_delta.get(region, 0)
            for nv in set(pl0) | set(pl1):
                d = pl1.get(nv, 0) - pl0.get(nv, 0)
                if d:
                    plain_d[nv] = d
                    explained += d
            if explained != ld:
                out['unexplained'].append((region, ld))
                continue
            if side_tag == 'shadow':
                if naz_d > 0:
                    out['naz']['plus'].append((region, naz_d))
                elif naz_d < 0:
                    out['naz']['minus'].append((region, -naz_d))
            for nv, d in sorted(plain_d.items()):
                if emitter._side_of_nation(nv) != side_tag:
                    continue
                slot = out['units'].setdefault(
                    '%s:leader' % nv, {'minus': [], 'plus': []})
                if d > 0:
                    slot['plus'].append((region, d))
                else:
                    slot['minus'].append((region, -d))
    return out


def rows_empty(rows):
    return not (any(s['minus'] or s['plus']
                    for s in rows['units'].values())
                or rows['naz']['minus'] or rows['naz']['plus']
                or rows['named']['leaves'] or rows['named']['joins']
                or rows['unexplained'])


def expressed_rows(view, ops, side_tag):
    """The removal/arrival rows a rendered op list ALREADY expressed,
    in removal_rows' space — subtracted before the forced sweep so a
    planner-rendered allocation (casualty kills, downgrade swaps,
    retreats) is never re-rendered.  Returns (rows, decode_failed):
    decode_failed=True means a dead/moved pid could not be typed — the
    caller must then HOLD the sweep rather than risk a double kill."""
    import log_replay as _lr
    from .emission_card import _named_identity
    from board import sanitize
    rows = {'units': {}, 'naz': {'minus': [], 'plus': []},
            'named': {'leaves': {}, 'joins': {}},
            'cas': {}, 'unexplained': []}
    failed = [False]

    def _classify(pid):
        try:
            nat, ut = emitter._unit_type(pid, view, side_hint=side_tag)
        except Exception:
            failed[0] = True
            return None
        if ut is None:
            failed[0] = True
            return None
        return nat, str(ut).lower()

    def _reg_of(zone_code):
        name = _lr.decode_location(str(zone_code)) or ''
        rk = sanitize(str(name))
        import board as bd
        if rk.endswith(' stronghold'):
            rk = rk[:-len(' stronghold')]
        return rk if rk in bd.regions else None

    def _add(bucket, key, region, n=1):
        slot = rows['units'].setdefault(key, {'minus': [], 'plus': []})
        slot[bucket].append((region, n))

    for op, f in (ops or ()):
        pids = list(f.get('pieces') or ())
        if not pids and f.get('piece'):
            pids = [f['piece']]
        pids = [str(p) for p in pids if str(p) not in _DIE_PIDS]
        if not pids:
            continue
        if op in _RETRACK_DEAD:
            reg = _reg_of(f.get('frm') or f.get('to') or '')
            for p in pids:
                c = _classify(p)
                if c is None:
                    continue
                nat, ut = c
                if ut == 'nazgul':
                    rows['naz']['minus'].append((reg, 1))
                elif ut in ('companion', 'minion'):
                    rows['named']['leaves'].setdefault(
                        _named_identity(str(nat)), []).append(reg)
                elif ut in ('regular', 'elite', 'leader'):
                    if emitter._side_of_nation(nat) == side_tag:
                        key = '%s:%s' % (str(getattr(nat, 'value', nat)).lower(), ut)
                        _add('minus', key, reg)
                        if side_tag == 'fp' and reg is not None:
                            rows['cas'][key] = rows['cas'].get(key, 0) + 1
        elif op in _RETRACK_TO:
            frm_r = _reg_of(f.get('frm') or '')
            to_r = _reg_of(f.get('to') or '')
            for p in pids:
                c = _classify(p)
                if c is None:
                    continue
                nat, ut = c
                if ut == 'nazgul':
                    if frm_r:
                        rows['naz']['minus'].append((frm_r, 1))
                    if to_r:
                        rows['naz']['plus'].append((to_r, 1))
                elif ut in ('companion', 'minion'):
                    if frm_r:
                        rows['named']['leaves'].setdefault(
                            _named_identity(str(nat)), []).append(frm_r)
                    if to_r:
                        rows['named']['joins'].setdefault(
                            _named_identity(str(nat)), []).append(to_r)
                elif ut in ('regular', 'elite', 'leader'):
                    if emitter._side_of_nation(nat) == side_tag:
                        key = '%s:%s' % (
                            str(getattr(nat, 'value', nat)).lower(),
                            ut)
                        if frm_r:
                            _add('minus', key, frm_r)
                            if side_tag == 'fp' and f.get('to') == emitter.FP_DEAD:
                                rows['cas'][key] = rows['cas'].get(key, 0) + 1
                        if to_r:
                            _add('plus', key, to_r)
                            if side_tag == 'fp' and ut == 'regular' and op == 'swap_remove':
                                source = {emitter.FP_DEAD: 'cas', emitter.FP_POOL: 'pool'}.get(f.get('frm'))
                                if source is not None:
                                    rows.setdefault('downgrade_sources', []).append(
                                        (key.rsplit(':', 1)[0], source))
    return rows, failed[0]


def _sub_pairs(base, minus_counts):
    """Subtract Counter-like {region: n} from a [(region, n)] list."""
    out = []
    left = dict(minus_counts or {})
    for reg, n in base:
        take = min(n, left.get(reg, 0))
        if take:
            left[reg] = left.get(reg, 0) - take
        if n - take > 0:
            out.append((reg, n - take))
    return out


def subtract_rows(rows, sub):
    """rows := rows - sub (both in removal_rows space), in place-safe
    copy.  Named subtraction is by (name, one region occurrence)."""
    out = {'units': {}, 'naz': {'minus': [], 'plus': []},
           'named': {'leaves': {}, 'joins': {}},
           'cas': dict(rows.get('cas') or {}),
           'unexplained': list(rows.get('unexplained') or [])}
    # Unit rows are signed changes. A selected Regular removal followed by
    # an automatic Elite reduction has net Regular delta zero, but after
    # rendering that removal the residual must still place one Regular.
    # Clipping subtraction within separate +/- lists silently lost it.
    from collections import Counter
    # Death evidence is consumed with the already rendered loss. Leaving the
    # original total here either rejects a valid residual loss (Great Host)
    # or credits that same death twice and permits an unproved second kill.
    out['cas'] = {key: int(rows.get('cas', {}).get(key, 0))
                 - int((sub or {}).get('cas', {}).get(key, 0))
                 for key in set(rows.get('cas', {})) | set((sub or {}).get('cas', {}))}
    owed_sources = [tuple(source) for source in rows.get('downgrade_sources', ())]
    for source in (sub or {}).get('downgrade_sources', ()):
        source = tuple(source)
        if source in owed_sources:
            owed_sources.remove(source)
    if 'downgrade_sources' in rows:
        out['downgrade_sources'] = owed_sources
    for tkey in sorted(set(rows['units']) | set((sub or {}).get('units', {}))):
        delta = Counter()
        for source, multiplier in ((rows, 1), (sub or {}, -1)):
            slot = source.get('units', {}).get(tkey, {})
            for direction, sign in (('plus', 1), ('minus', -1)):
                for region, count in slot.get(direction, []):
                    delta[region] += multiplier * sign * count
        minus = [(region, -count) for region, count in sorted(delta.items()) if count < 0]
        plus = [(region, count) for region, count in sorted(delta.items()) if count > 0]
        if minus or plus:
            out['units'][tkey] = {'minus': minus, 'plus': plus}
    for b in ('minus', 'plus'):
        d = {}
        for r, n in (sub['naz'][b] if sub else ()):
            d[r] = d.get(r, 0) + n
        out['naz'][b] = _sub_pairs(rows['naz'][b], d)
    for b in ('leaves', 'joins'):
        subd = dict((sub['named'][b] if sub else {}) or {})
        for nm, regs in rows['named'][b].items():
            left = list(regs)
            for r in (subd.get(nm) or []):
                if r in left:
                    left.remove(r)
                elif None in left:
                    left.remove(None)
            if left:
                out['named'][b][nm] = left
    return out


def _companion_out_of_play(post, pid):
    """Known Companion absent from every live or deferred engine location."""
    if post is None:
        return False
    import log_replay
    if str(pid) not in log_replay.COMPANIONS:
        return False
    names = set(log_replay.names_for_pid(str(pid)) or ())
    if not names:
        return False
    fellowship = post.fellowship
    living = {str(c.name) for c in fellowship.companions}
    guide = getattr(fellowship, 'guide', None)
    if guide is not None:
        living.add(str(guide.name))
    for army in post.armies.values():
        living.update(str(getattr(figure, 'name', '')) for figure in army.leaders)
    living.update(str(getattr(figure, 'name', figure))
                  for figure in (getattr(post, '_hobbit_place_queue', None) or ()))
    pending = getattr(post, 'pending_hunt', None) or {}
    if pending.get('phase') == 'hobbit_place':
        living.add(str(pending.get('companion', '')))
        living.add(str(getattr(pending.get('companion_obj'), 'name', '')))
    return not names.intersection(living)


def render_forced_rows(view, nick, side, rows, pre, label='', post=None):
    """Render residual side-owned rows as that side's own physical
    lines (the bounded emission_card row rules, applied to the
    forced-allocation class).  Returns (Emission, ledger_rows):
    ledger_rows = the REMOVAL fingerprints rendered (echo-ledger
    entries).  Any row outside the literal rules => HOLD naming it."""
    from . import emission_card as emc
    from .board_transfers import balanced_transfers, physical_move_ops
    from board import sanitize
    side_name = emitter.canon_side(side)
    side_tag = 'shadow' if side_name == 'SHADOW' else 'fp'
    scratch = emc._Scratch(view, pre)
    holds, ops, ledger = [], [], []

    def _pick(region, unit_kind, n, nation):
        kw = {'n_regular': 0, 'n_elite': 0, 'n_leader': 0,
              'n_nazgul': 0}
        kw['n_' + unit_kind] = n
        return emitter.pick_units(scratch, sanitize(str(region)),
                                  side_tag, nation=nation,
                                  side_hint=side_tag, **kw)

    def _kill(region, n, nation, unit_kind, tname):
        try:
            pids = _pick(region, unit_kind, n, nation)
            op = ('hand_exit_b', {
                'nick': nick, 'pieces': pids,
                'frm': emitter.garrison_arrival_zone(
                    pre, sanitize(str(region)), side_tag)})
            ops.append(op)
            emc._retrack_scratch(scratch, [op])
            ledger.append({'kind': 'unit', 'side': side_tag,
                           'region': sanitize(str(region)),
                           'tkey': tname, 'n': n})
        except LookupError as e:
            holds.append('%s: cannot supply %d casualt%s at %s: %s'
                         % (tname, n, 'y' if n == 1 else 'ies',
                            region, e))

    def _recruit(region, n, nation, unit_kind, tname):
        params = {'nation': nation, 'region': sanitize(str(region))}
        if emitter.garrison_side_of(pre, params['region']) \
                == side_name:
            params['_besieged_garrison'] = True
        # The engine records the exact replacement source for FP reductions.
        # Consume each residual proof once across arrivals; ordinary recruits
        # keep their existing pool policy. Never substitute another source.
        if not hasattr(_recruit, 'sources'):
            _recruit.sources = list(rows.get('downgrade_sources') or [])
        for _ in range(n):
            try:
                source_index = next((i for i, (nat, source) in
                    enumerate(_recruit.sources)
                    if side_tag == 'fp' and unit_kind == 'regular'
                    and str(nat).lower() == str(nation).lower()), None)
                if source_index is None:
                    op = emitter.plan_muster(scratch, nick, side_name,
                                             params, unit_kind)
                else:
                    _, source = _recruit.sources.pop(source_index)
                    if source not in ('cas', 'pool'):
                        raise LookupError('invalid recorded downgrade source %r' % (source,))
                    source_zone = emitter.FP_DEAD if source == 'cas' else emitter.FP_POOL
                    candidates = []
                    for pid in sorted(scratch.piece_region):
                        nat, kind = emitter._unit_type(pid, scratch, side_hint='fp')
                        if kind != 'regular' or str(getattr(nat, 'value', nat)).lower() != str(nation).lower():
                            continue
                        try:
                            if emitter._piece_zone_code(scratch, pid) == source_zone:
                                candidates.append(pid)
                        except LookupError:
                            pass
                    if not candidates:
                        raise LookupError('recorded %s source has no observed %s Regular' % (source, nation))
                    op = ('move', {'nick': nick, 'pieces': candidates[:1],
                        'frm': source_zone, 'to': emitter.garrison_arrival_zone(
                            pre, params['region'], side_tag)})
                ops.append(op)
                emc._retrack_scratch(scratch, [op])
            except LookupError as e:
                holds.append('%s: pool cannot supply a %s at %s: %s'
                             % (tname, unit_kind, region, e))
                return

    def _move(frm, to, n, nation, unit_kind, tname):
        try:
            pids = _pick(frm, unit_kind, n, nation)
            transfers = physical_move_ops(scratch, nick, pids,
                emitter._garrison_dest(scratch, sanitize(str(to)), side_name))
            ops.extend(transfers)
            emc._retrack_scratch(scratch, transfers)
        except LookupError as e:
            holds.append('%s: cannot supply the %d-piece move '
                         '%s -> %s: %s' % (tname, n, frm, to, e))

    for tkey in sorted(rows['units']):
        M = rows['units'][tkey]['minus']
        P = rows['units'][tkey]['plus']
        nation, unit_kind = tkey.rsplit(':', 1)
        cas = int((rows.get('cas') or {}).get(tkey, 0))
        tm = sum(n for _, n in M)
        if not M:
            for r, n in P:
                _recruit(r, n, nation, unit_kind, tkey)
        elif cas == 0 and balanced_transfers(M, P) is not None:
            for source, target, count in balanced_transfers(M, P):
                _move(source, target, count, nation, unit_kind, tkey)
        elif side_tag == 'fp' and cas == tm:
            for r, n in M:
                _kill(r, n, nation, unit_kind, tkey)
            for r, n in P:
                _recruit(r, n, nation, unit_kind, tkey)
        elif side_tag == 'shadow':
            # SA dead RECYCLE (p.30): casualty and return converge on
            # one physical kill drag; pluses are downgrade recruits
            for r, n in M:
                _kill(r, n, nation, unit_kind, tkey)
            for r, n in P:
                _recruit(r, n, nation, unit_kind, tkey)
        else:
            holds.append('%s: FP removal not covered by the casualty '
                         'ledger (minus=%r plus=%r cas%+d) - casualty '
                         'or return, cannot tell' % (tkey, M, P, cas))
    M, P = rows['naz']['minus'], rows['naz']['plus']
    if M or P:
        if not M:
            for r, n in P:
                _recruit(r, n, None, 'leader', 'nazgul')
        elif balanced_transfers(M, P) is not None:
            for source, target, count in balanced_transfers(M, P):
                _move(source, target, count, None, 'nazgul', 'nazgul')
        elif not P:
            for r, n in M:
                try:
                    pids = emitter.pick_units(
                        scratch, sanitize(str(r)), 'shadow',
                        n_nazgul=n, side_hint='shadow')
                    op = ('hand_exit_b', {
                        'nick': nick, 'pieces': pids,
                        'frm': emitter.garrison_arrival_zone(
                            pre, sanitize(str(r)), 'shadow')})
                    ops.append(op)
                    emc._retrack_scratch(scratch, [op])
                    ledger.append({'kind': 'naz', 'side': 'shadow',
                                   'region': sanitize(str(r)),
                                   'tkey': 'nazgul', 'n': n})
                except LookupError as e:
                    holds.append('nazgul: cannot supply %d at %s: %s'
                                 % (n, r, e))
        else:
            holds.append('nazgul: AMBIGUOUS pairing (minus=%r '
                         'plus=%r)' % (M, P))
    for nm in sorted(set(rows['named']['leaves'])
                     | set(rows['named']['joins'])):
        ls = rows['named']['leaves'].get(nm, [])
        js = rows['named']['joins'].get(nm, [])
        pid, pside = emc._named_pid_side(nm)
        if pid is None:
            holds.append('named: unknown figure %r' % (nm,))
            continue
        if len(js) == 1 and len(ls) <= 1:
            frm = None
            try:
                frm = emitter._piece_zone_code(scratch, pid)
            except LookupError:
                frm = None
            if frm is None or frm == emitter.FP_DEAD:
                holds.append('named: no safe client source for %r'
                             % (nm,))
                continue
            op = ('move', {'nick': nick, 'pieces': [pid], 'frm': frm,
                           'to': emitter._garrison_dest(
                               scratch, sanitize(str(js[0])), pside)})
            ops.append(op)
            emc._retrack_scratch(scratch, [op])
        elif ls and not js and pside == 'FREE_PEOPLES' and _companion_out_of_play(post, pid):
            # An army Companion has no separate casualty-counter stock.
            # The committed post-state distinguishes elimination from a
            # return to the Fellowship or an outstanding Hobbit placement.
            try:
                frm = emitter._piece_zone_code(scratch, pid)
            except LookupError:
                frm = None
            if frm is None or frm == emitter.FP_DEAD:
                holds.append('named: no live client source for eliminated %r' % (nm,))
                continue
            op = ('move', {'nick': nick, 'pieces': [pid], 'frm': frm,
                           'to': emitter.FP_DEAD})
            ops.append(op)
            emc._retrack_scratch(scratch, [op])
            ledger.append({'kind': 'named', 'side': 'fp',
                           'region': sanitize(str(ls[0] or '')),
                           'tkey': nm, 'n': 1})
        elif ls and not js and pside == 'SHADOW':
            # a Shadow named figure eliminated in the apply: minions
            # go out of play via the same kill grammar (converged)
            try:
                frm = emitter.garrison_arrival_zone(
                    pre, sanitize(str(ls[0] or '')), 'shadow') \
                    if ls[0] else emitter._piece_zone_code(scratch,
                                                          pid)
            except LookupError:
                holds.append('named: no zone for %r' % (nm,))
                continue
            op = ('hand_exit_b', {'nick': nick, 'pieces': [pid],
                                  'frm': frm})
            ops.append(op)
            emc._retrack_scratch(scratch, [op])
            ledger.append({'kind': 'named', 'side': 'shadow',
                           'region': sanitize(str(ls[0] or '')),
                           'tkey': nm, 'n': 1})
        else:
            holds.append('named: %r ambiguous or FP-fatal '
                         '(leaves=%r joins=%r) - an FP named leave '
                         'is casualty-or-separation, cannot tell'
                         % (nm, ls, js))
    for region, ld in rows['unexplained']:
        holds.append('leader row %s %+d does not decompose' % (region,
                                                               ld))
    if holds:
        return _hold('forced %s rows (%s) unrenderable - HOLDING, '
                     'never guessing: %s'
                     % (side_name, label, ' | '.join(holds))), []
    em = _ops_to_emission(view, nick, 'forced_rows', ops)
    return em, ledger


# ---- settlement-capture markers --------------------------------------
# G4 control markers + the VP reconciler.  Engine truth:
# post.enemy_controlled sets + game_logic.calculate_victory_points(post).
# Ledger-reconciled per commit, idempotent; a marker the HUMAN already
# claimed (visible in the dump) is ADOPTED, never duplicated.
_VP_TRACK_ZONES = {0: 'Game.27', 1: 'Game.28', 2: 'Game.29',
                   3: 'Game.30', 4: 'Game.31', 5: 'Game.32',
                   6: 'Game.33', 7: 'Game.388', 8: 'Game.389',
                   9: 'Game.391', 10: 'Game.392'}
_CTRL_POOLS = {'fp': ('Game.2038', tuple(str(i)
                                         for i in range(405, 411))),
               'sh': ('Game.2040', tuple(str(i)
                                         for i in range(411, 421)))}


def render_capture_markers(view, nick, post, ctrl, vp):
    """Control-marker walks + VP drags for captures a committed apply
    resolved (the delta rows name the settlement + side — engine
    events, never diff-guessed pairing).  ctrl = {'fp': {region: pid},
    'sh': {...}} (the driver's ledger, mutated); vp = {'fp': n,
    'sh': n}.  Returns (Emission, notes): notes are LOUD rows the
    caller records (pool exhaustion, zone failures, adoptions)."""
    from board import sanitize
    from game_state import Side
    import game_logic as gl
    em = Emission()
    notes = []
    try:
        want = {'sh': {sanitize(str(k)) for k in
                       post.enemy_controlled.get(Side.SHADOW, set())},
                'fp': {sanitize(str(k)) for k in
                       post.enemy_controlled.get(Side.FREE_PEOPLES,
                                                 set())}}
    except Exception as e:
        return em, ['control-marker read failed: %r' % (e,)]
    for key in ('sh', 'fp'):
        pool_zone, pids = _CTRL_POOLS[key]
        have = ctrl[key]
        for reg in sorted(set(have) - want[key]):
            pid = have.pop(reg)
            try:
                zc = emitter.zone_code_for(reg)
            except LookupError:
                have[reg] = pid     # never strand un-emitted
                notes.append('control-return: no zone code for %r - '
                             'marker %s left in the ledger'
                             % (reg, pid))
                continue
            em.lines.append(render('move', {
                'nick': nick, 'pieces': [pid],
                'frm': zc, 'to': pool_zone}, local=False))
        for reg in sorted(want[key] - set(have)):
            # the HUMAN may have claimed it already (mouse handler,
            # client fact: ax_0.java) - adopt the dump's marker
            adopted = None
            for p in pids:
                z = sanitize(str(view.piece_region.get(p, '') or ''))
                if z == reg:
                    adopted = p
                    break
            if adopted is not None:
                have[reg] = adopted
                notes.append('control-adopt: %s marker %s already at '
                             '%s (human claim) - no line emitted'
                             % (key, adopted, reg))
                continue
            used = set(have.values())
            pid = next((p for p in pids if p not in used), None)
            if pid is None:
                notes.append('control-marker POOL EXHAUSTED (%s) for '
                             '%r - physical supply, nothing placed'
                             % (key, reg))
                continue
            try:
                zc = emitter.zone_code_for(reg)
            except LookupError:
                notes.append('control-place: no zone code for %r'
                             % (reg,))
                continue
            em.lines.append(render('move', {
                'nick': nick, 'pieces': [pid],
                'frm': pool_zone, 'to': zc}, local=False))
            have[reg] = pid
    try:
        vpe = gl.calculate_victory_points(post)
    except Exception as e:
        notes.append('vp-marker skipped: %r' % (e,))
        return em, notes
    for key, pid, cur in (('fp', '422', vpe.get('free_peoples', 0)),
                          ('sh', '423', vpe.get('shadow', 0))):
        cur = max(0, min(10, int(cur)))
        old = max(0, min(10, int(vp.get(key, 0))))
        if cur == old:
            continue
        source = _VP_TRACK_ZONES[old]
        if view.piece_region.get(pid):
            # A human or native capture handler may already have updated
            # the physical marker before this engine commit reaches us.
            # The ledger describes the prior score, not its current source.
            try:
                source = emitter._piece_zone_code(view, pid)
            except LookupError:
                source = None
            if source not in _VP_TRACK_ZONES.values():
                notes.append('vp-marker: %s marker %s has no known track '
                             'source; no move emitted' % (key, pid))
                continue
            if source == _VP_TRACK_ZONES[cur]:
                vp[key] = cur
                notes.append('vp-adopt: %s marker %s already at %d; '
                             'no move emitted' % (key, pid, cur))
                continue
        em.lines.append(render('move', {
            'nick': nick, 'pieces': [pid],
            'frm': source,
            'to': _VP_TRACK_ZONES[cur]}, local=False))
        vp[key] = cur
    return em, notes
