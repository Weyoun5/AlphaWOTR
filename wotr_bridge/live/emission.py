"""Emission: engine action -> client gesture lines for the thin driver.

DESIGN RULE 6 (REBUILD_PLAN.md, paid for by the phantom/mis-pair defect
family): emission renders ENGINE EVENTS -- the action the engine
applied, with its params -- NEVER a re-inference from state diffs.  The
old monolith's `_emit_board_diff` generative differ (ai_move_server.py
:8940-9759) is the disease and is NOT lifted.  When no planner can
render an action type this module returns a loud HOLD naming the type;
it never falls back to diffing.

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
    pre/post engine states around ONE decision, literal rows only,
    every ambiguity a loud HOLD -- see emission_card's docstring for
    why that is allowed while the generative differ was not.  Other
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

# donor :18150 -- die-carrying actions whose die goes to the HUNT BOX
# (via the planner's die_to_hunt / eye_alloc grammar), never to Used.
_DIE_TO_HUNT_ACTIONS = frozenset(('fellowship_move', 'eye_allocate'))

# actions whose FIRST die-carrying leg opens an engine multi-step
# (ai.py :9136 pending_army_move_die, :9909 pending_muster_die): the
# die kill is deferred until the step closes, so rendering them without
# `post` cannot answer "is the kill due" -- HOLD instead of guessing.
_MULTISTEP_TYPES = frozenset(('army_move', 'muster_recruit'))

# donor :15783-15785 -- the chain-open predicate's pending fields.
_CHAIN_PENDINGS = ('pending_combat', 'pending_hunt',
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
        # WAVE-I: the planner ops behind self.lines (attached by
        # _ops_to_emission) — the forced-casualty sweep subtracts the
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
# the refund predicate (donor :15929-15949, decoupled from self)
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
    10); ClientView does not carry that map yet (REPORTED gap), so this
    accepts either a `card_pid(cid)` callable or a `card_pid_map` dict
    on the view, both driver-supplied.  Returns None when unresolvable
    -- callers HOLD, they never guess a pid."""
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
                view.piece_region[p] = 'dead'
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
    """Bonus dice granted by this action join the side's dice pool
    (donor :13196-13232 minus the driver ledgers): one move line per
    NEW engine die index, piece id from the emitter's base tables."""
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
            out.append(render('move', {
                'nick': nick, 'pieces': [base[i]],
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
# engine-truth param threading (post-dependent members only; the
# parser-lag members of donor :16059-16206 die here because view.state
# IS the engine state)
# --------------------------------------------------------------------
def _threaded_params(view, side_name, action, post):
    a = action
    at = a.action_type
    params = a.params
    st = view.state

    if at in ('muster_recruit', 'muster_elite', 'transform_gandalf'):
        # SIEGE musters join the garrison INSIDE (donor :16161-16181;
        # asked through the ONE function, of the ENGINE's state).
        from board import sanitize
        reg = sanitize(str(params.get('region', '')))
        if emitter.garrison_side_of(st, reg) == \
                emitter.canon_side(side_name):
            params = dict(params)
            params['_besieged_garrison'] = True

    if at in ('army_move', 'victor_advance') and post is not None \
            and not params.get('moved_by_nation'):
        # WAVE-K K5 (wkg5 LIVE: 'units dimrill dale isengard/regular:
        # engine=0 client=1'): the ENGINE moves SPECIFIC units - the
        # exact per-(nation, kind) composition that LEFT the source
        # between pre and post (donor _moved_by_nation, ported) rides
        # the params; plan_army_move honours 'moved_by_nation' and a
        # nation-blind pick shipped an Isengard regular where the
        # engine moved a Sauron one (the donor's t131 class).
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
        # the ENGINE's true landing spot (donor :16123-16131); without
        # post the planner reads the ENGINE pending_hunt damage, which
        # is already the rules answer -- the old parser-lag walk-one-
        # step bug cannot recur.
        params = dict(params)
        params['corruption_target'] = int(post.fellowship.corruption)

    if at == 'fellowship_move' \
            and getattr(st.fellowship, 'mordor_entered', False):
        # Mordor track: the pawn advances only when the ENGINE's step
        # advanced (donor :16101-16114).  Needs post; callers HOLD when
        # it is absent (checked in render_action).
        params = dict(params)
        params['from_step'] = int(getattr(st.fellowship,
                                          'mordor_step', 0))
        if post is not None:
            params['mordor_post_step'] = int(
                getattr(post.fellowship, 'mordor_step', 0))

    if at == 'hunt_sacrifice_guide':
        # name the ENGINE's guide (donor :16059-16067) and, given post,
        # the engine's true ALIVE-SEPARATION placement region (donor
        # :16068-16088; USER ruling 2026-08-15).
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


# The three deliberately-killable play-on-table cards, by CLIENT piece
# id.  Exactly the set log_replay._TABLE_KILL_PIDS parses for ruling
# 1.149 (106 / 145 / 202); keyed by NAME here because the engine action
# carries card_name, not a pid.  These pieces exist once per game, so
# the ids are fixed -- no dump lookup and no fungible pick.
# (REMOVED 2026-08-20) a static card-pid table used to live here.
# It claimed 145 was "The Power of Tom Bombadil"; this client
# renders 145 as "Kindred of Glorfindel", so the AI dragged that
# out of the FP Strategy Deck and killed it in front of the USER.
# Piece ids are the CLIENT's to state, never ours to hardcode:
# the target now comes from driver.table_seats (the slot we
# watched the card go into) resolved against the client dump.
# The actor's own Current-Card zone (log_replay:158 -- 2036 renders as
# 'Current SA Card', 2037 as 'Current FP Card').
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
        # kind 'pass' (WAVE-H): the old ('nc', 'pass') pair failed the
        # 'nc' kind's ^nc$ validator at the ChatOut exit - the FIRST AI
        # pass in any live game raised ChatPolicyError
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

    # ---- play_event: staged prologue (donor :15874-15900) -----------
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
        # (USER invariant 'die -> play -> resolve -> kill', donor
        # :15760-15829) and the resolution rides the !CALL -- the
        # driver flushes render_die_kill when the chain closes.
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
        # the battle die is killed at battle END (donor :8558-8570);
        # political activation/advance is ENGINE-side state -- each
        # subsequent chain decision renders separately.
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
        # G19 (donor :715-718): discards ride the 296 kill-in-hand
        # grammar (piece, hand).
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
    # the first leg (donor :16210-16216)
    if pre_pending:
        ops = [(op, f) for op, f in ops
               if not str(op).startswith('die_spend')]

    em = _ops_to_emission(view, nick, at, ops)

    # the LEARNED zero-eye choice is visible only in chat (D3: '0e';
    # absence-is-the-choice)
    if at == 'eye_allocate' and int(params.get('count') or 0) == 0:
        em.chat.append(('zero_eyes', '0e'))

    # ---- die kill to the Used box (donor :16363-16375) ---------------
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
        except LookupError as e:
            return _hold('%s post bookkeeping: %s' % (at, e))
    return em


# --------------------------------------------------------------------
# render_decision: one pending-chain DECISION -> Emission
# --------------------------------------------------------------------
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

    # ---- the AI's own hand-limit discard (cbt8: the DECISION path
    # lacked the branch render_action already had - the eye step
    # halted on 'no L2 planner for discard_card' every tick) ----------
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
        # the human's pending commit on a one-client board - measured:
        # the reveal then showed the AI's card only and the round
        # parked for ever.
        return Emission(lines=[render('combat_card_pick',
                                      {'nick': nick, 'piece': str(pid)},
                                      local=False)])

    # combat protocol chats: each text under ITS OWN registered kind
    # (WAVE-H: 'stop'/'press'/'gandalf shines'/'ffN' all rode the 'nc'
    # or unregistered kinds and the ChatOut validators refused them -
    # the two-vocabularies seam defect, third specimen)
    if at == 'siege_assault':
        # WAVE-K K8 (wkg6 LIVE WEDGE: the AI's own siege ASSAULT of a
        # besieged stronghold): the attacker_siege_decision -> assault
        # has NO board gesture of its own - the region was already
        # declared by the army_attack that OPENED the pending (corpus
        # 2023-01-02: after the withdrawal into Lorien Stronghold the
        # attacker chats the region, then the assault round rolls); the
        # engine runs a combat round here, whose d6 rolls the funnel
        # asks and whose hits/casualties get their own producers.  An
        # empty emission lets the round proceed instead of DriverHalt
        # ('no L2 planner for siege_assault').  siege_continue (maintain,
        # no round) likewise emits nothing new here.
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
        # WAVE-I (WAVE-H blocker 3): the non-dread magnitude classes
        # (Relentless Assault self-hits, Onslaught sacrifice, Heroic
        # Death leader/companion forfeit) no longer wedge.  The DECISION
        # itself has NO physical gesture: the engine only RECORDS the
        # count here (combat.py choices['*_effect_count']) and applies
        # the sacrifice later, inside the round-resolution apply — the
        # corpus confirms the chat-first shape (2023-01-13 Relentless
        # Assault: the owner says 'I wont inflict a hit', no drag; the
        # sacrifice kills render inside the round: 'kills Isengard
        # Regular in Orthanc', 2023-01-02).  So this decision emits the
        # G3 magnitude statement ONLY (a decision of 0 must still be
        # stated — chat.py kind 'magnitude'); the physical kill lines
        # are rendered when they materialize, by the post-commit
        # forced-casualty sweep (render_forced_rows) on whichever apply
        # fights the round.  Dread and Despair keeps its dedicated ffN
        # grammar (leadership forfeit has no board form at all).
        cid = str(a.params.get('card_id', ''))
        cnt = int(a.params.get('count', 0) or 0)
        # the effect CLASS comes from the engine's own classifier over
        # the card's printed COMBAT name (combat.combat_effect_kind) —
        # 'dread' never appears in a live card_id ('Grond, Hammer of
        # the Underworld' CARRIES Dread and Despair), so the donor's
        # substring test would have mislabeled every live dread choice
        eff = None
        try:
            from cards import get_card
            from combat import combat_effect_kind
            card = get_card(cid)
            eff = combat_effect_kind(
                getattr(card, 'combat_name', '') or '')
        except Exception:
            eff = None
        if eff == 'dread_despair' or 'dread' in cid.lower():
            return Emission(chat=[(
                'ff', 'ff%d' % int(a.params.get('count', 1) or 1))])
        title = parser_card_name(cid).strip().lower() or 'combat card'
        return Emission(chat=[('magnitude',
                               '%s: i choose %d' % (title[:40], cnt))])

    # ---- card_option: the SANCTIONED delta-scoped row renderer ------
    # (bounded exception to rule 6, coordinator 2026-08-19; provenance
    # + row rules in emission_card's docstring.  card_followup KIND
    # decisions carry real army_move/army_attack/followup_done actions
    # and ride their own planners below.)
    if at == 'card_option':
        from . import emission_card
        return emission_card.render_card_effect(view, nick, side, kind,
                                                a, pre, post)
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
        # the press announce disambiguates the swap (donor :8610-8614)
        em.chat.insert(0, ('press', 'press'))
        return em

    # ---- everything else rides the planner table ---------------------
    plan_at = at
    if at == 'hunt_sacrifice_random':
        # WAVE-K K7 (wkf2 LIVE WEDGE): the victim is the CLIENT's own
        # random draw (randomcomp funnel) - live_server defers this
        # render until the apply succeeded; the POST state names the
        # companion that LEFT the fellowship and the physical grammar
        # is the guide-sacrifice one (death drag to FP Casualties, or
        # the ALIVE separation onto the map - USER ruling 2026-08-16).
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
    return _ops_to_emission(view, nick, at, ops)


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


def round_hits_chat(post, human_side):
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
    call = getattr(post, '_last_round_call', None) or {}
    atk_side = call.get('atk_side')
    if atk_side is None or human_side is None:
        return []
    em = render_hits(int(getattr(r, 'attacker_total_hits', 0) or 0),
                     int(getattr(r, 'defender_total_hits', 0) or 0),
                     attacker_is_human=(atk_side is human_side))
    return list(em.chat)


# ====================================================================
# WAVE-I: forced-allocation rows + settlement-capture markers
# (WAVE-H blockers 1 and 2).  Both are ENGINE EVENTS a committed apply
# resolved that NO gesture or decision-planner rendered:
#   * forced casualties: the engine auto-allocates a side's hits when
#     the allocation is no choice — those unit removals were UNEMITTED
#     (the cbt10 class) and the client board kept dead units alive;
#   * settlement captures: enemy_controlled/VP move engine-side with
#     no marker lines, drifting the client's control/VP markers.
# Rendering follows emission_card's SANCTIONED bounded pattern: the
# delta between the driver's own pre/post states around ONE apply,
# literal row rules only, every ambiguity a loud HOLD.
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
    from board import sanitize
    delta = None
    from . import client_board
    delta = client_board.board_delta(pre, post)
    out = {'units': {}, 'naz': {'minus': [], 'plus': []},
           'named': {'leaves': {}, 'joins': {}},
           'cas': {}, 'unexplained': []}
    if not delta:
        return out
    cas_inc = emc._cas_increments(pre, post)
    if side_tag == 'fp':
        out['cas'] = dict(cas_inc)
    named = delta.get('__named__', ()) or ()
    named_delta = {}
    for region, before, after in named:
        b, a2 = set(before), set(after)
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
    for region in sorted(k for k in delta
                         if not str(k).startswith('__')):
        ud, ld = delta[region]
        for tkey, n in ud:
            if _row_side(tkey) != side_tag:
                continue
            slot = out['units'].setdefault(str(tkey),
                                           {'minus': [], 'plus': []})
            if n > 0:
                slot['plus'].append((region, int(n)))
            else:
                slot['minus'].append((region, -int(n)))
        if ld:
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
                        str(nat), []).append(reg)
                elif ut in ('regular', 'elite', 'leader'):
                    if emitter._side_of_nation(nat) == side_tag:
                        _add('minus', '%s:%s' % (
                            str(getattr(nat, 'value', nat)).lower(),
                            ut), reg)
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
                            str(nat), []).append(frm_r)
                    if to_r:
                        rows['named']['joins'].setdefault(
                            str(nat), []).append(to_r)
                elif ut in ('regular', 'elite', 'leader'):
                    if emitter._side_of_nation(nat) == side_tag:
                        key = '%s:%s' % (
                            str(getattr(nat, 'value', nat)).lower(),
                            ut)
                        if frm_r:
                            _add('minus', key, frm_r)
                        if to_r:
                            _add('plus', key, to_r)
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
    for tkey, slot in rows['units'].items():
        s2 = (sub['units'].get(tkey) if sub else None) or \
            {'minus': [], 'plus': []}

        def _cnt(pairs):
            d = {}
            for r, n in pairs:
                d[r] = d.get(r, 0) + n
            return d
        m = _sub_pairs(slot['minus'], _cnt(s2['minus']))
        p = _sub_pairs(slot['plus'], _cnt(s2['plus']))
        if m or p:
            out['units'][tkey] = {'minus': m, 'plus': p}
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


def render_forced_rows(view, nick, side, rows, pre, label=''):
    """Render residual side-owned rows as that side's own physical
    lines (the bounded emission_card row rules, applied to the
    forced-allocation class).  Returns (Emission, ledger_rows):
    ledger_rows = the REMOVAL fingerprints rendered (echo-ledger
    entries).  Any row outside the literal rules => HOLD naming it."""
    from . import emission_card as emc
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
        for _ in range(n):
            try:
                op = emitter.plan_muster(scratch, nick, side_name,
                                         params, unit_kind)
                ops.append(op)
                emc._retrack_scratch(scratch, [op])
            except LookupError as e:
                holds.append('%s: pool cannot supply a %s at %s: %s'
                             % (tname, unit_kind, region, e))
                return

    def _move(frm, to, n, nation, unit_kind, tname):
        try:
            pids = _pick(frm, unit_kind, n, nation)
            op = ('move', {
                'nick': nick, 'pieces': pids,
                'frm': emitter.zone_code_for(sanitize(str(frm))),
                'to': emitter._garrison_dest(scratch,
                                             sanitize(str(to)),
                                             side_name)})
            ops.append(op)
            emc._retrack_scratch(scratch, [op])
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
        elif len(M) == 1 and len(P) == 1 and M[0][1] == P[0][1] \
                and cas == 0:
            # the auto-advance / forced-relocation shape (post-retreat
            # victor advance: one army, one source, one destination)
            _move(M[0][0], P[0][0], M[0][1], nation, unit_kind, tkey)
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
        elif len(M) == 1 and len(P) == 1 and M[0][1] == P[0][1]:
            _move(M[0][0], P[0][0], M[0][1], None, 'nazgul', 'nazgul')
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
        pid, pside = None, None
        from . import emission_card as _emc
        pid, pside = _emc._named_pid_side(nm)
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


# ---- settlement-capture markers (WAVE-H blocker 2) ------------------
# donor grammar: ai_move_server.py :10094-10236 (G4 control markers,
# VP reconciler).  Engine truth: post.enemy_controlled sets + game_
# logic.calculate_victory_points(post).  Ledger-reconciled per commit,
# idempotent; a marker the HUMAN already claimed (visible in the dump)
# is ADOPTED, never duplicated.
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
        em.lines.append(render('move', {
            'nick': nick, 'pieces': [pid],
            'frm': _VP_TRACK_ZONES[old],
            'to': _VP_TRACK_ZONES[cur]}, local=False))
        vp[key] = cur
    return em, notes
