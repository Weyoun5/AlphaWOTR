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
    when the chain closes without a die-carrying leg); which dice the
    Hunt Box takes instead is the planners' table (emitter.DIE_TO_HUNT)
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
import re

# R6-DEAD-CODE-4: src/ and wotr_bridge/ come from live/__init__.py's one path
# setup (R5-LIVE-PACKAGE-SYSPATH), which runs before any submodule in every
# package shape; this module keeps no copy of it. Keep `emitter` the bare
# module: one module object.
from . import emitter
from . import layout
from .emitter import render
from . import words

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
        # Paid movement/guide chains may outlive their first rendered move.
        # The server owns closing these dice after all pending work settles.
        self.deferred_die_kills = []
        # lines a PERSON may perform as well as / instead of the last step of
        # `lines` (the declared Fellowship put straight onto the Mordor
        # Track).  Never sent: the AI's own move is `lines`.
        self.may = []

    def __repr__(self):
        if self.hold:
            return 'Emission(HOLD: %s)' % self.hold
        return ('Emission(%d line(s), %d call(s), %d chat)' %
                (len(self.lines), len(self.calls), len(self.chat)))


def plan_of(emitted):
    """The piece changes of an Emission as [(op, fields)] WITHOUT sending it:
    every rendered line read back through the emitter's own line grammar
    (some renderers build lines directly, so `ops` alone is not the whole
    plan), then the client calls as ('directive', {'line': ...})."""
    plan = [emitter.classify_line(line) for line in emitted.lines]
    return plan + [('directive', {'line': call}) for call in emitted.calls]


def may_of(emitted):
    """The piece changes a person may ALSO make for this step, as [(op,
    fields)]: read on top of plan_of's changes, never sent."""
    return [emitter.classify_line(line) for line in emitted.may]


def _hold(reason):
    return Emission(hold=reason)


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
def card_announce(cid):
    """The chat request that names a played card, by the title every other sentence uses."""
    return ('card_announce', 'plays %s' % words.chat_card_name(words.card_title(cid)))


# ---------------------------------------------------------------------
# BL-153 target announce (USER 2026-11-01: "it needs to specify the
# region it is targeting" - 'plays dreadful spells' alone said nothing
# about WHICH army).  Generic over the card_options spec keys that name
# BOARD targets; every other key (hand identities 'army_card' /
# 'character_card' / 'card_id', payment or muster compositions, chain
# markers '_from_pending' / '_tsl_done', 'action' verbs) is ignored by
# construction, so a hidden-hand choice or a face-down pick can never
# leak through this channel.  Order = the order the parts are said.
_CARD_TARGET_LABELS = (
    ('target_region',     '%s'),
    ('target_fp_army',    'the free peoples army in %s'),
    ('attack_from',       'attack from %s'),
    ('attack_target',     'attack on %s'),
    ('from_region',       'from %s'),
    ('region',            'region %s'),
    ('regions',           'regions %s'),
    ('destination',       'to %s'),
    ('to_region',         'to %s'),
    ('recruit_region',    'recruit in %s'),
    ('witch_king_region', 'witch-king to %s'),
    ('strider_region',    'strider in %s'),
    ('stronghold',        'stronghold %s'),
    ('strongholds',       'strongholds %s'),
    ('settlements',       'settlements %s'),
    ('nation',            'nation %s'),
    ('companions',        'companions %s'),
    ('companion',         'companion %s'),
    ('character',         'character %s'),
)


def _target_value_text(v):
    """One board target (or a list of them) as said text: region keys
    are already the sanitized board keys ('dol guldur'); figures carry
    their names.  Never a dict (a nested option body is not a target)."""
    if isinstance(v, (list, tuple, set, frozenset)):
        parts = [_target_value_text(x) for x in v]
        parts = [p for p in parts if p]
        return ', '.join(parts) if parts else ''
    if isinstance(v, dict) or v is None:
        return ''
    s = str(getattr(v, 'value', v)).replace('_', ' ').strip()
    return s.lower()


def card_target_announce(card_name, choices):
    """'<card>: <role> targets <board targets>' for one resolved
    option, or None when the option names no board target (a pure
    verb / count choice says nothing extra).  Shape pinned by the
    'card_target' chat kind: it always carries the word 'targets'.
    The ONLY card named is the played card itself."""
    if not isinstance(choices, dict):
        return None
    name = str(card_name or '').strip().lower()
    if name.startswith('__'):
        # L1104-R01: engine-internal pendings ('__overstack__') BEFORE the
        # accent fold below drops their '_' (C05's hunk alone announced
        # "overstack: ... targets ...")
        return None
    name = words.chat_card_name(name)
    if not name or name.startswith('__'):
        return None          # engine-internal pendings (overstack...)
    parts = []
    for key, label in _CARD_TARGET_LABELS:
        if key not in choices:
            continue
        vt = _target_value_text(choices.get(key))
        if vt:
            parts.append((key, label % vt))
    if not parts:
        return None
    keys = [k for k, _ in parts]
    if 'target_region' in keys and 'target_fp_army' in keys:
        # the Dreadful Spells shape (USER's own wording): the casting
        # Nazgul army and the Free Peoples army it strikes (alone, a target
        # region is where something goes or happens: no army is claimed)
        src = 'army in ' + dict(parts)['target_region']
        dst = dict(parts)['target_fp_army']
        if name == 'dreadful spells':
            src = 'nazgul ' + src
        return '%s: %s targets %s' % (name, src, dst)
    if 'attack_from' in keys and 'attack_target' in keys:
        return '%s: %s targets %s' % (
            name, dict(parts)['attack_from'].replace('attack from ', 'army in '),
            dict(parts)['attack_target'].replace('attack on ', ''))
    dest_key = next((k for k in ('destination', 'to_region') if k in keys),
                    None)
    if 'from_region' in keys and dest_key and len(keys) == 2:
        # a card-driven move: the moving army and where it goes
        return '%s: army in %s targets %s' % (
            name, dict(parts)['from_region'].replace('from ', '', 1),
            dict(parts)[dest_key].replace('to ', '', 1))
    return '%s: targets %s' % (name, '; '.join(t for _, t in parts))


def _forced_card_targets(state, side, cid):
    """BL-153 forced path: the ONE option the engine auto-picks when a
    specced card enumerates exactly one choice at apply time (ai.py
    play_event executor: `len(_opts0) == 1` -> the pick IS the pending
    path, no card_option decision ever opens, so render_decision never
    sees it).  Re-enumerated here on the SAME pre-apply state through
    the same card_options.get_card_options, so the announce names what
    the engine will do.  None when the card is unspecced, enumerates 0
    (ruling 1.230 zero-effect play) or >=2 (a decision: the chosen path
    announces).  A spec failure is the engine's to raise loudly at
    apply; here it simply yields no announce."""
    try:
        from cards import get_card
        from card_options import get_card_options
        from game_state import Side as _Side
        card = get_card(cid)
        cname = getattr(card, 'card_name', '') or ''
        if not cname:
            return None, None
        if not isinstance(side, _Side):
            side = (_Side.SHADOW if emitter.canon_side(side) == 'SHADOW'
                    else _Side.FREE_PEOPLES)
        opts = get_card_options(state, side, cname)
    except Exception:
        return None, None
    if len(opts) != 1:
        return None, None
    return cname, dict(opts[0])


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
        # CR-D3-break-03: a later emitted movement of a piece voids any
        # unconfirmed replacement seat recorded for it below.
        _pending = getattr(view, 'shadow_slot_pending', None) or {}
        if op in _RETRACK_DEAD:
            for p in pids:
                _pending.pop(p, None)
                view.piece_region[p] = _lr.decode_location(layout.kill_box(p, view.piece_types))
        elif op in _RETRACK_TO and f.get('to'):
            name = _lr.decode_location(str(f['to'])) or str(f['to'])
            # a shared Shadow stronghold box names no region: a piece put into
            # one (an arrival, the Regular of a reduced Elite) stands in the
            # stronghold whose pieces the view already has in that box
            tied = next((seat for seat in (getattr(view, 'shadow_stronghold_regions', None) or {}).values()
                         if seat[1] == f['to']), None)
            for p in pids:
                _pending.pop(p, None)
                if tied:
                    view.shadow_stronghold_regions[p] = tied
                    view.piece_region[p] = tied[0] + ' stronghold'
                else:
                    view.piece_region[p] = str(name)


def _ops_to_emission(view, nick, action_type, ops):
    """[(op, fields)] -> Emission: chat ops become (kind, text)
    requests, 'directive' ops become calls, the rest render to lines.
    Emitted piece movement is retracked into the view (see
    _retrack_ops)."""
    em = Emission()
    may = [(op, f) for op, f in ops if (f or {}).get('_may')]
    ops = [(op, f) for op, f in ops if not (f or {}).get('_may')]
    for op, f in may:
        em.may.append(render(op, {k: v for k, v in f.items() if k != '_may'},
                             local=False))
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
            # LIVE1104 Y73 (Y69-6): a new slot takes a piece NO other slot holds (base[i]
            # may already be another slot's piece after interchangeable dice were rebound)
            pid = by_index.get(i) or base[i]
            if pid not in {p for p, (o, _x) in view.die_id_map.items() if o is s}:
                view.die_id_map[pid] = (s, i)
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


def _mouth_used(view, side_name, action, pre_pending=False):
    """True when this Shadow action spends the Mouth of Sauron's Muster-as-Army
    ability for the first time this turn (the figure is flipped once)."""
    params = getattr(action, 'params', None) or {}
    if side_name != 'SHADOW' or not params.get('mouth_muster') or pre_pending:
        return False
    st = getattr(view, 'state', None)
    if st is None or getattr(st, 'sh_mouth_muster_used_this_turn', False):
        return False
    mouth = getattr(st, 'mouth_of_sauron', None)
    return bool(mouth is not None and getattr(mouth, 'in_play', False))


def _mouth_flip(em, view, nick, side_name, action, pre_pending, person):
    """Turn the Mouth of Sauron to his used side when this action spends his ability (_mouth_used):
    to a SIDE - a figure that shows it already is left alone, a blind flip would turn him back -
    and for a person something he may leave to the server."""
    if _mouth_used(view, side_name, action, pre_pending):
        (em.may if person else em.lines).extend(
            render(op, f, local=False) for op, f in emitter.flip_to(view, nick, '255', 1))


def _append_die_close(em, view, nick, side_name, die_idx, post):
    if chain_open(post):
        if die_idx not in em.deferred_die_kills:
            em.deferred_die_kills.append(die_idx)
    else:
        em.lines.append(_die_kill_line(view, nick, side_name, die_idx))


def _deferred_die_kill(em, view, nick, side_name, pre, post, action):
    """A multi-step die whose chain closed WITHOUT a die-carrying leg
    (a pass / die-less leg 2) still gets its kill, or it strands in
    Current Action forever (donor :15731-15758, the die-105 deadlock).
    A new pending effect retains the die until that effect finishes."""
    for pf in ('pending_army_move_die', 'pending_muster_die'):
        pidx = getattr(pre, pf, None)
        if pidx is None or getattr(post, pf, None) is not None:
            continue
        if action.die_idx is not None and action.die_idx >= 0:
            continue      # the closing leg carried the die itself
        _append_die_close(em, view, nick, side_name, int(pidx), post)


# --------------------------------------------------------------------
# engine-truth param threading (post-dependent members only:
# view.state IS the engine state, so no parser-lag member exists)
# --------------------------------------------------------------------
def _moved_leader_figures(pre, post, src, side_name):
    """B9-02: the ENGINE's leader figures of `side_name` that LEFT the
    source region between pre and post, as [[kind, ident], ...] (kind
    'leader' + nation value, 'companion' / 'minion' + figure name, or
    'nazgul' + '').  An explicit LA-8 n_leaders count moves plain
    Leaders first and then Companions/Minions in ARMY-LIST order
    (game_logic.move_army _must_first) - only the engine knows which;
    emitter._explicit_leader_figures renders exactly these."""
    from collections import Counter
    from board import sanitize
    import game_logic
    from game_state import Companion, Nazgul
    want = emitter.canon_side(side_name)
    src = sanitize(str(src or ''))

    def _figs(state):
        c = Counter()
        for k, army in state.armies.items():
            if sanitize(str(k)) != src:
                continue
            for ld in (getattr(army, 'leaders', None) or ()):
                if emitter.canon_side(game_logic._leader_side(ld)) \
                        != want or game_logic._is_saruman(ld):
                    continue
                if isinstance(ld, Nazgul):
                    c[('nazgul', '')] += 1
                elif isinstance(ld, Companion):
                    c[('companion', str(ld.name))] += 1
                elif type(ld).__name__ == 'Minion':
                    c[('minion', str(ld.name))] += 1
                else:
                    c[('leader', str(getattr(ld.nation, 'value',
                                             ld.nation)).lower())] += 1
        return c
    gone = _figs(pre) - _figs(post)
    return [[k, i] for (k, i), n in sorted(gone.items())
            for _ in range(n)]


def _moved_units_by_nation(pre, post, src, side_name):
    """K5: the ENGINE's units of `side_name` that LEFT the source region
    between pre and post, as {'<nation value>:<kind>': n} (lower case -
    the vocabulary plan_army_move's 'moved_by_nation' reads).
    game_logic.move_army moves SPECIFIC units: the first k of each kind
    in ARMY-LIST order, limited to the frozen battle participants on a
    victor_advance and to the border predicate (nation_ok) on a
    voluntary crossing.  A nation-blind pick_units (first pids in board
    order) would ship another nation's piece from a MIXED army.  Shared
    by render_action (_threaded_params) and render_decision
    (R2-RS-B9-EMISSION: victor_advance / card follow-up army_move)."""
    from board import sanitize
    src = sanitize(str(src or ''))
    own = 'shadow' if emitter.canon_side(side_name) == 'SHADOW' else 'fp'

    def _comp(state):
        out = {}
        for k, army in state.armies.items():
            if sanitize(str(k)) != src:
                continue
            for u in army.units:
                nat = str(getattr(u.nation, 'value', u.nation))
                if emitter._side_of_nation(nat) != own:
                    continue      # a shared besieged entry: enemy units
                key = '%s:%s' % (nat.lower(), str(getattr(
                    u.kind, 'value', u.kind)).lower())
                out[key] = out.get(key, 0) + 1
        return out
    c0, c1 = _comp(pre), _comp(post)
    return {k: n - c1.get(k, 0) for k, n in c0.items()
            if n - c1.get(k, 0) > 0}


def _track_reveal_due(pre, post):
    """R5-MORDOR-REVEAL-FLIP: True iff the ENGINE turned the Fellowship from
    hidden to revealed on the Mordor Track between `pre` and `post`.  The
    Track has no reveal placement, so the commit that settled the Hunt owes
    the pawn's flip (emitter.TRACK_REVEAL_FLIP); off the Track the
    fellowship_reveal_place decision renders it.  Engine truth only: no
    post (a pre-apply render) is never due."""
    if pre is None or post is None:
        return False
    f0 = getattr(pre, 'fellowship', None)
    f1 = getattr(post, 'fellowship', None)
    return (f0 is not None and f1 is not None
            and bool(getattr(f0, 'hidden', False))
            and not bool(getattr(f1, 'hidden', True))
            and bool(getattr(f1, 'mordor_entered', False)))


def _placement_owns_flip(side, fp_seat_ai=False):
    """R7-REVEAL-FLIP-RESIDUALS (a): True iff a render by `side` leaves an off-
    Track reveal's ONE 247 flip to the Free Peoples reveal placement.  A Free
    Peoples render always does (R6-OFFTRACK-DOUBLE-FLIP).  A Shadow render does
    only when `fp_seat_ai` - the live caller's seat fact (seats.seat_is_ai,
    never a single-seat read): the FP seat is an AI and so is the rendering
    seat, i.e. AI vs AI, where the AI-FP's fellowship_reveal_place renders
    the flip.  With a human FP the Shadow keeps the flip (the human-FP
    placement grammar, decision_prompts / fellowship_reveal_gestures, expects
    it); sided AI-FP with a human Shadow is unchanged (that Shadow commit is
    the human's own gesture)."""
    s = emitter.canon_side(side)
    return s == 'FREE_PEOPLES' or (s == 'SHADOW' and bool(fp_seat_ai))


def _reveal_placement_follows(pre, post, side, fp_seat_ai=False):
    """R6-OFFTRACK-DOUBLE-FLIP: True iff the ENGINE turned the Fellowship from
    hidden to revealed OFF the Mordor Track between `pre` and `post` in a Free
    Peoples commit and left the reveal placement due (post._reveal_place_due).
    That placement is the FP's own fellowship_reveal_place, and its planner
    renders the pawn's ONE flip (reset -> walk -> flip), so this commit must
    not flip as well (the face is a toggle: two flips showed the client
    hidden while the engine was revealed).  A placement the engine resolved
    itself (forced: nothing due on post) is _forced_reveal_placement's, and
    on the Track _track_reveal_due applies.  Which renders defer is
    _placement_owns_flip (R7-REVEAL-FLIP-RESIDUALS (a): a Shadow render too
    when fp_seat_ai, AI vs AI).  Engine truth only: no post (a pre-apply
    render) never defers."""
    if pre is None or post is None \
            or not _placement_owns_flip(side, fp_seat_ai):
        return False
    f0 = getattr(pre, 'fellowship', None)
    f1 = getattr(post, 'fellowship', None)
    return (f0 is not None and f1 is not None
            and bool(getattr(f0, 'hidden', False))
            and not bool(getattr(f1, 'hidden', True))
            and not bool(getattr(f1, 'mordor_entered', False))
            and bool(getattr(post, '_reveal_place_due', False)))


# R7-REVEAL-FLIP-RESIDUALS (b): actions that render their own Fellowship
# marker gesture; the reveal placement is the choreography itself.
_OWN_FELLOWSHIP_GESTURE = ('fellowship_declare', 'fellowship_hide',
                           'fellowship_reveal_place')


def _forced_reveal_placement(pre, post, side, fp_seat_ai=False):
    """R7-REVEAL-FLIP-RESIDUALS (b): the reveal placement the ENGINE resolved
    INSIDE this commit, as emitter.FORCED_REVEAL_PLACEMENT's value, else None.
    ai.open_reveal_placement places the figure itself - no
    fellowship_reveal_place decision follows - at progress 0, at Morannon /
    Minas Morgul, or with one legal region; the commit that triggered it owes
    the placement's choreography (248 reset -> walk -> 247 flip).  Two shapes:
      - this commit turned the Fellowship from hidden to revealed off the
        Track (a Hunt answer, a Hunt-reduction card, a Gollum reveal, the
        hobbit placement a reveal waited for, a Shadow card's reveal): the
        flip is this commit's - exactly one for the transition;
      - the reveal was rendered earlier and its placement was still due
        (pre._reveal_place_due) when this commit forced it: the earlier
        render left the flip to the placement exactly when
        _placement_owns_flip held, so the flip is owed by the same rule.
    A placement decision still due (post._reveal_place_due) is
    _reveal_placement_follows's; the Track is _track_reveal_due's.  Callers
    skip _OWN_FELLOWSHIP_GESTURE actions.  Engine truth only: no post (a
    pre-apply render) is never forced."""
    if pre is None or post is None:
        return None
    f0 = getattr(pre, 'fellowship', None)
    f1 = getattr(post, 'fellowship', None)
    if (f0 is None or f1 is None
            or bool(getattr(f1, 'hidden', True))
            or bool(getattr(f1, 'mordor_entered', False))
            or bool(getattr(post, '_reveal_place_due', False))):
        return None
    revealed_here = bool(getattr(f0, 'hidden', False))
    deferred = (not revealed_here
                and bool(getattr(pre, '_reveal_place_due', False))
                and not bool(getattr(f0, 'mordor_entered', False)))
    if not (revealed_here or deferred):
        return None
    import game_logic as gl
    from board import sanitize
    return {'region': sanitize(str(gl.fellowship_region(post))),
            'from_progress': int(getattr(f0, 'progress', 0) or 0),
            'reset': int(getattr(f1, 'progress', 0) or 0) == 0,
            'flip': revealed_here or _placement_owns_flip(side, fp_seat_ai)}


def _threaded_params(view, side_name, action, post, fp_seat_ai=False):
    a = action
    at = a.action_type
    params = a.params
    st = view.state

    if at in ('army_move', 'victor_advance') and post is not None \
            and not params.get('moved_by_nation'):
        # K5: the ENGINE moves SPECIFIC units - the exact per-(nation,
        # kind) composition that LEFT the source between pre and post
        # rides the params; plan_army_move honours 'moved_by_nation'
        # (a nation-blind pick would ship the wrong nation's regular)
        _mv = _moved_units_by_nation(st, post, params.get('from'),
                                     side_name)
        if _mv:
            params = dict(params)
            params['moved_by_nation'] = _mv

    if at in ('army_move', 'victor_advance') and post is not None \
            and params.get('n_leaders') is not None \
            and 'moved_leader_figures' not in params:
        # B9-02: an explicit LA-8 leader count names Companions/Minions
        # by ENGINE army order - the exact figures ride the params
        params = dict(params)
        params['moved_leader_figures'] = _moved_leader_figures(
            st, post, params.get('from'), side_name)

    if at in ('army_move', 'victor_advance') and post is not None \
            and params.get('take_leaders') \
            and params.get('n_leaders') is None \
            and 'moved_leader_figures' not in params:
        # B9-03: the take-all form is NOT every leader piece at the
        # source - move_army holds back a non-belligerent Nation's plain
        # Leader on a voluntary foreign-border crossing (nation_ok) and
        # a victor_advance carries only the frozen battle participants
        # (combat_context).  The leader figures that LEFT ride the
        # params, like the units' moved_by_nation above.
        params = dict(params)
        params['moved_leader_figures'] = _moved_leader_figures(
            st, post, params.get('from'), side_name)

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
        n = int(params.get('count') or 0)       # (no count is the engine's "no Eye")
        if n > 0:
            idxs = [i for i, d in enumerate(st.dice.get(Side.SHADOW, []))
                    if not d.used][:n]
            params = dict(params)
            params['die_indices'] = idxs

    if _track_reveal_due(st, post):
        # R5-MORDOR-REVEAL-FLIP: a Track reveal settled by this action (a
        # no-damage reveal tile on the Fellowship move, a Hunt answer) -
        # the planner appends the pawn's one flip after the action's own ops
        params = dict(params)
        params[emitter.TRACK_REVEAL_FLIP] = True
    if _reveal_placement_follows(st, post, side_name, fp_seat_ai):
        # R6-OFFTRACK-DOUBLE-FLIP: an off-Track reveal whose placement is
        # still due - the FP's fellowship_reveal_place renders the one flip
        # (R7-REVEAL-FLIP-RESIDUALS (a): fp_seat_ai, the AI vs AI Shadow too)
        params = dict(params)
        params[emitter.REVEAL_FLIP_TO_PLACEMENT] = True
    _forced = (None if at in _OWN_FELLOWSHIP_GESTURE else
               _forced_reveal_placement(st, post, side_name, fp_seat_ai))
    if _forced is not None:
        # R7-REVEAL-FLIP-RESIDUALS (b): the engine placed the revealed
        # figure inside this action (a no-damage reveal tile on the move) -
        # the planner appends the placement's reset -> walk -> flip
        params = dict(params)
        params[emitter.FORCED_REVEAL_PLACEMENT] = _forced
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
    pid = _card_pid(view, params.get('card_id') or '')
    if pid is not None and emitter._piece_zone_code(
            view, pid, default=None) not in emitter._TABLE_SLOTS:
        pid = None              # that piece does not lie in a Board slot
    seats = getattr(view, 'table_seats', None)
    getter = getattr(view, 'table_seat_pid', None)
    if pid is not None:
        pass
    elif callable(getter):
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
        if side_name == 'FREE_PEOPLES' and die_idx >= 0:
            # the Ring makes his die a Will of the West: a person may turn
            # the staged die to show it (the owner did, 2026-11-04)
            em.may.append(render('die_change', {
                'nick': nick, 'die': f['die'], 'face': 'W'}, local=False))
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


def render_action(view, nick, side, action, post=None, ring_paid=None,
                  fp_seat_ai=False, person=False):
    """Render the client gestures for ONE applied engine action.

    view   ClientView (engine state + client dump + die bindings)
    side   Side enum or name (the actor)
    action the engine Action (action_type, params, die_idx)
    post   the committed post-action engine state, when the driver has
           it (it always does after a successful apply).  Post-
           dependent bookkeeping HOLDS without it rather than guessing.
    ring_paid  optional set: the driver's FP elven-ring departure
           ledger (donor :16182-16190), threaded to pick_fp_ring_walk.
    fp_seat_ai  R7-REVEAL-FLIP-RESIDUALS (a): the live caller's seat fact
           (the FP seat and the rendering seat are AI seats) for the
           off-Track reveal-flip rule (_placement_owns_flip).
    person True (core/menu.py): the move as a PERSON performs it, before
           its result is known - the Fellowship is walked on the Mordor
           Track before the tile is drawn (the engine may then keep it
           where it was), the Dwarven Ring is spent before the d6.  The
           AI's own moves never pass it.

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
                        _append_die_close(em, view, nick, side_name, int(pidx), post)
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
        _mouth_flip(em, view, nick, side_name, a, pre_pending, person)
        em.chat.append(card_announce(cid))
        em.calls.append('!CALL play_card %s' % pid)
        # BL-153: a specced card whose spec yields exactly ONE option is
        # resolved inside the play itself (ai.py play_event executor
        # auto-pick) - no card_option decision opens, so the target is
        # said HERE, from the same enumeration on the same pre-apply
        # state.  >=2 options defer to a decision (render_decision's
        # card_option branch announces the chosen one); 0 = zero-effect.
        if a.params.get('choices') is None:
            _fc_name, _fc_choices = _forced_card_targets(st, side, cid)
            _fc_text = card_target_announce(_fc_name, _fc_choices)
            if _fc_text:
                em.chat.append(('card_target', _fc_text))
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
        _mouth_flip(em, view, nick, side_name, a, pre_pending, person)
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
        em.calls.append(emitter.draw_call(side_name, a.params['deck']))
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
    put_off = ((post.pending_card_option or {}).get('resume_action') or {}) if post is not None else {}
    if not person and put_off.get('action_type') == at:
        # the engine put the action off behind a card choice of the other
        # side and takes it up when that is answered: nothing has moved yet,
        # only its die is staged.  (A person's own move is read whole, as he
        # makes it.)
        try:
            return _ops_to_emission(view, nick, at, [emitter.plan_die_spend(
                view, nick, side_name, die_idx)] if die_idx >= 0 else [])
        except LookupError as e:
            return _hold('%s die spend: %s' % (at, e))

    params = _threaded_params(view, side_name, a, post, fp_seat_ai)
    if person:
        params = dict(params, _person=True)
        params.pop('mordor_post_step', None)    # his walk comes before the tile
    if post is None and not person and (at in _POST_ONLY_TYPES or (
            at == 'fellowship_move'
            and getattr(st.fellowship, 'mordor_entered', False))):
        return Emission()
    if at == 'elven_ring' and ring_paid is not None:
        params = dict(params)
        params['_ring_paid'] = ring_paid
    to_hunt = emitter.die_goes_to_hunt(st, at, post)    # (the engine's answer, after the move when known)
    params = dict(params, die_to_hunt=to_hunt) if at in emitter.DIE_TO_HUNT else params

    try:
        ops = emitter.plan(at, params, view, nick, side_name,
                           die_idx=(die_idx if die_idx >= 0 else None))
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
    _mouth_flip(em, view, nick, side_name, a, pre_pending, person)

    # the LEARNED zero-eye choice is visible only in chat (D3: '0e';
    # absence-is-the-choice)
    if at == 'eye_allocate' and int(params.get('count') or 0) == 0:
        em.chat.append(('zero_eyes', '0e'))

    # ---- die kill to the Used box (the Hunt Box's die is the planner's; an undecided one stays staged) ----
    if die_idx >= 0 and not to_hunt:
        due = True
        if post is not None:
            due = (post.pending_army_move_die is None
                   and getattr(post, 'pending_muster_die', None) is None)
        elif at in _MULTISTEP_TYPES:      # unreachable (held above)
            due = False
        if due:
            try:
                if person and post is None:
                    # before the result his action is not over: the kill is his to leave
                    em.may.append(_die_kill_line(view, nick, side_name, die_idx))
                elif to_hunt is False:
                    _append_die_close(em, view, nick, side_name, die_idx, post)
            except LookupError as e:
                return _hold('%s die kill: %s' % (at, e))

    if post is not None:
        try:
            _deferred_die_kill(em, view, nick, side_name, st, post, a)
            # what the engine does alone after the move (a bonus die enters
            # the pool, the one possible new Guide is seated): the AI's
            # render shows it; a person MAY do it himself
            also = em.may if person else em.lines
            also += _pool_growth_lines(view, nick, st, post)
            also += _forced_guide_lines(view, nick, st, post)
        except LookupError as e:
            return _hold('%s post bookkeeping: %s' % (at, e))
    return em


def _render_unit_loss(view, nick, side_name, action, post, person):
    """One unit its owner gives up by a choice of his own that is not the
    battle's casualty decision - a card's casualty, a unit a combat card makes
    its own side pay: the casualty planner's lines.  The Regular of a reduced
    Elite comes from the box the engine stamped; before the engine applied
    the choice (a person's own move) from the box the board offers."""
    st = view.state
    params = emitter.casualty_action_params(action)
    pending = getattr(st, 'pending_card_option', None) or {}
    if params is None or (action.action_type == 'card_option'
                          and action.params.get('card_id') != pending.get('card_id')):
        return _hold('card casualty action does not identify this pending allocation')
    if post is not None:
        n0 = len(getattr(st, '_dg_src_log', None) or [])
        params['_dg_srcs'] = list(getattr(post, '_dg_src_log', None) or [])[n0:]
    elif person:
        params['_before'] = True
    try:
        ops = emitter.plan_casualty_operation(params, view, nick, side_name)
    except (LookupError, ValueError) as error:
        return _hold('card casualty operation: %s' % error)
    return _ops_to_emission(view, nick, action.action_type, ops)


def render_step(view, nick, side, kind, action, post=None, pre=None, fp_seat_ai=True, person=False):
    """ONE step of either seat as an Emission: THE dispatch from the kind of its decision
    to the renderer that knows it (a decision of a pending chain, or an action)."""
    if kind.startswith(('combat', 'hunt', 'guide')) or kind in ('discard', 'card_option', 'card_followup',
                                                                'reveal_place'):
        return render_decision(view, nick, side, kind, action, post, pre, fp_seat_ai, person)
    return render_action(view, nick, side, action, post=post, fp_seat_ai=fp_seat_ai, person=person)


def render_decision(view, nick, side, kind, action, post=None,
                    pre=None, fp_seat_ai=False, person=False):
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
    `fp_seat_ai` (R7-REVEAL-FLIP-RESIDUALS (a)): the live caller's seat
    fact for the off-Track reveal-flip rule (_placement_owns_flip).
    """
    side_name = emitter.canon_side(side)
    st = view.state
    a = action
    at = a.action_type

    # ---- the AI's own hand-limit discard: the DECISION path renders
    # it exactly as render_action does ----------------------------------
    if at == 'discard_card':
        return render_action(view, nick, side, a, post=post,
                             fp_seat_ai=fp_seat_ai, person=person)

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
        if emitter.casualty_action_params(a) is not None:
            return _render_unit_loss(view, nick, side_name, a, post, person)
        # a count decision announces its magnitude; the board cannot show it
        return Emission(chat=[('prompt', words.combat_effect_choice_note(a.params))])

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
    if at == 'card_option' and (getattr(st, 'pending_card_option', None) or {}).get('card_casualty'):
        return _render_unit_loss(view, nick, side_name, a, post, person)

    # ---- card_option: the SANCTIONED delta-scoped row renderer ------
    # (bounded exception to rule 6, coordinator 2026-08-19; provenance
    # + row rules in emission_card's docstring.  card_followup KIND
    # decisions carry real army_move/army_attack/followup_done actions
    # and ride their own planners below.)
    if at == 'card_option':
        from . import emission_card
        first = emission_card.own_move(view, nick, side_name, a,
                                       pre if pre is not None else st)
        if post is None and first is not None:
            # the choice needs the client's dice / tile / random discard
            # before the engine can apply it: its own move is known already
            return first
        if first is not None and first.hold:
            first = None                # with the result known the rows alone say what moves
        em = emission_card.render_card_effect(view, nick, side, kind,
                                              a, pre, post,
                                              fp_seat_ai=fp_seat_ai)
        if em.hold and first is not None:
            # a row without a rule is the comparison's to show; the own move stays known
            em, first = first, None
        if not person:
            em.lines += em.may          # what the rules do alone the AI's render shows
            em.may = []
        if not em.hold and first is not None and person:
            # for a PERSON the choice's own move is all he owes, whether or not its result is
            # known already (the redrawn tile's damage, the chained card's effect): what the
            # engine resolves after it is his to leave to the server
            em.may = list(em.lines) + list(em.may)
            em.lines, em.calls = list(first.lines), list(first.calls)
            em.chat += first.chat
        elif not em.hold and first is not None:
            # the AI's render: the rows show the board as the engine made it; the own move
            # adds what no row can show, its commands to the client
            em.calls += [c for c in first.calls if c not in em.calls]
            em.chat += first.chat
        if not em.hold:
            try:
                (em.may if person else em.lines).extend(
                    _forced_guide_lines(view, nick, pre or st, post))
            except LookupError as e:
                return _hold('%s forced Guide: %s' % (at, e))
        if not em.hold and isinstance(_chx, dict):
            # BL-153 chosen path: the AI's resolved option names its
            # board targets out loud ('dreadful spells: nazgul army in
            # lorien targets the free peoples army in lorien').  The
            # card name comes from the pending the decision answered
            # (engine truth), the targets from the chosen option's
            # board-target keys only (card_target_announce whitelist).
            _pco = getattr(pre if pre is not None else st,
                           'pending_card_option', None) or {}
            _ct_name = _pco.get('card_name')
            if not _ct_name:
                try:
                    from cards import get_card as _gc_ct
                    _ct_name = getattr(_gc_ct(str(
                        (a.params or {}).get('card_id') or '')),
                        'card_name', '')
                except Exception:
                    _ct_name = ''
            _ct_text = card_target_announce(_ct_name, _chx)
            # a cargo sub-step of a staged army card repeats nothing
            if _ct_text and not (isinstance(_chx, dict) and 'cargo' in _chx):
                em.chat.append(('card_target', _ct_text))
        if not em.hold and post is not None and isinstance(_chx, dict) \
                and _chx.get('wizardstaff_prevent'):
            # BL-40 (B5-05): the Wizard's Staff PREVENT discards the
            # decider's OWN table card and the Shadow draws NO tile -
            # nothing on the board says why the hunt stopped, so it is
            # SAID.  Stated only when the ENGINE took the Staff off the
            # table (I1).
            ws_cid = str((a.params or {}).get('card_id') or '')

            def _fp_table(state):
                pcs = getattr(state, 'persistent_cards', None)
                return {str(getattr(c, 'card_id', ''))
                        for c in (getattr(pcs, 'fp_cards', None) or ())}
            if ws_cid and ws_cid in _fp_table(pre if pre is not None
                                              else st) \
                    and ws_cid not in _fp_table(post):
                em.chat.append(('prompt', words.table_discard_cancel_note()))
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
    if at in ('casualty_allocate', 'siege_press') and post is None and person:
        params = dict(params)
        params['_before'] = True        # a person reduces his Elite before the next roll or draw

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
            ops = emitter.plan(at, params, view, nick, side_name)
        except NotImplementedError as e:
            return _hold('no planner for %r: %s' % (at, e))
        except LookupError as e:
            return _hold('%s: boards cannot supply the pieces: %s'
                         % (at, e))
        em = _ops_to_emission(view, nick, at, ops)
        # the press announce disambiguates the swap
        em.chat.insert(0, ('press', 'press'))
        return em

    if at in ('victor_advance', 'army_move') and post is not None \
            and not params.get('moved_by_nation'):
        # R2-RS-B9-EMISSION (K5 on the DECISION path): a victor_advance
        # and a card follow-up army_move move SPECIFIC units (army-list
        # order, the frozen battle participants, the border predicate) -
        # the ENGINE's per-(nation, kind) composition that LEFT the
        # source rides the params on EVERY row (partial advances and
        # splits of a mixed army included), exactly as render_action's
        # _threaded_params does
        _mv = _moved_units_by_nation(pre if pre is not None else st,
                                     post, params.get('from'), side_name)
        if _mv:
            params = dict(params)
            params['moved_by_nation'] = _mv

    if at in ('victor_advance', 'army_move') and post is not None \
            and params.get('n_leaders') is not None \
            and 'moved_leader_figures' not in params:
        # B9-02: the LA-8 count names Companions/Minions by ENGINE army
        # order - only the post state says which rode (without post the
        # planner renders plain Leaders only and refuses anything else
        # loudly - the render_action contract).  R2-RS-B9-EMISSION: a
        # card follow-up army_move LA-8 row (the Witch-king's army with
        # the Mouth) rides the same list, not only the advance
        params = dict(params)
        params['moved_leader_figures'] = _moved_leader_figures(
            pre if pre is not None else st, post, params.get('from'),
            side_name)

    if at in ('victor_advance', 'army_move') and post is not None \
            and params.get('take_leaders') \
            and params.get('n_leaders') is None \
            and 'moved_leader_figures' not in params:
        # B9-03: a take-all advance carries only the frozen battle
        # participants (move_army combat_context: a rearguard Leader
        # stays home) and a card follow-up army_move obeys the same
        # border predicate as the action path - the ENGINE's figures
        # that left ride the params (see _threaded_params)
        params = dict(params)
        params['moved_leader_figures'] = _moved_leader_figures(
            pre if pre is not None else st, post, params.get('from'),
            side_name)

    if at == 'combat_retreat' and post is not None \
            and 'moved_by_nation' not in params \
            and 'moved_leader_figures' not in params:
        # B9-04: ai._retreat_army moves SPECIFIC figures - never Saruman
        # (he never leaves Orthanc), and on an attacker's 'retreat your
        # army' card only the battle participants (the rearguard frozen
        # at declaration stays home).  The units and leader figures of
        # the retreating side that LEFT the battle region between pre
        # and post ride the params; emitter.plan('combat_retreat')
        # renders exactly those.
        from board import sanitize as _san_r
        _pre_r = pre if pre is not None else st
        _src_r = _san_r(str(
            params.get('from') or params.get('region')
            or emitter._battle_region_of(
                getattr(_pre_r, 'pending_combat', None) or {}, side_name)
            or ''))
        _own_r = ('shadow' if emitter.canon_side(side_name) == 'SHADOW'
                  else 'fp')

        def _units_r(state):
            out = {}
            for _k, _army in state.armies.items():
                if _san_r(str(_k)) != _src_r:
                    continue
                for _u in _army.units:
                    _nat = str(getattr(_u.nation, 'value', _u.nation))
                    if emitter._side_of_nation(_nat) != _own_r:
                        continue
                    _key = '%s:%s' % (_nat.lower(), str(getattr(
                        _u.kind, 'value', _u.kind)).lower())
                    out[_key] = out.get(_key, 0) + 1
            return out
        _u0, _u1 = _units_r(_pre_r), _units_r(post)
        params = dict(params)
        params['moved_by_nation'] = {
            k: n - _u1.get(k, 0) for k, n in _u0.items()
            if n - _u1.get(k, 0) > 0}
        params['moved_leader_figures'] = _moved_leader_figures(
            _pre_r, post, _src_r, side_name)

    # ---- everything else rides the planner table ---------------------
    plan_at = at
    deferred_victim = False
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
        # A surviving Hobbit is held by the engine's placement pending/queue.
        # Its eventual companion_separate decision owns the physical move;
        # absence from the Fellowship alone does not establish a death.
        deferred_victim = (not params.get('to_region') and not
                           _companion_out_of_play(post, emitter._companion_pid(gone[0])))
    if _track_reveal_due(pre if pre is not None else st, post):
        # R5-MORDOR-REVEAL-FLIP: this decision settled a Track Hunt reveal
        # (a damage answer, or the Guide choice after a Guide's death) -
        # the planner appends the pawn's one flip after its own ops
        params = dict(params)
        params[emitter.TRACK_REVEAL_FLIP] = True
    if _reveal_placement_follows(pre if pre is not None else st, post,
                                 side_name, fp_seat_ai):
        # R6-OFFTRACK-DOUBLE-FLIP: this decision settled an off-Track reveal
        # (hunt_gollum_reveal) and the FP's fellowship_reveal_place follows -
        # that placement renders the one flip, this decision none
        # (R7-REVEAL-FLIP-RESIDUALS (a): fp_seat_ai, the AI vs AI Shadow too)
        params = dict(params)
        params[emitter.REVEAL_FLIP_TO_PLACEMENT] = True
    _forced = (None if at in _OWN_FELLOWSHIP_GESTURE else
               _forced_reveal_placement(pre if pre is not None else st, post,
                                        side_name, fp_seat_ai))
    if _forced is not None:
        # R7-REVEAL-FLIP-RESIDUALS (b): the engine placed the revealed figure
        # inside this decision (a Hunt answer or Gollum reveal at Morannon /
        # Minas Morgul, at progress 0 or with one legal region; the hobbit
        # placement a reveal waited for) - the planner appends the
        # placement's reset -> walk -> ONE flip after the decision's own ops
        params = dict(params)
        params[emitter.FORCED_REVEAL_PLACEMENT] = _forced
    if person:
        params = dict(params, _person=True)
    try:
        ops = ([] if deferred_victim else
               emitter.plan(plan_at, params, view, nick, side_name))
    except NotImplementedError as e:
        return _hold('no planner for decision %r (kind %r): %s'
                     % (at, kind, e))
    except LookupError as e:
        return _hold('%s: boards cannot supply the pieces: %s'
                     % (at, e))
    em40 = _ops_to_emission(view, nick, at, ops)
    if not em40.hold:
        try:
            (em40.may if person else em40.lines).extend(
                _forced_guide_lines(view, nick, st, post))
        except LookupError as e:
            return _hold('%s forced Guide: %s' % (at, e))
    return em40


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

