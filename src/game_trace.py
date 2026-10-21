"""Self-play game trace recording and loading.

Every self-play training episode can be recorded as a trace file:
a gzip-pickled dict with one full GameState snapshot per executed action,
so any game can be stepped through afterwards in the GUI:

    python main.py --gui --replay-trace selfplay_traces/ep00042_sh_military.trace.gz

Format:
    {
        'meta':  {'winner', 'victory_type', 'turns', 'mcts_sims', ...},
        'steps': [
            {'kind': 'initial'|'action'|'new_turn',
             'turn': int, 'side': 'FP'|'SH'|None,
             'action_type': str|None, 'params': dict, 'die_idx': int|None,
             'state': GameState (board stripped — reattached on load)},
            ...
        ],
    }
"""
import gzip
import copy
import os
import pickle


def _side_tag(side):
    if side is None:
        return None
    # Avoid importing Side here; accept the enum name AND the
    # canonical short tag
    s = getattr(side, 'name', str(side)).upper()
    return 'FP' if s in ('FP', 'FREE_PEOPLES') else 'SH'


def snapshot_state(state):
    """Deep-enough copy of a GameState, safe to pickle.

    Uses ai.clone_state (the fast tree-search clone) and strips the shared
    Board object; load_trace() reattaches a freshly loaded board.
    """
    from ai import clone_state
    snap = clone_state(state)
    snap.board = None
    return snap


class TraceRecorder:
    """Collects per-action snapshots during one self-play episode."""

    def __init__(self, path):
        self.path = path
        self.steps = []
        import ai
        from trace_decisions import current_learning_context
        self.meta = {'source_integrity_version': 1, 'source_errors': [],
                     **current_learning_context()}
        from selector_contract import training_identity
        self.meta.update(training_identity())
        self._execution_start = None
        self._source_cursor = self._source_count()
        # Annotation rows fired DURING an action's execution (a hunt
        # roll consumed inside fellowship_move) are deferred and flushed
        # right AFTER the action that consumed them (nested rows carry
        # 'ctx'); turn-level rolls (no ctx) stay immediate.

    @staticmethod
    def _source_count():
        import state_invariants
        return len(state_invariants.VIOLATIONS)

    def _source_error(self, code, detail, step=None):
        self.meta['source_errors'].append(
            {'code': str(code), 'detail': str(detail),
             'step': len(self.steps) if step is None else step})

    def _capture_source_errors(self, start, step=None, only=None):
        """Capture real execution/rollover notices, never search notices."""
        import state_invariants
        for record in state_invariants.VIOLATIONS[start:]:
            code = str(record.get('code') or 'source_error_unknown')
            if only is not None and code not in only:
                continue
            self._source_error(code, record.get('detail') or record.get('ctx') or code,
                               step=step)
        self._source_cursor = self._source_count()

    def _abandon_action_buffer(self):
        if self._execution_start is not None:
            self._capture_source_errors(self._execution_start)
            self._source_error('uncommitted_action',
                               'real action began but no post-action row was recorded')
            self._execution_start = None
            self._ann_buffer = []
            self.buffering = False

    def record_initial(self, state):
        self.steps.append({
            'kind': 'initial', 'turn': state.turn, 'side': None,
            'action_type': None, 'params': {}, 'die_idx': None,
            'state': snapshot_state(state),
        })

    def begin_action_buffer(self):
        """Buffer annotations until the causing action's row lands."""
        # The old orphan path appended (side, action) tuples to `steps`,
        # whose schema requires dictionaries. An interrupted transaction
        # has no truthful post-state to attach: retain a failure record.
        self._abandon_action_buffer()
        self.buffering = True
        self._ann_buffer = []
        self._execution_start = self._source_count()

    def record_action(self, state, side, action, note='',
                      search_stats=None, regret=None, decision_context=None,
                      forced_policy=None):
        """Record a snapshot of the state AFTER `action` was executed.

        search_stats / regret (item 5 instrumentation, 2026-08-02):
        optional per-decision root-search statistics — a dict
        {'ae','ri','p','q','n','played'} (see ai._search_stats_payload)
        and the scalar completed-Q regret of the executed pick.  Only
        stamped when provided (annotation rows carry neither).
        forced_policy records a fixed AI transition without learning targets.
        Its state and annotations remain in chronological game history."""
        role = {}
        if forced_policy is not None:
            from trace_decisions import forced_policy_reason
            role = {'policy_forced': True, 'forced_policy_reason': forced_policy}
            forced_policy_reason(dict(kind='action', side=_side_tag(side), **role))
            if search_stats is not None or regret is not None:
                raise ValueError('forced policy actions cannot carry search targets or regret')
        if self._execution_start is not None and side is None:
            # A later turn/phase snapshot is not the missing decision's
            # commit. Phase exception handlers can resume at this anchor.
            self._abandon_action_buffer()
        if self._execution_start is not None:
            self._capture_source_errors(self._execution_start)
            self._execution_start = None
        else:
            # This diagnostic is emitted only by the actual episode's
            # phase dispatcher, outside _traced_execute's action window.
            self._capture_source_errors(self._source_cursor,
                                        only={'eye_phase_hijacked'})
        step = {
            'kind': 'action', 'turn': state.turn, 'side': _side_tag(side),
            'action_type': action.action_type,
            'params': dict(action.params) if action.params else {},
            'die_idx': action.die_idx,
            'note': note,
            'state': snapshot_state(state),
        }
        step.update(role)
        if search_stats is not None:
            # Search arrays and resolved settings are evidence for this action,
            # not a live view of a mutable caller payload.
            step['search_stats'] = copy.deepcopy(search_stats)
        if regret is not None:
            step['regret'] = float(regret)
        if decision_context is not None:
            # Context describes the state BEFORE this action. The producer
            # owns its schema/phase timing; copy mutable arrays and mappings
            # so later execution cannot change recorded training inputs.
            step['decision_context'] = copy.deepcopy(decision_context)
        # SINGLE ordering mechanism (USER 2026-07-09: 'why does there
        # have to be this buffering'): the execution-scoped buffer in
        # _traced_execute places annotations right after their causing
        # action; rows are otherwise written in true chronological
        # order.
        self.steps.append(step)

        if getattr(self, 'buffering', False):
            self.buffering = False
            _buf = self._ann_buffer
            self._ann_buffer = []
            for _bside, _bns in _buf:
                self.record_action(state, _bside, _bns)

    def record_new_turn(self, state):
        self._abandon_action_buffer()
        self._capture_source_errors(self._source_cursor)
        self.steps.append({
            'kind': 'new_turn', 'turn': state.turn, 'side': None,
            'action_type': None, 'params': {}, 'die_idx': None,
            'state': snapshot_state(state),
        })

    def save(self, **meta):
        """Write the trace to self.path (creates parent dir if needed)."""
        self._abandon_action_buffer()
        self.meta.update(meta)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with gzip.open(self.path, 'wb') as f:
            pickle.dump({'meta': self.meta, 'steps': self.steps}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        return self.path


# ---------------------------------------------------------------------
# iter 38 (user directive): SELF-PLAY trace visibility for combat rolls,
# leader re-rolls, hunt rolls and hunt tile draws.  Engine roll sites
# call emit_annotation(); it is a no-op unless a tracer is active (set
# by the self-play episode drivers).  The closed loop has its own
# trace_cb and never sets ACTIVE_TRACER.
ACTIVE_TRACER = None


def emit_annotation(state, atype, params, side=None):
    """Emit a dice_roll / hunt_tile_draw annotation row into the active
    self-play trace (no-op when no tracer is active).

    Ordering (USER 2026-07-09): annotations emitted while an action is
    EXECUTING are buffered and flushed right after that action's own
    row — they used to appear before it (tile draws printed before the
    fellowship_move that caused them, reveal flicker in the GUI)."""
    t = ACTIVE_TRACER
    if t is None:
        return
    try:
        from types import SimpleNamespace
        ns = SimpleNamespace(action_type=atype, params=dict(params),
                             die_idx=-1)
        if getattr(t, 'buffering', False):
            t._ann_buffer.append((side, ns))
        else:
            t.record_action(state, side, ns)
    except Exception as exc:
        # A missing annotation can omit exogenous outcomes or a source
        # failure. Preserve a rejectable marker even if gameplay continues.
        t._source_error('annotation_failed', '%s: %s' % (type(exc).__name__, exc))


def load_trace(path):
    """Load a trace file and reattach the board to every snapshot."""
    import board as board_mod
    b = board_mod.load_board()
    with gzip.open(path, 'rb') as f:
        trace = pickle.load(f)
    for step in trace['steps']:
        if step.get('state') is not None:
            step['state'].board = b
        context = step.get('decision_context')
        if isinstance(context, dict) and context.get('state') is not None:
            context['state'].board = b
    return trace


# Canonical die-face letters (the app's own convention: its rolls
# print "H A C P").  ONE map, keyed by the enum; any string input is
# resolved through the enum itself (its values are the canonical
# strings, e.g. MUSTER_ARMY = "Muster/Army") — no alias tables.
from game_state import ActionDieFace as _ADF

_FACE_LETTER = {
    _ADF.CHARACTER: 'C', _ADF.ARMY: 'A', _ADF.MUSTER: 'M',
    _ADF.MUSTER_ARMY: 'H', _ADF.EVENT: 'P', _ADF.EYE: 'E',
    _ADF.WILL_OF_WEST: 'W',
}


def _face_letter(face):
    """Canonical letter for a die face given as enum, enum value string
    ('Muster/Army'), enum name ('MUSTER_ARMY'), or already a letter."""
    if isinstance(face, _ADF):
        return _FACE_LETTER[face]
    s = str(face)
    try:
        return _FACE_LETTER[_ADF(s)]          # enum VALUE string
    except ValueError:
        pass
    if s.upper() in _ADF.__members__:
        return _FACE_LETTER[_ADF[s.upper()]]  # enum NAME string
    if s.upper() in ('C', 'A', 'M', 'H', 'P', 'E', 'W'):
        return s.upper()                      # already canonical
    if s.isdigit():
        return s                              # numeric combat pips
    return '?'


def _card_display_name(params):
    """Best-effort human name for a card referenced in step params."""
    name = params.get('card_name') or params.get('name')
    if name:
        return str(name)
    card_id = params.get('card_id') or params.get('card')
    if not card_id:
        return '?'
    card_id = str(card_id)
    try:
        from cards import get_card
        card = get_card(card_id)
        if card is not None and getattr(card, 'card_name', None):
            return card.card_name
    except Exception:
        pass
    # Fall back to prettifying the id (strip SIDE_Deck_ prefix, underscores)
    pretty = card_id
    for prefix in ('FREE_PEOPLES_Character_', 'FREE_PEOPLES_Strategy_',
                   'SHADOW_Character_', 'SHADOW_Strategy_'):
        if pretty.startswith(prefix):
            pretty = pretty[len(prefix):]
            break
    return pretty.replace('_', ' ')


def _step_side_tag(step, params):
    """'FP'/'SH' for a step, preferring explicit params['side']."""
    ps = params.get('side')
    if ps:
        ps = getattr(ps, 'name', str(ps))
        return 'FP' if ps.upper() in ('FP', 'FREE_PEOPLES') else 'SH'
    return step.get('side') or '?'


def describe_step(step):
    """Human-readable one-liner for a trace step (for GUI/log display)."""
    if step['kind'] == 'initial':
        return "Game start"
    if step['kind'] == 'new_turn':
        return f"--- Turn {step['turn']} begins ---"
    if step['kind'] == 'final':
        return f"Game end - turn {step['turn']}"
    params = step.get('params') or {}
    atype = step.get('action_type')
    side = _step_side_tag(step, params)
    prefix = f"T{step['turn']} "
    text = None

    if atype == 'dice_roll':
        faces = params.get('faces') or params.get('roll') or []
        try:
            faces_txt = ' '.join(_face_letter(f) for f in faces)
        except TypeError:
            faces_txt = str(faces)
        text = f"{side} rolls: {faces_txt}" if faces_txt else f"{side} rolls dice"
        # Show the roll's context (hunt roll / combat round / Sudden Strike
        # pre-combat attack, ...) so a roll is not an unlabelled bag of pips.
        # Combat sites emit ctx like 'siege battle at Erebor r1 attacker';
        # the pre-combat card attacks (Sudden Strike / Charge / Durin's
        # Bane) SHOULD emit ctx naming the card so they read distinctly.
        _ctx = str(params.get('ctx') or '')
        if _ctx:
            text += f" ({_ctx})"
    elif atype == 'hunt_tile_draw':
        tile = (params.get('tile_name') or params.get('tile')
                or params.get('value') or '?')
        text = f"Hunt tile: {tile}"
        _ctx = str(params.get('ctx') or '')
        if _ctx and _ctx != 'hunt':
            text += f" ({_ctx})"
    elif atype == 'stronghold_tile_draw':
        tile = params.get('tile_name', '?')
        text = (f"Shadow-stronghold hunt tile at "
                f"{params.get('region', '?')}: {tile}"
                f" [{params.get('ctx', '')}]")
    elif atype == 'random_draw':
        text = (f"Random draw ({params.get('ctx', '?')}): "
                f"{params.get('result', '?')}")
        if params.get('pool'):
            text += f"  [pool: {', '.join(params['pool'])}]"
    elif atype == 'fellowship_reveal_place':
        text = (f"Fellowship revealed - Ring-bearers placed "
                f"{params.get('from', '?')} -> {params.get('to', '?')}"
                f" ({params.get('steps', '?')} steps, progress reset)")
    elif atype == 'combat_card':
        text = f"{side} combat card: {_card_display_name(params)}"
    elif atype == 'combat_card_play':
        _cnm = _card_display_name(params)
        # show the COMBAT side's name — the event-side title misled a
        # user audit into seeing "illegal" plays (The King is Revealed
        # played as its combat side Relentless Assault)
        try:
            from cards import get_card
            _c = get_card(str(params.get('card_id') or ''))
            if _c is not None and (_c.combat_name or '').strip():
                _cnm = f"{_c.combat_name} (combat side of {_cnm})"
        except Exception:
            pass
        text = f"{side} plays combat card: {_cnm}"
    elif atype == 'combat_card_none':
        text = f"{side} declines combat card"
    elif atype == 'combat_effect_choice':
        n = params.get('count', '?')
        text = (f"{side} combat-card effect choice: count={n} "
                f"({_card_display_name(params)})")
    elif atype == 'card_demo_before':
        text = (f"DEMO [{params.get('card_name', '?')}] BEFORE - "
                f"expected: {params.get('expect', '?')}")
    elif atype == 'card_demo_after':
        text = (f"DEMO [{params.get('card_name', '?')}] AFTER - "
                f"result: {params.get('result') or '(no effect reported)'}")
    elif atype == 'eye_allocate':
        n = params.get('count', params.get('eyes', '?'))
        text = f"{side} allocates {n} eye(s)"
    elif atype == 'draw_card':
        deck = params.get('deck_type') or params.get('deck') or '?'
        text = f"{side} draws {str(deck).title()}"
    elif atype == 'play_event':
        text = f"{side} plays: {_card_display_name(params)}"
    elif atype == 'crown_aragorn':
        text = (f"{side} CROWNS ARAGORN - Strider becomes King Aragorn "
                f"(Will of the West die spent; leadership 2, +1 FP die "
                f"from next turn)")
    elif atype == 'transform_gandalf':
        text = (f"{side} TRANSFORMS GANDALF THE WHITE at "
                f"{params.get('region', '?')} (Will of the West die "
                f"spent; +1 FP die from next turn)")
    elif atype == 'casualty_allocate':
        # 'count' is the elite-DOWNGRADE choice k (often 0 -> reads as
        # "0 casualties"); 'casualties' (added engine-side) is the real
        # total hits being removed this allocation
        _tot = params.get('casualties')
        _k = params.get('count', 0)
        if _tot is not None:
            text = (f"{side} allocates {_tot} casualt"
                    f"{'y' if _tot == 1 else 'ies'} "
                    f"({_k} as elite downgrade{'s' if _k != 1 else ''})")
        else:
            text = f"{side} allocates casualties (count={_k})"

    if text is None:
        detail = ''
        for key in ('from', 'to', 'region', 'nation', 'card_id', 'count',
                    'companion', 'minion', 'type'):
            if key in params and params[key] is not None:
                val = params[key]
                val = getattr(val, 'name', val)  # enums -> name
                detail += f" {key}={val}"
        # USER 2026-07-28 (TSL chain unreadable): card_option picks
        # carry their substance in the 'choices' dict — show it, or a
        # chained card's second pick is indistinguishable from a no-op.
        ch = params.get('choices')
        if isinstance(ch, dict) and ch:
            # full chain-terminator class (mirrors ai.py's set)
            if any(ch.get(k9) for k9 in
                   ('_tsim_done', '_ffm_done', '_rage_done',
                    '_ents_chain_done', '_naz_done', '_comp_done',
                    '_tsl_done')):
                detail += ' [chain done - no further move]'
            else:
                bits = []
                for k in ('from_region', 'to_region', 'region',
                          'companion', 'companions', 'nation',
                          'n_regular', 'n_elite', 'action'):
                    if ch.get(k) is not None:
                        bits.append('%s=%s' % (k, ch[k]))
                if bits:
                    detail += ' {%s}' % ', '.join(str(b) for b in bits)
        text = f"{side}: {atype}{detail}"

    # Annotation flags some traces carry (NET_PLAYS/HUMAN_DID/AGREE/EVENT)
    if 'AGREE' in params:
        text += " [AGREE]" if params['AGREE'] else " [DISAGREE]"
    return prefix + text
