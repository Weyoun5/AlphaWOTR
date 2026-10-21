"""Select a legal gesture result by its completed public native board.

This is a commit-time witness, not a rule engine or a board adopter. Candidate
states and their recorded randomness belong to the matcher and are never changed.
Internal action ordering, dictionary spelling and private bookkeeping are not
physical choices. A shorter proven path retains the first real next decision.
"""
from dataclasses import dataclass, field
import os


@dataclass
class EndpointResult:
    status: str
    candidates: list = field(default_factory=list)
    reason: str = ''
    evidence: dict = field(default_factory=dict)


def completed(window):
    """A player completion signal, not a pause between native records."""
    return bool(window is not None and (
        (getattr(window, 'closed', None) in ('die_killed', 'die_to_hunt')
         and not getattr(window, '_borrowed_payment_context', False))
        or getattr(window, '_staged_card_closed', False)
        or getattr(window, '_casualty_closed', False)
        or getattr(window, '_victor_close_record', None) is not None))


def _freeze(value):
    """Canonicalize explicit public values, never an entire engine object."""
    if isinstance(value, dict):
        return tuple(sorted((str(getattr(k, 'name', k)), _freeze(v))
                            for k, v in value.items()))
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(v) for v in value), key=repr))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if hasattr(value, 'name'):
        return str(value.name)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError('unsupported public endpoint value')


def public_key(state):
    """All modeled public board surfaces; no hands, decks or pending payloads.

    Hidden card identities/order and engine continuation metadata deliberately
    do not distinguish two physical endpoints. Hand *sizes* remain public.
    Duplicate region aliases are malformed, rather than silently merged.
    """
    from board import sanitize
    from .client_board import proj_board
    from .combat_recovery import _engine_counts
    regions = [sanitize(str(k)) for k in state.armies]
    if len(regions) != len(set(regions)):
        raise ValueError('duplicate region aliases in candidate')
    projection = proj_board(state)
    counts, _labels, _owners = _engine_counts(state)
    projection['armies'] = dict(counts)
    projection.pop('named_leaders', None)  # counts includes every named class
    projection['hands'] = {side: len(hand.hand) for side, hand in state.hands.items()}
    fellowship = list(projection['fellowship'])
    fellowship[-1] = sanitize(str(fellowship[-1]))
    projection['fellowship'] = fellowship
    table = projection['table']
    projection['table'] = (table[0], table[1],
                           tuple((side, tuple(sorted(cards))) for side, cards in table[2]))
    projection['activation'] = state.political.active
    projection['control'] = {side: sorted(sanitize(str(k)) for k in keys)
                             for side, keys in state.enemy_controlled.items()}
    projection['sieges'] = sorted(sanitize(str(k)) for k in state.besieged_strongholds)
    projection['casualties'] = {nation: {kind: n for kind, n in kinds.items() if n}
                                for nation, kinds in state.fp_casualties.items()
                                if any(kinds.values())}
    projection['hunt_dice'] = len(state.hunt_box)
    projection['turn'] = state.turn
    return _freeze(projection)


def read_complete(driver):
    """Read one current stable export, without quiescence or pending gates.

    A pending rule choice is compatible with a completed physical decision.
    Never refresh/adopt the engine or infer missing pieces as casualties.
    """
    from . import client_board, client_protocol, recovery_board, repair_evidence
    path = getattr(driver.view, 'dump_path', None)
    if not path or not os.path.isfile(path):
        raise ValueError('waiting for a complete native board export')
    if int(getattr(driver, '_feed_seq', 0)) < len(driver.feed.lines):
        raise ValueError('waiting for the rest of the native gesture batch to finish routing')
    if getattr(driver.feed, '_buf', b''):
        raise ValueError('waiting for the last native feed record to finish writing')
    signature = repair_evidence.dump_signature(driver)
    offset = getattr(driver.feed, '_offset', None)
    if offset is not None and offset != signature['feed_stat'][0]:
        raise ValueError('waiting for newly written native feed records to finish routing')
    header = signature['header']
    if header is None or not int(header.get('session', 0) or 0):
        raise ValueError('waiting for a native board export with its session and feed position')
    nonce = getattr(driver, 'session_nonce', None)
    if nonce is not None and not client_protocol.session_matches(header, nonce):
        raise ValueError('native board export belongs to a different session')
    if not client_protocol.dump_at_feed_position(header, len(driver.feed.lines)):
        raise ValueError('waiting for the native board export at the current feed position')
    dump = client_board.read_board(path, retries=2, delay=0)
    if dump is None:
        raise ValueError('waiting for a complete stable native board export')
    recovery_board.validate(driver, dump)
    import log_replay
    required = {'247', '248', '249', '250', '251', '252'}
    required.update(str(pid) for pid in log_replay.POLITICAL_MARKERS)
    if required - set(dump):
        raise ValueError('native board export is missing public markers')
    markers = client_protocol.read_marker_states(path)
    if not repair_evidence.same_dump(driver, signature):
        raise ValueError('native board or feed changed during comparison')
    from .corsairs_presentation import projected as project_corsairs
    return project_corsairs(driver, dump), markers, signature


def _granted(driver, state, dump):
    """Use the census's existing bonus-die shield for this candidate state."""
    from types import SimpleNamespace
    method = getattr(type(driver), '_granted_die_pids', None)
    if method is None:
        return None
    return method(SimpleNamespace(es=state, view=driver.view), dump)


def _rows(driver, state, dump, markers):
    import ai
    from . import client_board
    rows = client_board.diff_engine_vs_dump(
        state, dump, dynamic_types=driver.view.dynamic_types, ai_module=ai,
        initial_dump=getattr(driver, '_initial_dump', None),
        granted_pids=_granted(driver, state, dump),
        revoked_pids=getattr(driver, '_revoked_die_pids', None),
        marker_states=markers,
        pid_to_cid={str(pid): str(cid) for cid, pid in driver.view.card_pid_map.items()})
    if rows is None or any('check unavailable' in row for row in rows):
        raise ValueError('native public board comparison is incomplete')
    # Only public counts: native PIDs do not disclose hidden hand identities.
    rows.extend(client_board.card_census(state, dump))
    from .combat_recovery import _permanent_losses_match
    from game_state import Side
    if not _permanent_losses_match(state, Side.FREE_PEOPLES, dump, driver.view.dynamic_types):
        rows.append('permanent casualties: engine differs from the native FP casualty pile')
    # Guide identity has its own physical box, beyond the census's fellowship
    # membership check. Do not select an unvoiced guide change merely because
    # both companions are still in the fellowship.
    from board import sanitize
    import log_replay
    guide_pids = [pid for pid, zone in dump.items()
                  if sanitize(str(zone)) in ('guide box', 'guide of the fellowship')
                  and str(pid) in log_replay.COMPANIONS]
    if len(guide_pids) == 1:
        wanted = getattr(getattr(state.fellowship, 'guide', None), 'name', None)
        if wanted not in log_replay.names_for_pid(str(guide_pids[0])):
            rows.append('guide: engine guide differs from the native guide box')
    return list(rows)


def _family(row):
    for prefix, family in (
            ('units ', 'armies'), ('leaders ', 'leaders'), ('leader nation ', 'leaders'),
            ('nazgul ', 'named'), ('minion ', 'named'), ('companion ', 'named'),
            ('track progress:', 'fs_progress'), ('track corruption:', 'corruption'),
            ('fellowship visibility:', 'hidden_marker'), ('political ', 'political'),
            ('siege ', 'sieges'), ('control ', 'control'), ('vp ', 'victory'),
            ('deck ', 'draws'), ('hand size ', 'draws')):
        if row.startswith(prefix):
            return family
    return None


def _owed_rows(chain, window, baseline, rows, post=None):
    """Only existing matcher-authorized, unvoiced automatic consequences.

    The native surface must still match the BEFORE state for that exact row.
    An arbitrary partial endpoint or an unrelated preexisting mismatch cannot
    be excused as follow-through. Exact completed endpoints always win.
    """
    from .gesture_match import follow_through_for, window_facts
    allowed = set(follow_through_for(chain, live=True))
    # A revealing Hunt result changes the engine's visibility before the
    # human places and flips the pawn. In particular 0r opens reveal_place
    # immediately, without a damage choice in between. Retain that pending
    # choice while accepting its still-hidden physical marker.
    if (post is not None and getattr(post, '_reveal_place_due', False)
            and not post.fellowship.hidden):
        allowed.add('hidden_marker')
    # Captures and score markers are consequences, never separately chosen
    # moves. Political changes caused by a completed movement/attack likewise
    # belong to the post-commit marker renderer, not the human gesture grammar.
    allowed.update(('control', 'victory'))
    if any(action.action_type in (
            'army_move', 'army_attack', 'victor_advance', 'companion_move',
            'companion_separate', 'crown_aragorn', 'transform_gandalf')
            for action in chain):
        allowed.add('political')
    if any(action.action_type in ('casualty_allocate', 'siege_trim') for action in chain):
        # Wiping the last army unit removes its leaders and can capture its
        # settlement. Cleanup and the capture's political consequence follow
        # the committed loss; their markers cannot be prerequisites for it.
        allowed.update(('leaders', 'named', 'political'))
    facts = window_facts(window)
    voiced = {name for name in allowed if bool(getattr(facts, name, None))}
    if facts.fel_flips:
        voiced.add('hidden_marker')
    if facts.activations:
        voiced.add('political')
    allowed -= voiced
    baseline_ids = {row.split(':', 1)[0] for row in baseline}
    return [row for row in rows if _family(row) in allowed
            and row.split(':', 1)[0] not in baseline_ids]


def filter(driver, window, candidates):
    """Return matched one-row candidates, or a specific retryable reason.

    Inputs must already be legal matcher results for the entire gesture batch.
    No search, action application, RNG consumption or state mutation occurs.
    """
    from . import gesture_match, repair_evidence
    candidates = list(candidates or ())
    evidence = {'candidate_count': len(candidates)}
    if not candidates:
        return EndpointResult('hold', [], 'no completed legal endpoint is available', evidence)
    try:
        dump, markers, signature = read_complete(driver)
        evidence.update(dump=signature['header'].get('dump'),
                        feed_position=signature['header'].get('feed_paragraphs'),
                        piece_count=len(dump))
        baseline = _rows(driver, driver.es, dump, markers)
        survivors = []
        for index, row in enumerate(candidates):
            key = public_key(row['post'])
            execution = gesture_match._typed_value_key(row.get('execution'))
            if row.get('execution') is None or execution is None:
                continue
            rows = _rows(driver, row['post'], dump, markers)
            # Deck lengths are model fiction; they are measured for diagnostics
            # but cannot disprove an otherwise completed physical action.
            rows = [item for item in rows if not item.startswith('deck ')]
            owed = _owed_rows(row['chain'], window, baseline, rows, row['post'])
            unresolved = [item for item in rows if item not in owed]
            if not unresolved:
                survivors.append((bool(owed), len(owed), key, execution, row, index))
        if not repair_evidence.same_dump(driver, signature):
            raise ValueError('native board or feed changed during comparison')
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        return EndpointResult('unavailable', [], str(exc), evidence)
    evidence['matching_count'] = len(survivors)
    if not survivors:
        return EndpointResult('mismatch', [],
                              'waiting for the completed native board to match the recorded action', evidence)
    exact = [item for item in survivors if not item[0]]
    if exact:
        survivors = exact
    keys = {(item[2], item[3]) for item in survivors}
    if len(keys) != 1:
        reason = ('waiting for the recorded random outcome at this decision'
                  if len({item[3] for item in survivors}) > 1 else
                  'waiting for the remaining public decision to be represented on the native board')
        return EndpointResult('hold', [], reason, evidence)
    def rank(item):
        chain = item[4]['chain']
        # Typed action parameters provide a stable tie-break without publishing
        # their possibly private card data or depending on input enumeration.
        return (len(chain), repr(gesture_match._typed_value_key(chain)), item[5])
    chosen = min(survivors, key=rank)
    evidence.update(equivalent_count=len(survivors), chain_length=len(chosen[4]['chain']),
                    deferred_rows=chosen[1], exact=not chosen[0])
    return EndpointResult('matched', [chosen[4]],
                          'completed native board identifies this legal decision endpoint', evidence)


def recover_visible_choice(driver, pending=None, window=None):
    """Prove a visible legal choice directly, without reconstructing drags.

    Used at an explicit completion or Resync. This path cannot infer hidden
    card identities, select an unexpressed no-op, or request/sample randomness.
    The normal commit still applies the chosen rule action exactly once.
    """
    import ai
    import copy
    from .combat_recovery import _preserve_randomness
    from .randomness import NeedClientCall
    from . import gesture_match as gm
    if getattr(driver, 'spectator', False) or not getattr(driver.view, 'dump_path', None):
        return None
    pending = ai.describe_pending(driver.es) if pending is None else pending
    if not pending or pending['side'] is not driver.human_side:
        return None
    visible_actions = {
        'casualty_allocate', 'combat_retreat', 'retreat_into_siege',
        'victor_advance', 'siege_trim', 'choose_guide', 'card_option',
    }
    if not any(a.action_type in visible_actions for a in pending['actions']):
        return None
    try:
        read_complete(driver)
    except (OSError, ValueError):
        return None
    before_key = public_key(driver.es)
    candidates = []
    for action in pending['actions']:
        if action.action_type not in visible_actions:
            continue
        with _preserve_randomness(driver.rctl):
            try:
                driver.rctl.root_attempt()
                positions = copy.deepcopy(driver.rand._feed_pos)
                with driver.rctl.apply_cm():
                    post = ai.apply_action(driver.es, action)
                if driver.rand._feed_pos != positions:
                    continue
            except NeedClientCall:
                continue  # the original dice/card/tile request retains ownership
        if (public_key(post) == before_key or any(
                post.hands[side].hand != hand.hand for side, hand in driver.es.hands.items())):
            continue
        candidates.append(dict(chain=[action], post=post, execution=()))
    if not candidates:
        return None
    original = window if window is not None else driver.pending_buf
    witness = copy.copy(original) if original is not None else driver._new_phase_window()
    witness.records = list(getattr(original, 'records', ()))
    result = filter(driver, witness, candidates)
    if result.status != 'matched':
        return None
    chosen = result.candidates[0]
    return gm.MatchResult('match', chain=chosen['chain'], post=chosen['post'],
                          candidates=[chosen], note=result.reason)
