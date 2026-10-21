"""Acknowledge native turn bookkeeping without choosing cards or game actions.

The client owns the menus and their random results. Human draws are requested
here; the existing AI draw funnel remains their sole AI-side owner. Recovery
retains the ended turn's exact dice until both native pools are observed back.
"""
from dataclasses import dataclass, field


@dataclass
class Recovery:
    turn: int
    pools: dict
    sent: set = field(default_factory=set)
    confirmed: set = field(default_factory=set)
    complete: bool = False
    retirement_sent: set = field(default_factory=set)


@dataclass
class Bookkeeping:
    recovery: object = None
    draws_sent: set = field(default_factory=set)
    rolls_sent: set = field(default_factory=set)
    wait: str = ''


def _journal(server):
    value = getattr(server, '_turn_bookkeeping', None)
    if value is None:
        value = server._turn_bookkeeping = Bookkeeping()
    return value


def capture_pools(driver):
    """Capture all registered dice, including earned dice, before turn reset."""
    pools = {}
    for side, dice in driver.es.dice.items():
        entries = [(int(index), str(pid)) for pid, (owner, index)
                   in driver.view.die_id_map.items()
                   if owner is side and 0 <= int(index) < len(dice)]
        entries.sort()
        pools[side.name] = {
            'count': len(dice), 'pids': [pid for _, pid in entries],
            'complete': [index for index, _ in entries] == list(range(len(dice))),
        }
    return pools


def bind_unrolled_bonus_dice(driver, dump):
    """Bind interchangeable bonus pieces to the unrolled slots actually in play.

    A player can take any spare die of their colour. Before any allocation or
    roll, replace a provisional reinforcement PID with the observed bonus PID.
    Preserve all existing in-play identities and never alter engine dice.
    """
    from game_state import Side
    from emitter import _FP_DIE_IDS, _SH_DIE_IDS
    if driver.phase not in ('draws', 'declare'):
        return
    for side, family, base, prefix in ((Side.FREE_PEOPLES, _FP_DIE_IDS, 4, 'fp'),
                                       (Side.SHADOW, _SH_DIE_IDS, 7, 'sa')):
        dice = driver.es.dice.get(side, ())
        if len(dice) <= base or any(d.current is not None or d.used for d in dice):
            continue
        bindings = [(pid, index) for pid, (owner, index) in driver.view.die_id_map.items()
                    if owner is side]
        if sorted(index for _, index in bindings) != list(range(len(dice))):
            continue
        home = {prefix + ' dice', prefix + ' dead', prefix + ' casualties',
                'hunt box', 'the hunt for the ring'}
        reserve = {prefix + ' reinf', prefix + ' reinforcements'}
        zones = {pid: str(dump.get(pid, '')).strip().lower() for pid in family}
        actual = [pid for pid in family[base:] if zones[pid] in home]
        if len(actual) != len(dice) - base:
            continue
        missing = [(pid, index) for pid, index in bindings
                   if index >= base and pid not in actual]
        if any(pid not in family[base:] or zones.get(pid) not in reserve
               for pid, _ in missing):
            continue
        bound = {pid for pid, _ in bindings}
        replacements = [pid for pid in actual if pid not in bound]
        if len(missing) != len(replacements):
            continue
        for (old, index), pid in zip(sorted(missing, key=lambda row: row[1]), replacements):
            driver.view.bind_die(pid, side, index)
            driver.findings.append(('bonus_die_bound_from_board',
                                    '%s slot %d: %s -> %s' % (side.name, index, old, pid)))


def capture_recovery_pools(driver):
    """The ended turn's dice include results discarded by Day Without Dawn.

    That card compacts the active result list, while the native Recover Dice
    menu returns the discarded pieces too. The prior bonus count is unchanged
    until the turn reset. Preserve this full membership for recovery and any
    subsequent retirement of a dead character's bonus die.
    """
    from game_state import Side
    from emitter import _FP_DIE_IDS
    pools = capture_pools(driver)
    fp = pools.get(Side.FREE_PEOPLES.name)
    count = 4 + int(getattr(driver.es, 'fp_bonus_dice', 0))
    if fp is None or fp['count'] >= count:
        return pools
    registered = set(fp['pids'])
    regions = driver.view.piece_region
    bonuses = list(_FP_DIE_IDS[4:])
    present = [pid for pid in bonuses if pid in registered]
    # Native placement identifies a discarded bonus result; a piece still
    # in Reinforcements has not joined the ended turn's pool.
    for pid in bonuses:
        if (pid not in present and str(regions.get(pid, '')).strip().lower()
                in ('fp dice', 'fp dead', 'fp casualties', 'current action',
                    'hunt box', 'the hunt for the ring')):
            present.append(pid)
    pids = list(_FP_DIE_IDS[:4]) + present
    pools[Side.FREE_PEOPLES.name] = {
        'count': count, 'pids': pids,
        'complete': fp['complete'] and len(pids) == count
                    and registered <= set(pids),
    }
    return pools


def restore_discarded_result_bindings(driver, recovery_pools, result_counts):
    """Rebind a restored unrolled roster after a temporary result removal."""
    from emitter import _FP_DIE_IDS, _SH_DIE_IDS
    from game_state import Side
    for side, family in ((Side.FREE_PEOPLES, _FP_DIE_IDS),
                         (Side.SHADOW, _SH_DIE_IDS)):
        pool = recovery_pools.get(side.name, {})
        if (not pool.get('complete') or pool['count'] <= result_counts[side.name]):
            continue
        count = len(driver.es.dice.get(side, ()))
        pids = list(pool['pids'])
        pids.extend(pid for pid in family if pid not in pids)
        pids = pids[:count]
        if len(pids) != count:
            continue
        driver.view.die_id_map = {
            pid: binding for pid, binding in driver.view.die_id_map.items()
            if binding[0] is not side}
        driver.view.die_id_map.update(
            (pid, (side, index)) for index, pid in enumerate(pids))


def _fresh_board(server):
    from . import client_board
    from .repair_evidence import dump_signature, same_dump
    driver = server.driver
    journal = _journal(server)
    if (driver._feed_seq != len(driver.feed.lines)
            or getattr(driver.feed, '_buf', '')):
        journal.wait = 'the complete native feed'
        return None
    if not driver.view.dump_path:
        journal.wait = 'a complete native board export'
        return None
    try:
        evidence = dump_signature(driver)
        header = evidence['header']
        if header is None or driver._dump_position_hold(header):
            journal.wait = 'a native board export at the current feed position'
            return None
        driver._check_dump_session(header)
        dump = client_board.read_board(driver.view.dump_path)
        if dump is None or not same_dump(driver, evidence):
            journal.wait = 'a stable complete native board export'
            return None
        return dump
    except (OSError, ValueError):
        journal.wait = 'a stable complete native board export'
        return None


def _pool_status(pool, side, dump):
    pids = tuple(pool.get('pids', ()))
    count = pool.get('count')
    if (not pool.get('complete', True) or count is None
            or len(pids) != count or len(set(pids)) != count):
        return 'unknown'
    home = 'fp dice' if side == 'FREE_PEOPLES' else 'sa dice'
    dead = 'fp dead' if side == 'FREE_PEOPLES' else 'sa dead'
    zones = [str(dump.get(pid, '')).strip().lower() for pid in pids]
    if all(zone == home for zone in zones):
        return 'home'
    if all(zone in (home, dead, 'hunt box', 'the hunt for the ring') for zone in zones):
        return 'spent'
    return 'unknown'


def _retirement(server, recovery, dump):
    """Retire only the exact trailing dice removed by this turn's core reset.

    Native Recover Dice also returns dice in Dead. A removed bonus die must
    therefore reach Reinforcements before that menu can safely run again.
    Return it directly from its observed native zone, then wait for the
    board acknowledgment before running either recovery menu.
    """
    driver = server.driver
    current = capture_pools(driver)
    pools = dict(recovery.pools)
    retired = []
    for side, old in recovery.pools.items():
        now = current.get(side, {})
        old_count, count = old.get('count', 0), now.get('count', 0)
        if count >= old_count:
            continue
        old_pids = tuple(old.get('pids', ()))
        if (not old.get('complete', True) or not now.get('complete')
                or len(old_pids) != old_count
                or len(set(old_pids)) != old_count
                or tuple(now['pids']) != old_pids[:count]):
            return None, []
        pools[side] = dict(count=count, pids=old_pids[:count], complete=True)
        retired.extend((side, pid) for pid in old_pids[count:])
    if not retired:
        return pools, []
    remaining = []
    for side, pid in retired:
        prefix = 'fp' if side == 'FREE_PEOPLES' else 'sa'
        zone = str(dump.get(pid, '')).strip().lower()
        if zone in (prefix + ' reinf', prefix + ' reinforcements'):
            continue
        if zone not in (prefix + ' dice', prefix + ' dead', 'hunt box', 'the hunt for the ring'):
            return None, []
        remaining.append((prefix, pid, zone))
    if not remaining:
        return pools, []
    from emitter import render, zone_code_for
    lines = []
    for prefix, pid, zone in remaining:
        if pid in recovery.retirement_sent:
            continue
        lines.append(render('move', {'nick': driver.nick, 'pieces': [pid],
                                    'frm': zone_code_for(zone),
                                    'to': zone_code_for(prefix + ' reinf')}, local=False))
        recovery.retirement_sent.add(pid)
    return None, lines


def _busy(driver, *, turn_start=False):
    import ai
    if (driver.echo_pending() or driver.human_physical_work_pending()
            or driver.deferred is not None or driver.chat_deferred is not None
            or getattr(driver, '_human_casualty_prefix', None)
            or getattr(driver, '_chat_attack_window', None)
            or driver._human_window_open()
            or getattr(driver.seg, 'window', None) is not None
            or (driver.pending_buf is not None and driver.pending_buf.records)):
        return True
    pending = ai.describe_pending(driver.es)
    allowed = ('declare', 'discard') if turn_start else ()
    if pending is not None and pending['kind'] not in allowed:
        return True
    return any(getattr(driver.es, name, None) for name in (
        'pending_combat', 'pending_hunt', 'pending_guide_change',
        'pending_card_option', 'pending_card_followup', '_pending_hunt_damage',
        '_reveal_place_due', '_hunt_reveal_pending', '_fellowship_tail_phase'))


def request_recovery(server, request=None):
    """Join the existing recover_ai path; never issue a second pool request."""
    driver = server.driver
    journal = _journal(server)
    turn = int(driver.es.turn)
    if driver.game_over is not None:
        return []
    request = dict(request or {})
    target = int(request.get('turn', turn))
    if target != turn:
        return []
    current = journal.recovery
    if current is None or current.turn != target:
        pools = request.get('pools') or capture_pools(driver)
        pools = {side: dict(pool) for side, pool in pools.items()}
        if 'pids' in request and not request.get('pools'):
            pools[server.ai_side.name] = {
                'pids': list(request['pids']), 'count': request['count'],
                'complete': True,
            }
        current = journal.recovery = Recovery(target, pools)
    if current.complete:
        server._recovery_request = None
        return []
    if getattr(server, '_recovery_dispatched_turn', None) == target:
        current.sent.add(server.ai_side.name)
    server._recovery_request = {'turn': target, 'pools': current.pools}
    return flush_recovery(server)


def flush_recovery(server):
    """Return at most one native menu, then wait for physical confirmation."""
    driver = server.driver
    journal = _journal(server)
    recovery = journal.recovery
    if recovery is None or recovery.complete or driver.game_over is not None:
        return []
    if getattr(server, '_resync_quiet', False):
        journal.wait = 'the current board resync to finish'
        return []
    turn = int(driver.es.turn)
    if turn not in (recovery.turn - 1, recovery.turn):
        journal.wait = 'recovery belongs to a different turn'
        return []
    starting = turn == recovery.turn
    if (starting and driver.phase not in ('draws', 'declare')):
        journal.wait = 'the turn-start recovery window'
        return []
    if _busy(driver, turn_start=starting):
        journal.wait = 'the current decision and committed board changes'
        return []
    dump = _fresh_board(server)
    if dump is None:
        return []
    order = (server.human_side.name, server.ai_side.name)
    pools = recovery.pools
    if starting:
        pools, lines = _retirement(server, recovery, dump)
        if pools is None:
            journal.wait = 'retired bonus dice to reach their own Reinforcements'
            return lines
    states = {side: _pool_status(pools.get(side, {}), side, dump)
              for side in order}
    # Validate both memberships before issuing either pool's native menu.
    if any(value == 'unknown' for value in states.values()):
        journal.wait = 'all registered dice in their own pool or spent/Hunt boxes'
        return []
    recovery.confirmed = {side for side, value in states.items() if value == 'home'}
    if server.ai_side.name in recovery.confirmed:
        server._recovery_completed_turn = recovery.turn
    if len(recovery.confirmed) == 2:
        recovery.complete = starting
        server._recovery_request = None if starting else {'turn': recovery.turn}
        journal.wait = '' if starting else 'the native recovery feed to advance the turn'
        return []
    # A first recovery command must be acknowledged before the other pool.
    if recovery.sent - recovery.confirmed:
        journal.wait = 'the requested native Recover Dice acknowledgment'
        return []
    if not starting and recovery.confirmed:
        journal.wait = 'the native recovery feed to advance the turn'
        return []
    side = next(side for side in order if side not in recovery.confirmed)
    recovery.sent.add(side)
    if side == server.ai_side.name:
        server._recovery_dispatched_turn = recovery.turn
    command = 'frecover' if side == 'FREE_PEOPLES' else 'srecover'
    journal.wait = 'the requested native Recover Dice acknowledgment'
    return ['!CALL menu ' + command]


def draws_ready(server):
    """Both current pools, including newly earned dice, must be back first."""
    driver = server.driver
    journal = _journal(server)
    if (driver.game_over is not None or getattr(driver, 'spectator', False)
            or getattr(server, '_resync_quiet', False)
            or driver.phase not in ('draws', 'declare')
            or _busy(driver, turn_start=True)):
        return False
    # CardDecoder's native claim format contains 24 positions in each deck.
    # Do not start a random draw while either claimed hand is undecodable.
    decoder = getattr(getattr(server, 'cards', None), 'decoder', None)
    positions = set(range(24))
    if any(set(getattr(decoder, name, {})) != positions
           for name in ('fp_char_deck', 'fp_strat_deck', 'sa_char_deck', 'sa_strat_deck')):
        journal.wait = 'both native hand claims and complete card commitment tables'
        return False
    recovery = journal.recovery
    if recovery is not None and (int(driver.es.turn) != recovery.turn
                                or not recovery.complete):
        journal.wait = 'both native Recover Dice acknowledgments'
        return False
    dump = _fresh_board(server)
    if dump is None:
        return False
    bind_unrolled_bonus_dice(driver, dump)
    pools = capture_pools(driver)
    if any(_pool_status(pools.get(side.name, {}), side.name, dump) != 'home'
           for side in (server.human_side, server.ai_side)):
        journal.wait = 'both complete current dice pools, including earned dice'
        return False
    journal.wait = ''
    return True


def roll_pools(server):
    """Roll the exact unrolled dice once, adopting Java's result before FP rolls."""
    from game_state import Side
    from emitter import FP_DICE_POOL, SH_DICE_POOL
    driver = server.driver
    journal = _journal(server)
    if (driver.phase != 'roll' or server._open_step is not None
            or server._step_queue or _busy(driver)):
        return []
    dump = _fresh_board(server)
    if dump is None:
        return []
    turn = int(driver.es.turn)
    journal.rolls_sent = {key for key in journal.rolls_sent if key[0] == turn}
    for side, zone, home in ((Side.SHADOW, SH_DICE_POOL, 'sa dice'),
                             (Side.FREE_PEOPLES, FP_DICE_POOL, 'fp dice')):
        missing = [i for i, die in enumerate(driver.es.dice.get(side, []))
                   if not die.used and die.current is None]
        if not missing:
            continue
        by_index = {int(index): str(pid) for pid, (owner, index)
                    in driver.view.die_id_map.items() if owner is side}
        pids = [by_index.get(index) for index in missing]
        if (None in pids or len(set(pids)) != len(pids)
                or any(str(dump.get(pid, '')).lower() != home for pid in pids)):
            journal.wait = 'the unrolled %s dice in their native pool' % side.name
            return []
        journal.wait = 'the native %s action-dice roll acknowledgment' % side.name
        if any((turn, pid) in journal.rolls_sent for pid in pids):
            return []
        journal.rolls_sent.update((turn, pid) for pid in pids)
        return [('line', '!CALL roll_action_dice ' + zone + ' ' + ' '.join(pids))]
    # Empty/all-allocated pools have no roll event to advance the phase.
    driver.phase = 'actions'
    driver.last_actor = Side.SHADOW
    journal.wait = ''
    return []


def poll(server):
    """Driver-style items for the server's existing dispatch/AI-step queue."""
    driver = server.driver
    journal = _journal(server)
    if driver.game_over is not None or getattr(driver, 'spectator', False):
        return []
    if getattr(server, '_resync_quiet', False):
        journal.wait = 'the current board resync to finish'
        return []
    if driver.phase == 'roll':
        return roll_pools(server)
    if (driver.phase == 'actions' and server._open_step is None
            and not server._step_queue and not _busy(driver)
            and driver._turn_over_pending()):
        # Immediate ring outcomes already end play. The end-turn result must
        # also be settled before requesting any physical dice recovery.
        if driver.check_victory(end_of_turn=False) is not None:
            return []
        target = int(driver.es.turn) + 1
        if journal.recovery is None or journal.recovery.turn != target:
            import ai
            import game_logic as gl
            # Ask the same core boundary that the native acknowledgment uses.
            # A nonterminal probe must not reset live dice or charge inactivity
            # before that acknowledgment; an existing request needs no re-probe.
            probe = ai.clone_state(driver.es)
            if gl.begin_turn_reset(probe, target, deal_cards=False, bonus_dice=False):
                # Reuse the live settlement and its final corruption emission.
                # No native recover has occurred, so this ledger entry has no
                # rendered recovery record. The terminal core leaves pools,
                # hands, decks and the turn number intact.
                return driver._on_recover({})
            journal.recovery = Recovery(target, capture_recovery_pools(driver))
        server._recovery_request = {'turn': target, 'pools': journal.recovery.pools}
        return [('line', line) for line in flush_recovery(server)]
    if journal.recovery is not None:
        lines = flush_recovery(server)
        if lines:
            return [('line', line) for line in lines]
    if driver.phase != 'draws' or not draws_ready(server):
        return []
    turn = int(driver.es.turn)
    journal.draws_sent = {key for key in journal.draws_sent if key[0] == turn}
    draws = []
    awaiting = []
    for deck, suffix in (('character', 'c'), ('strategy', 's')):
        key = (turn, server.human_side.name, deck)
        if driver.turn_draws.get((server.human_side, deck), 0):
            continue
        if driver._deck_depleted(server.human_side, deck):
            continue
        awaiting.append(deck)
        if key not in journal.draws_sent:
            journal.draws_sent.add(key)
            prefix = 'f' if server.human_side.name == 'FREE_PEOPLES' else 's'
            draws.append(('line', '!CALL menu ' + prefix + 'draw' + suffix))
    if awaiting:
        # These compulsory draws are independent. Send both native menu calls
        # in one reply; the normal feed still acknowledges each separately.
        journal.wait = 'the native %s turn-start draw acknowledgment' % ' and '.join(awaiting)
        return draws
    # With both decks depleted no physical draw can trigger the old handoff.
    waived = driver._depleted_deck_draw_waivers(server.human_side)
    if waived is not None:
        journal.wait = ''
        return waived + driver._ai_turn_start_draws()
    return []
