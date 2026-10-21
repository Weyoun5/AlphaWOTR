"""Recover visible combat choices without replacing the retained battle.

The caller supplies a stable, siege/temporary-figure projected dump.  This
module constructs missing physical evidence, then asks the ordinary matcher.
It never commits, changes a hand, resets randomness, or selects an invisible
decision.  Existing gesture records remain authoritative, including native
Elite-downgrade controls and their ordering.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import copy


@dataclass
class Recovery:
    status: str
    note: str
    window: object = None
    match: object = None
    added: int = 0


@contextmanager
def _preserve_randomness(controller):
    if controller is None:
        yield
        return
    snapshot = randomness_snapshot(controller)
    try:
        yield
    finally:
        restore_randomness(controller, snapshot)


def randomness_snapshot(controller):
    """Keep feeds, the exact outstanding request, and step/seed lifecycle."""
    if controller is None:
        return None
    owner = controller.r
    fields = vars(owner).copy()
    for key in ('_feed', '_feed_pos'):
        fields[key] = copy.deepcopy(fields[key])
    for key, value in list(fields.items()):
        if isinstance(value, list):
            fields[key] = list(value)
    return fields, owner._rng.getstate()


def restore_randomness(controller, snapshot):
    if snapshot is None:
        return
    fields, rng_state = snapshot
    owner = controller.r
    vars(owner).clear()
    vars(owner).update(fields)
    owner._rng.setstate(rng_state)


_PHYSICAL_ACTIONS = frozenset((
    'casualty_allocate', 'combat_retreat', 'retreat_into_siege',
    'victor_advance', 'siege_trim',
))
_DISPLAY = {'ELVES': 'Elven', 'DWARVES': 'Dwarven', 'GONDOR': 'Gondor',
            'THE_NORTH': 'North', 'ROHAN': 'Rohan', 'ISENGARD': 'Isengard',
            'SAURON': 'Sauron', 'SOUTHRON_EASTERLINGS': 'S&E'}


def observed_political_steps(driver, pre, post, actions, gesture):
    """Carry the measured source and any already performed recovered steps."""
    if (not getattr(driver, '_resync_commit_transaction', False)
            or not getattr(pre, 'pending_combat', None) or not actions
            or any(action.action_type not in _PHYSICAL_ACTIONS for action in actions)
            or gesture is None
            or not any(record.get('_combat_recovery') for record in gesture.records)):
        return []
    import log_replay as lr
    from . import client_board
    path = getattr(driver.view, 'dump_path', None)
    dump = client_board.read_board(path) if path else None
    if not dump:
        return []
    # A genuine settlement capture still owns its nation's political
    # consequence at the At War cap. The legal transition's new control
    # entries prove which nation is affected; unrelated marker drift is
    # deliberately left to ordinary board adoption.
    import ai
    import board
    captured_nations = set()
    for side, regions in post.enemy_controlled.items():
        for region in set(regions) - set(pre.enemy_controlled.get(side, ())):
            meta = board.regions.get(board.sanitize(region), {})
            if any(meta.get(key) for key in ('is_town', 'is_city', 'is_stronghold')):
                nation = ai._region_nation(board.sanitize(region))
                if nation is not None:
                    captured_nations.add(nation)
    covered = []
    for pid, nation in lr.POLITICAL_MARKERS.items():
        source = pre.political.boxes.get(nation)
        target = post.political.boxes.get(nation)
        shown = {'Political %d' % (3 - box): box for box in range(4)}.get(dump.get(str(pid)))
        if (source is not None and target is not None
                and (source != target or nation in captured_nations)):
            # The recovered rule consequence is authoritative, but its
            # physical correction must start at the measured marker. Keep
            # unknown sources explicit so the sweep cannot guess from pre.
            covered.append(('recovered_political_source',
                            {'piece': str(pid), 'box': shown}))
        if (source is None or target is None or shown is None
                or not 0 <= source < shown <= target <= 3):
            continue
        # Activation is a separate native face operation. A passive face
        # cannot invalidate a track move that is already visibly complete.
        for step in range(source, shown):
            covered.append((('political_91', 'political_96', 'political_101')[step],
                            {'piece': str(pid), 'frm': 'Game.%d' % (2000 + step),
                             'to': 'Game.%d' % (2001 + step), '_recovered_political': True}))
    return covered


def _physical_action(state, action):
    if action.action_type in _PHYSICAL_ACTIONS:
        return True
    from .casualty_randomness import is_round_unit_payment
    from .onslaught_gestures import is_payment_action
    effect = (state.pending_combat.get('effect_choice') or {}).get('effect')
    return (is_round_unit_payment(state, action)
            or (effect == 'onslaught_payment' and is_payment_action(action)))


def _token(nation, kind):
    return '%s %s' % (_DISPLAY[nation.name], kind.title())


def _key(token):
    import rendered_replay as rr
    piece = rr.parse_piece(token)
    if not piece or piece[0] not in ('unit', 'leader', 'nazgul', 'named'):
        raise ValueError('unrecognized army figure: %s' % token)
    return tuple(piece[:3])


def _zone(zone):
    from .gesture_match import zone_key
    value = zone_key(zone)
    if value[0] in ('region', 'stronghold'):
        return value[1] + (' stronghold' if value[0] == 'stronghold' else '')
    return None


def _engine_counts(state):
    import ai
    import emitter
    import game_state as gs
    from board import sanitize
    counts, labels, owners = Counter(), {}, {}
    for region, army in state.armies.items():
        for figure in list(army.units) + list(army.leaders):
            if isinstance(figure, gs.Unit):
                token = _token(figure.nation, figure.kind.value)
                side = ai._side_of_nation(figure.nation)
            elif getattr(figure, 'name', None):
                token = figure.name
                side = (gs.Side.SHADOW if type(figure).__name__ in ('Minion', 'ShadowMinion')
                        else gs.Side.FREE_PEOPLES)
            elif isinstance(figure, gs.Nazgul):
                token, side = 'Nazgul', gs.Side.SHADOW
            else:
                token = _token(figure.nation, 'leader')
                side = ai._side_of_nation(figure.nation)
            key = _key(token)
            zone = sanitize(str(region))
            if emitter.garrison_side_of(state, zone) == side.name:
                zone += ' stronghold'
            counts[(zone, key)] += 1
            labels[key], owners[key] = token, side
    return counts, labels, owners


def _dump_counts(dump, dynamic_types, labels, owners):
    import ai
    import game_state as gs
    import log_replay as lr
    from .client_board import unit_type
    from .shadow_siege_slots import GENERIC_SLOTS
    from board import sanitize
    counts = Counter()
    for pid, raw_zone in dump.items():
        nation, kind = unit_type(pid, dynamic_types)
        if kind not in ('regular', 'elite', 'leader', 'nazgul', 'companion', 'minion'):
            continue
        if sanitize(str(raw_zone)) in GENERIC_SLOTS:
            raise ValueError('generic stronghold needs its observed origin')
        zone = _zone(raw_zone)
        if zone is None:
            if kind in ('regular', 'elite', 'leader', 'nazgul') and sanitize(str(raw_zone)) not in {
                    'fp reinf', 'sa reinf', 'fp reinforcements', 'sa reinforcements',
                    'fp dead', 'sa dead', 'fp casualties', 'sa casualties',
                    'dead', 'casualties', 'reinforcements'}:
                raise ValueError('an army figure is parked outside the board and casualty/reserve boxes')
            continue
        if kind in ('companion', 'minion'):
            # The board does not reveal the face of a dual-sided character.
            # Preserve the living engine identity; never infer a promotion.
            aliases = [_key(name) for name in lr.names_for_pid(str(pid))]
            known = [key for key in aliases if key in labels]
            if len(known) != 1:
                raise ValueError('board character has no unique retained identity')
            key = known[0]
        else:
            if kind == 'nazgul':
                token, side = 'Nazgul', gs.Side.SHADOW
            else:
                token, side = _token(nation, kind), ai._side_of_nation(nation)
            key = _key(token)
            labels[key], owners[key] = token, side
        counts[(zone, key)] += 1
    return counts


def _permanent_losses_match(state, side, dump, dynamic_types):
    """FP reserve returns and permanent deaths cannot be interchanged."""
    if side.name != 'FREE_PEOPLES':
        return True
    import ai
    from board import sanitize
    from .client_board import unit_type
    observed = Counter()
    for pid, zone in dump.items():
        if sanitize(str(zone)) not in ('fp dead', 'fp casualties', 'dead', 'casualties'):
            continue
        nation, kind = unit_type(pid, dynamic_types)
        if kind in ('regular', 'elite') and ai._side_of_nation(nation) is side:
            observed[(nation, kind)] += 1
    expected = Counter({(nation, kind): number
                        for nation, rows in (state.fp_casualties or {}).items()
                        for kind, number in rows.items() if kind in ('regular', 'elite') and number})
    return observed == expected


def _record_delta(records):
    """Only physical army records may be supplemented from a board."""
    import rendered_replay as rr
    result = Counter()
    for record in records:
        op = record.get('op')
        if op == 'roll_num':
            continue
        if op == 'kill':
            tokens = record.get('pieces') or ()
            zones = [(_zone(record.get('zone')), -1)]
        elif op == 'move':
            obj = record.get('obj', '')
            tokens = rr.split_group(obj[7:-1]) if obj.startswith('group (') and obj.endswith(')') else [obj]
            zones = [(_zone(record.get('from')), -1), (_zone(record.get('to')), 1)]
        else:
            raise ValueError('retained non-army evidence needs its original decision')
        for token in tokens:
            key = _key(token)
            count = rr.parse_piece(token)[3]
            for zone, sign in zones:
                if zone is not None:
                    result[(zone, key)] += sign * count
    return result


def _missing_records(state, side, counts, target, labels, owners, baseline):
    """Express only one-source/one-destination moves or current casualties."""
    import emitter
    missing = Counter({key: target[key] - counts[key] for key in set(target) | set(counts)
                       if target[key] != counts[key]})
    if any(owners[key] is not side for _zone_name, key in missing):
        raise ValueError('another player has unmatched physical army changes')
    losses, gains = defaultdict(list), defaultdict(list)
    for (zone, key), number in missing.items():
        (losses if number < 0 else gains)[key].append([zone, abs(number)])
    records = []
    for key in sorted(set(losses) | set(gains), key=repr):
        negative, positive = losses[key], gains[key]
        if len(negative) > 1 or len(positive) > 1:
            raise ValueError('board does not identify a unique movement path')
        token = labels[key]
        if negative and positive:
            moved = min(negative[0][1], positive[0][1])
            for _ in range(moved):
                records.append(dict(op='move', obj=token, **{'from': negative[0][0], 'to': positive[0][0]}))
            negative[0][1] -= moved
            positive[0][1] -= moved
        for zone, count in negative:
            # Express only the visible removal. The ordinary matcher must
            # prove its hit cost, any replacement, and retained native intent;
            # FP permanent-loss equality below rejects a reserve return.
            for _ in range(count):
                records.append(dict(op='kill', pieces=[token], zone=zone))
        for zone, count in positive:
            if not count:
                continue
            context = emitter.casualty_context(state)
            effect = (state.pending_combat.get('effect_choice') or {}).get('effect')
            payment = effect in ('onslaught_payment', 'pre_combat_casualty',
                                 'relentless_assault_payment')
            if (not context and not payment) or key[0] != 'unit' or key[2] != 'REGULAR':
                raise ValueError('unexplained army addition is not a casualty replacement')
            elite_key = (key[0], key[1], 'ELITE')
            reductions = max(0, baseline[(zone, elite_key)] - target[(zone, elite_key)])
            if reductions < count:
                raise ValueError('Regular addition has no visible Elite replacement')
            source = 'FP Reinforcements' if side.name == 'FREE_PEOPLES' else 'SA Reinforcements'
            # Source is merely off-board here; engine supply rules still
            # choose and charge the actual replacement source.
            for _ in range(count):
                records.append(dict(op='move', obj=token, **{'from': source, 'to': zone}))
    for record in records:
        record.update(norm='', _combat_recovery=True)
    return records


def plan(driver, dump):
    """Return a proved normal-match candidate, or a non-mutating hold.

    Call after refreshing a stable dump and retrying existing feed evidence.
    A successful result can enter ``_commit_chain`` with ``window`` exactly
    like ``_try_pending``.  It must remain inside the driver's recovery
    transaction: the engine and feed must be restored if commit fails.
    """
    import ai
    from . import gesture_match as gm
    state = driver.es
    if not isinstance(getattr(state, 'pending_combat', None), dict):
        return Recovery('hold', 'no retained battle decision can be reconstructed')
    pending = ai.describe_pending(state)
    if not pending or pending['side'] is not driver.human_side:
        return Recovery('hold', 'the current decision belongs to the engine player')
    if not any(_physical_action(state, action) for action in pending['actions']):
        return Recovery('hold', 'the current combat choice needs its explicit answer')
    if not dump:
        return Recovery('hold', 'no stable physical board')
    old = driver.pending_buf
    window = copy.copy(old) if old is not None else driver._new_phase_window()
    window.records = list(getattr(old, 'records', ()) or ())
    from .decision_endpoint import recover_visible_choice
    direct = recover_visible_choice(driver, pending, window)
    if direct is not None and all(_physical_action(state, a) for a in direct.chain):
        # The board can prove the choice even when every movement paragraph
        # already arrived and the old matcher merely disliked its order.
        return Recovery('match', 'completed board proves the retained combat choice',
                        window=window, match=direct)
    try:
        before, labels, owners = _engine_counts(state)
        physical = _dump_counts(dump, driver.view.dynamic_types, labels, owners)
        expressed = _record_delta(window.records)
        current = before.copy()
        current.update(expressed)
        if any(n < 0 for n in current.values()):
            raise ValueError('retained army gestures do not belong to the current board')
        extra = _missing_records(state, driver.human_side, current, physical, labels, owners, before)
    except (KeyError, ValueError) as error:
        return Recovery('hold', str(error))
    if not extra:
        return Recovery('hold', 'the stable board supplies no missing combat gesture')
    window.records.extend(extra)
    with _preserve_randomness(driver.rctl):
        result = gm.match_window(state, pending['side'], window,
                                 menu=pending['actions'], pin=driver._attempt_pin(),
                                 rng=driver.rctl, advance_tolerance=True,
                                 fs_prefilter=True)
    if result.status != 'match':
        chain = getattr(result, 'proven_starve_chain', ())
        stopped = getattr(result, 'proven_starve_state', None)
        if (result.status == 'hold' and result.starved and chain
                and getattr(result, 'proven_starve_call', None) is not None
                and not getattr(result, 'proven_starve_conflict', False)
                and all(_physical_action(state, action) for action in chain)
                and stopped is not None
                and _permanent_losses_match(stopped, pending['side'], dump, driver.view.dynamic_types)):
            return Recovery('pending', 'recovered physical payment; its recorded continuation still needs client randomness',
                            window=window, match=result, added=len(extra))
        return Recovery('hold', result.note or 'board does not complete a unique legal combat choice',
                        match=result)
    if any(not _physical_action(state, action) for action in result.chain):
        return Recovery('hold', 'recovery would require an unexpressed combat choice', match=result)
    if not _permanent_losses_match(result.post, pending['side'], dump, driver.view.dynamic_types):
        return Recovery('hold', 'physical FP casualty and reserve boxes disagree with the legal payment', match=result)
    return Recovery('match', 'recovered %d missing physical combat gesture(s)' % len(extra),
                    window=window, match=result, added=len(extra))
