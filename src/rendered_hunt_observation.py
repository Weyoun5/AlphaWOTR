"""Public Hunt-bag observations, separate from replay's forced RNG pools.

Only an explicit snapshot or consumed-tile boundary is usable. Unknown histories
remain refusals; callers must exclude those rows from training admission.
"""
from contextlib import contextmanager
from copy import copy

PROVENANCE_VERSION = 'public_hunt_prefix_v2'


def mark(state, before_line, kind='snapshot'):
    state._rx_hunt_observation = (int(before_line), str(kind))


class PublicHuntHistory:
    def __init__(self, batches, query_pool):
        self.batches = batches
        self.query_pool = query_pool
        self.cache = {}
        self.current_batch = None

    def lookup(self, boundary):
        boundary = int(boundary)
        if boundary not in self.cache:
            crossed = any(int(b.get('start_line') or 0) < boundary
                          and (b.get('truth_before') or {}).get('fel', {}).get('mordor')
                          for b in self.batches)
            from rendered_mordor_pool import entry_line, public_pool
            pool, discarded, reason = (public_pool(self.batches, boundary)
                                      if crossed or entry_line(self.batches, boundary) is not None
                                      else self.query_pool(self.batches, boundary))
            self.cache[boundary] = (tuple(pool or ()), tuple(discarded or ()), reason)
        pool, discarded, reason = self.cache[boundary]
        return [copy(tile) for tile in pool], [copy(tile) for tile in discarded], reason

    def observation(self, state, phase, action):
        marker = getattr(state, '_rx_hunt_observation', None)
        if marker is not None and (not isinstance(marker, tuple) or len(marker) != 2
                or not isinstance(marker[0], int) or marker[0] < 1
                or marker[1] not in ('snapshot', 'stable_batch', 'before_draw',
                                     'after_draw', 'verified_reveal_prefix')):
            evidence = {'version': PROVENANCE_VERSION, 'boundary': None, 'timing': None}
            return None, None, evidence, 'boundary_invalid'
        batch = self.current_batch
        events = []
        if batch is not None:
            events = [int(r.get('line') or 0) for r in batch.get('rng', ())
                      if r.get('kind') == 'draw_tile']
            events.extend(int(r.get('line') or 0) for r in batch.get('records', ())
                          if r.get('op') == 'move' and str(r.get('to', '')).lower()
                          in ('hunt pool', 'the hunt pool'))
        # Census combat states bypass RX reconstruction. Their pool is stable
        # throughout a draw-free batch, so the shared boundary is sufficient.
        if marker is None and batch is not None:
            if not events:
                marker = (int(batch['start_line']), 'stable_batch')
        evidence = {'version': PROVENANCE_VERSION,
                    'boundary': marker[0] if marker else None,
                    'timing': marker[1] if marker else None}
        if marker is None:
            return None, None, evidence, 'boundary_unproven'
        boundary, timing = marker
        if getattr(state.fellowship, 'mordor_entered', False):
            from rendered_mordor_pool import entry_line
            if entry_line(self.batches, boundary) is None:
                evidence['missing'] = 'literal_mordor_entry'
                return None, None, evidence, 'mordor_history_unproven'
        # A reconstructed batch start is not a post-draw decision. Only the
        # emitter that actually consumed the tile may provide that boundary.
        if timing in ('snapshot', 'stable_batch'):
            if getattr(state, 'pending_hunt', None) or getattr(state, '_pending_hunt_damage', 0):
                return None, None, evidence, 'postdraw_boundary_unproven'
            if batch is not None:
                future_draws = [line for line in events if line >= boundary]
                pending = (getattr(state, 'pending_card_option', None)
                           or getattr(state, 'pending_combat', None))
                if future_draws and pending:
                    return None, None, evidence, 'interior_draw_boundary_unproven'
        pool, discarded, reason = self.lookup(boundary)
        if reason:
            return None, None, evidence, reason
        evidence.update(pool_size=len(pool), discard_size=len(discarded))
        return pool, discarded, evidence, None


@contextmanager
def encoding_pool(state, pool, discarded):
    """Never alter the replay RNG pool, even if encoding raises."""
    if pool is None:
        yield
        return
    original_pool, original_discard = state.hunt_pool, state.hunt_discard
    state.hunt_pool, state.hunt_discard = list(pool), list(discarded)
    try:
        yield
    finally:
        state.hunt_pool, state.hunt_discard = original_pool, original_discard


def modifier_refusal(batches, before_line, details=None):
    """Enforce observed Event obligations without confusing their combat faces.

    The tile ledger proves composition, but cannot infer an unrecorded required
    draw/return merely because a modifier card has left play.
    """
    from collections import Counter
    import re
    import rendered_extract as rx
    from rendered_mordor_pool import tile, key, _pieces, _norm
    for batch in batches:
        records = [r for r in batch.get('records', ())
                   if int(r.get('line') or 0) < before_line]
        relevant = [r for r in records if any(name in
                    (str(r.get('obj', '')) + ' ' + ' '.join(r.get('pieces', ()))).lower()
                    for name in ('challenge of the king', 'mithril coat'))]
        if not relevant:
            continue
        committed = {rx._title_to_cid(str(r.get('obj', ''))) for r in records
                     if r.get('op') == 'move' and '(chosen)' in str(r.get('from', '')).lower()}
        if batch.get('kind') in ('combat', 'combat?'):
            windows = rx._combat_round_windows(batch)
            if windows:
                rounds, _, _ = rx._round_cards(batch, windows)
                committed.update(row.get(side) for row in rounds for side in ('fp', 'sh'))
        challenge_line = None
        mithril_use = False
        for record in relevant:
            text = (str(record.get('obj', '')) + ' ' + ' '.join(record.get('pieces', ()))).lower()
            for name in ('challenge of the king', 'mithril coat'):
                if name not in text:
                    continue
                title = ('Challenge of the King/Sudden Strike' if name.startswith('challenge')
                         else 'Mithril Coat and Sting/Blade of Westernesse')
                cid = rx._title_to_cid(title)
                if cid is not None and cid in committed:
                    continue
                origin = rx._zone_norm(record.get('from', ''))
                target = rx._zone_norm(record.get('to', ''))
                if name.startswith('challenge'):
                    # Combat commitments explicitly say (chosen), even where a
                    # damaged battle window cannot be reconstructed.
                    if '(chosen)' in origin:
                        continue
                    if (record.get('op') == 'move' and origin == 'free peoples hand'
                            and target in ('current fp card', 'current sa card')):
                        challenge_line = int(record['line'])
                    continue
                current = ('current fp card', 'current sa card')
                inert = (record.get('op') == 'move'
                         and str(record.get('obj', '')).strip().lower() in
                         ('mithril coat and sting', 'mithril coat and sting/blade of westernesse')
                         and origin in ('free peoples hand', *current)
                         and (target in current or re.fullmatch(r'board[0-9]+', target)))
                if not inert:
                    mithril_use = True
        if mithril_use:
            proof = _mithril_redraw_proof(
                records, allow_held=int(batch.get('end_line') or 0) > before_line)
            if proof is None:
                return 'pool_modifier_history'
            if details is not None:
                details.setdefault('mithril', []).append(proof)
        if challenge_line is None:
            continue
        draws = {int(r['line']): tile(r.get('what', '')) for r in records
                 if r.get('op') == 'draw' and int(r['line']) > challenge_line
                 and tile(r.get('what', '')) is not None}
        for item in rx._hunt_tile_items(batch):
            line = int(item.get('line') or 0)
            if challenge_line < line < before_line:
                draws[line] = tile(item.get('value', ''))
        if len(draws) != 3 or any(t is None or t.is_special for t in draws.values()):
            return 'pool_modifier_history'
        returned, removed = Counter(), Counter()
        for record in records:
            if record.get('op') != 'move' or int(record.get('line') or 0) <= max(draws):
                continue
            target = _norm(record.get('to', ''))
            if target not in ('huntpool', 'thehuntpool', 'removedhunttiles'):
                continue
            values = _pieces(record.get('obj', ''))
            if values is None:
                return 'pool_modifier_history'
            (removed if target == 'removedhunttiles' else returned).update(key(t) for t in values)
        all_eyes = all(t.is_eye for t in draws.values())
        expected_returns = Counter(key(t) for t in draws.values() if all_eyes or not t.is_eye)
        expected_removed = Counter(key(t) for t in draws.values() if t.is_eye and not all_eyes)
        if returned != expected_returns or removed != expected_removed:
            return 'pool_modifier_history'
    return None


def _mithril_redraw_proof(records, *, allow_held=False):
    """Prove a completed return, or a physically held original in this action.

    A missing return cannot carry past the enclosing action boundary. Held
    originals stay outside both pool and discard, including empty-bag refill.
    """
    import rendered_replay as rr
    from rendered_mordor_pool import tile, key, _pieces, _norm
    draws = []
    for record in records:
        if record.get('op') != 'draw':
            continue
        line = int(record.get('line') or 0)
        prefix = [r for r in records if int(r.get('line') or 0) < line]
        value = tile(record.get('what', ''))
        if value is not None and rr._draw_kind(record.get('what', ''), prefix)[0] == 'tile':
            draws.append((line, value))
    if len(draws) != 2:
        return None
    (first_line, first), (second_line, _) = sorted(draws)
    activations = [r for r in records if r.get('op') == 'move'
                   and first_line < int(r.get('line') or 0) < second_line
                   and str(r.get('obj', '')).strip().lower() in
                       ('mithril coat and sting', 'mithril coat and sting/blade of westernesse')
                   and _norm(r.get('from', '')).startswith('board')
                   and _norm(r.get('to', '')) in ('currentfpcard', 'currentsacard')]
    # Some clients stage the used card next to the held original on a Board
    # slot, rather than in a Current Card slot. Require the same two named
    # draw owners, an explicit held original in that slot, and no intervening
    # different card/action. A bare table rearrangement proves no use.
    first_record = next(r for r in records if int(r.get('line') or 0) == first_line)
    second_record = next(r for r in records if int(r.get('line') or 0) == second_line)
    same_draw_owner = (first_record.get('player') is not None
                       and first_record.get('player') == second_record.get('player'))
    stages = [r for r in records if r.get('die_stage') and int(r.get('line') or 0) < first_line]
    stage = max(stages, key=lambda r: int(r['line'])) if stages else None
    stage_side = str((stage.get('die_stage') or ('',))[0]) if stage else ''
    draw_player = first_record.get('player')
    shadow_draw = bool(stage and stage.get('player') not in (None, 'auto') and draw_player
                       and ((stage_side == 'SHADOW' and stage['player'] == draw_player)
                            or (stage_side == 'FREE_PEOPLES' and stage['player'] != draw_player)))
    mithril_names = ('mithril coat and sting', 'mithril coat and sting/blade of westernesse')
    between = [r for r in records if first_line < int(r.get('line') or 0) < second_line]
    other_action = any(r.get('die_stage') or (r.get('op') == 'move' and
                       ('action die' in str(r.get('obj', '')).lower()
                        or ('/' in str(r.get('obj', ''))
                            and str(r.get('obj', '')).strip().lower() not in mithril_names)))
                       for r in between)
    if same_draw_owner and shadow_draw and not other_action:
        for record in between:
            origin, target = (_norm(record.get(k, '')) for k in ('from', 'to'))
            if (record.get('op') != 'move' or str(record.get('obj', '')).strip().lower() not in mithril_names
                    or not origin.startswith('board') or not target.startswith('board') or origin == target):
                continue
            if any(r.get('op') == 'move' and int(r.get('line') or 0) < int(record['line'])
                   and _norm(r.get('to', '')) == target
                   and (values := _pieces(r.get('obj', ''))) is not None
                   and len(values) == 1 and key(values[0]) == key(first) for r in between):
                activations.append(record)
    if not activations:
        return None
    # A temporarily picked-up card may first be returned to its table slot.
    # The last observed activation before the replacement is the actual use.
    use_line = max(int(r['line']) for r in activations)
    returns = []
    for record in records:
        if record.get('op') != 'move' or int(record.get('line') or 0) <= second_line:
            continue
        if _norm(record.get('to', '')) not in ('huntpool', 'thehuntpool'):
            continue
        values = _pieces(record.get('obj', ''))
        if values is not None and len(values) == 1 and key(values[0]) == key(first):
            returns.append(record)
    if len(returns) > 1 or (not returns and not allow_held):
        return None
    returned = returns[0] if returns else None
    if returned is not None and _norm(returned.get('from', '')) in ('huntpool', 'thehuntpool', 'removedhunttiles'):
        return None
    # The recorded activation followed by the actual replacement draw proves
    # the first tile is held by the card rule. Its cosmetic staging zone is
    # irrelevant. A closed action still requires the physical return above.
    return dict(use_line=use_line, return_line=int(returned['line']) if returned is not None else None, tile_key=key(first))
