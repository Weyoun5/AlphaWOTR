"""Reconstruct a public Hunt bag from literal tile gestures after Mordor entry.

No replay RNG, future draws, hidden deck order, or final snapshots supply counts.
Entry housekeeping must actually return all eligible used tiles before a row is
admitted. Named special tiles require their public introduction gesture.
"""
from collections import Counter
from copy import copy
import re
import unicodedata


def _norm(value):
    return re.sub(r'[^a-z0-9]', '', unicodedata.normalize('NFKD', str(value)).encode('ascii', 'ignore').decode().lower())


def tile(value):
    import game_state as gs
    text = str(value).strip()
    if '/' in text:  # an Event/Combat card is not its similarly named tile
        return None
    name = _norm(text)
    if name == 'eye':
        return gs.HuntTile(is_eye=True)
    if re.fullmatch(r'[0-3]r?', name):
        return gs.HuntTile(damage=int(name[0]), reveal=name.endswith('r'))
    for label, template in gs.SPECIAL_HUNT_TILES.items():
        if name == _norm(label):
            return copy(template)
    return None


def key(value):
    return ('special', _norm(value.special_name)) if value.is_special else ('standard', value.damage, value.reveal, value.is_eye)


def _pieces(obj):
    import rendered_replay as rr
    text = str(obj).strip()
    # This tile alone has an internal comma; the generic piece tokenizer
    # otherwise treats every comma as a group separator.
    text = re.sub(r'\bon,\s*on\b', 'On On', text, flags=re.I)
    parts = rr.split_group(text[text.find('(') + 1:text.rfind(')')]) if text.lower().startswith('group (') else [text]
    result = []
    for part in parts:
        match = re.match(r'^\[(\d+)\](.*)$', part.strip())
        count, caption = (int(match[1]), match[2]) if match else (1, part)
        parsed = tile(caption)
        if parsed is None:
            return None
        result.extend(copy(parsed) for _ in range(count))
    return result


def entry_line(batches, before_line):
    import rendered_replay as rr
    def contains_fellowship(obj):
        text = str(obj).strip()
        parts = (rr.split_group(text[text.find('(') + 1:text.rfind(')')])
                 if text.lower().startswith('group (') else [text])
        # The client can drag the Fellowship together with Army figures.
        # Those accompanying figures do not hide its literal entry, but an
        # unrelated group or duplicated Fellowship token cannot prove one.
        return sum(part.strip().lower() == 'the fellowship' for part in parts) == 1
    entries = [int(r['line']) for b in batches for r in b.get('records', ())
               if int(r.get('line') or 0) < before_line and r.get('op') == 'move'
               and contains_fellowship(r.get('obj', ''))
               and re.match(r'^mordor\s+0\b', str(r.get('to', '')).strip(), re.I)
               and not re.match(r'^mordor\s+', str(r.get('from', '')).strip(), re.I)]
    return min(entries) if entries else None


def public_pool(batches, before_line, *, require_entry=True):
    """Return (pool, used, refusal) at a strict physical-line prefix."""
    import game_state as gs
    import rendered_extract as rx
    import random
    from rendered_hunt_observation import modifier_refusal
    modifier_details = {}
    refusal = modifier_refusal(batches, before_line, modifier_details)
    if refusal:
        return None, None, refusal
    entry = entry_line(batches, before_line)
    if entry is None and require_entry:
        return None, None, 'mordor_entry_unproven'
    rng = random.getstate()
    try:
        pool = gs._initial_hunt_pool()
    finally:
        random.setstate(rng)
    used = []
    introduced = set()
    staged = {}
    held = {}
    pending_entry = Counter()
    events = {}

    def add(line, kind, payload):
        event_key = (int(line), kind)
        if event_key in events and events[event_key] != payload:
            raise ValueError('pool_event_conflict')
        events[event_key] = payload

    try:
        if entry is not None:
            add(entry, 'entry', None)
        for proof in modifier_details.get('mithril', ()):
            add(proof['use_line'], 'hold', proof['tile_key'])
            if proof['return_line'] is not None:
                add(proof['return_line'], 'before_move_release', proof['tile_key'])
        for batch in batches:
            for item in rx._hunt_tile_items(batch):
                line = int(item.get('line') or 0)
                if line < before_line:
                    value = tile(item.get('value', ''))
                    if value is None:
                        raise ValueError('pool_tile_unrecognized')
                    add(line, 'draw', key(value))
            for record in batch.get('records', ()):
                line = int(record.get('line') or 0)
                if line >= before_line:
                    continue
                if record.get('op') == 'draw':
                    import rendered_replay as rr
                    prefix = [r for r in batch.get('records', ())
                              if int(r.get('line') or 0) < line]
                    value = tile(record.get('what', ''))
                    if value is not None and rr._draw_kind(record.get('what', ''), prefix)[0] == 'tile':
                        add(line, 'draw', key(value))
                if record.get('op') != 'move':
                    continue
                if record.get('_rx_pool_duplicate_of') is not None:
                    # Parser verified the immediately repeated physical IDs,
                    # both zones and full tile multiset against paired raw
                    # transport. No second object moved or entered the bag.
                    continue
                source, target = (_norm(record.get(k, '')) for k in ('from', 'to'))
                bag_zones = ('huntpool', 'thehuntpool')
                relevant = (source in (*bag_zones, 'removedhunttiles')
                            or target in (*bag_zones, 'usedhunttiles', 'removedhunttiles'))
                values = _pieces(record.get('obj', ''))
                if not relevant and not (values and all(t.is_special for t in values)):
                    continue
                if values is None:
                    # Moving a non-tile into/out of a tile zone makes it unsafe
                    # to assume a standard initial bag plus known transitions.
                    raise ValueError('pool_tile_movement_unrecognized')
                add(line, 'move', (source, target, tuple(key(t) for t in values)))

        templates = {key(t): t for t in pool + list(gs.SPECIAL_HUNT_TILES.values())}

        def remove(items, wanted):
            found = next((i for i, value in enumerate(items) if key(value) == wanted), None)
            if found is None:
                raise ValueError('pool_tile_unavailable')
            return items.pop(found)

        for (line, kind), payload in sorted(events.items()):
            if kind == 'entry':
                pending_entry = Counter(key(t) for t in used if t.is_eye or t.is_special)
            elif kind == 'hold':
                if payload in held:
                    raise ValueError('mithril_held_tile_conflict')
                held[payload] = remove(used, payload)
            elif kind == 'before_move_release':
                if payload not in held:
                    raise ValueError('mithril_held_tile_unavailable')
                used.append(held.pop(payload))
            elif kind == 'draw':
                if entry is not None and line > entry and pending_entry:
                    raise ValueError('mordor_draw_before_pool_restored')
                if not pool:
                    pool = [t for t in used if not t.is_special]
                    used = [t for t in used if t.is_special]
                used.append(remove(pool, payload))
            else:
                source, target, values = payload
                if source in ('huntpool', 'thehuntpool') and target in ('huntpool', 'thehuntpool'):
                    if Counter(values) - Counter(key(t) for t in pool):
                        raise ValueError('pool_same_zone_tile_unavailable')
                    continue
                for wanted in values:
                    template = templates.get(wanted)
                    if template is None:
                        raise ValueError('pool_tile_unknown')
                    if target in ('huntpool', 'thehuntpool'):
                        if source in ('huntpool', 'thehuntpool'):
                            # A drag within one zone does not add or remove
                            # tiles. Presence is still checked below before
                            # accepting its entire multiset as a no-op.
                            continue
                        if source == staged.get(wanted):
                            if wanted in introduced:
                                raise ValueError('pool_special_introduction_invalid')
                            del staged[wanted]
                            introduced.add(wanted)
                            pool.append(copy(template))
                        elif source == 'removedhunttiles' and template.is_special:
                            if wanted in introduced:
                                raise ValueError('pool_special_introduction_invalid')
                            introduced.add(wanted)
                            pool.append(copy(template))
                        elif (source == 'usedhunttiles' and template.is_special
                              and wanted not in introduced):
                            # Some clients/logs omit the earlier staging move.
                            # This literal move still proves exactly which
                            # unique special tile enters the physical bag now.
                            # It grants no knowledge to any earlier prefix.
                            introduced.add(wanted)
                            pool.append(copy(template))
                        else:
                            pool.append(remove(used, wanted))
                            if pending_entry[wanted]:
                                pending_entry[wanted] -= 1
                                if not pending_entry[wanted]:
                                    del pending_entry[wanted]
                    elif source in ('huntpool', 'thehuntpool'):
                        value = remove(pool, wanted)
                        if target != 'removedhunttiles':
                            used.append(value)
                    elif target == 'removedhunttiles':
                        remove(used, wanted)
                        if pending_entry[wanted]:
                            pending_entry[wanted] -= 1
                            if not pending_entry[wanted]:
                                del pending_entry[wanted]
                    elif source == 'removedhunttiles' and target != 'usedhunttiles':
                        if not template.is_special or wanted in introduced or wanted in staged:
                            raise ValueError('pool_staged_tile_invalid')
                        staged[wanted] = target
                    elif wanted in staged and source == staged[wanted]:
                        staged[wanted] = target
                    elif target == 'usedhunttiles' and source == 'removedhunttiles':
                        if not template.is_special or wanted in introduced:
                            raise ValueError('pool_staged_tile_invalid')
                        introduced.add(wanted)
                        used.append(copy(template))
                        if entry is not None and line > entry:
                            pending_entry[wanted] += 1
                    # Draws already enter the used multiset. Physical Current
                    # Card -> Used housekeeping must not count them twice.
        if pending_entry:
            raise ValueError('mordor_pool_restoration_incomplete')
        return pool, used, None
    except ValueError as error:
        return None, None, str(error)
