"""Derived event evidence; original native records remain the undo authority."""


def chosen_card_pids(record, side):
    """Decode only the native chosen-card reveal protocol's physical PID.

    Game.a(card) emits ``reveal PID commitment face_index``. The generic
    metadata number collector includes that final face index in pids/segs;
    it is not a second piece. All other movement keeps ordinary PID parsing.
    """
    from .sidecar import meta_pids
    meta = record.get('meta') or {}
    tags = {'FP': {'Controls.861', 'Controls.862', 'Controls.863'},
            'SH': {'Controls.866', 'Controls.867', 'Controls.868'}}[side]
    segments = meta.get('segs')
    if tags <= set(meta.get('tags') or ()):
        if (isinstance(segments, list) and len(segments) > 1
                and isinstance(segments[1], list) and segments[1]
                and str(segments[1][0]).isdigit()):
            return [str(segments[1][0])]
        # Recognized native protocol with missing identity cannot fall back
        # to a face index collected elsewhere in the record.
        return None
    return meta_pids(meta)


def normalized_records(window):
    from . import gesture_match as gm
    from .sidecar import meta_pids
    records = list(window.body_records())
    side = {'FREE_PEOPLES': 'FP', 'SHADOW': 'SH'}.get(getattr(window, 'die_side', None))
    if not side or getattr(window, 'combat_cards_open', None) is not False:
        return records
    hand = 'Free Peoples Hand' if side == 'FP' else 'Shadow Hand'
    prefix = 'free' if side == 'FP' else 'shadow'
    trajectories, consumed = [], set()

    def stage(location):
        return location == ('play', side) or (location[0] == 'other' and bool(location[1]))

    def compatible(path, location, deck, pids):
        return (path['location'] == location and path['deck'] == deck
                and (not pids or not path['pids'] or pids == path['pids']))

    for index, record in enumerate(records):
        op = record.get('op')
        pids = meta_pids(record.get('meta'))
        if op == 'kill':
            location = gm.zone_key(record.get('zone'))
            for path in trajectories[:]:
                same_piece = pids and path['pids'] and set(pids) & set(path['pids'])
                # Native kill rendering names Current Action even for table
                # cards; without identity, conservatively cancel that proof.
                anonymous_card = ((location == path['location'] or location == ('other', 'current action'))
                                  and not (pids and path['pids'])
                                  and 'card' in str(record.get('pieces', '')).lower())
                if same_piece or anonymous_card:
                    if path.get('cid'):
                        path['closed'] = True
                    else:
                        trajectories.remove(path)
            continue
        if op != 'move':
            continue
        source, destination = gm.zone_key(record.get('from')), gm.zone_key(record.get('to'))
        obj = str(record.get('obj', ''))
        generic = obj.lower() in (prefix + ' strategy card', prefix + ' character card')
        # Once public, native card movement can use its title rather than
        # the generic deck name. Its known PID still owns the trajectory.
        if not generic and source[0] != 'chosen_other':
            paths = [p for p in trajectories if not p.get('closed') and pids and p['pids'] and set(pids) & set(p['pids'])]
            cid = gm.title_to_cid(obj)
            if not paths and cid is not None:
                deck = 'strategy' if '_Strategy_' in cid else 'character'
                anonymous = [p for p in trajectories if not p.get('closed') and (not p['pids'] or not pids)
                             and p['location'] == source and p['deck'] == deck]
                if len(anonymous) == 1:
                    paths = anonymous
            for path in paths:
                matching_title = (cid is not None and cid.startswith(
                    'FREE_PEOPLES_' if side == 'FP' else 'SHADOW_')
                    and ('strategy' if '_Strategy_' in cid else 'character') == path['deck']
                    and path.get('cid', cid) == cid)
                if ((path['pids'] and pids and pids != path['pids']) or source != path['location']
                        or not matching_title or (not stage(destination)
                            and not (destination[0] == 'dead' and path.get('cid')))):
                    trajectories.remove(path)
                else:
                    path['location'] = destination
                    path['pids'] = path['pids'] or pids
                    path['cid'] = cid
                    path['public_index'] = index
                    path['legs'].append(index)
                    path['closed'] = destination[0] == 'dead'
            if paths:
                continue
        if generic:
            deck = 'strategy' if 'strategy' in obj.lower() else 'character'
            if source == ('hand', side) and stage(destination):
                trajectories.append(dict(location=destination, deck=deck, pids=pids, legs=[index]))
                continue
            candidates = [p for p in trajectories if not p.get('closed') and compatible(p, source, deck, pids)]
            if len(candidates) != 1:
                # Anonymous branching cannot prove which card moved.
                for path in candidates:
                    trajectories.remove(path)
                continue
            path = candidates[0]
            if not stage(destination):
                if destination[0] == 'dead' and path.get('cid'):
                    path['closed'] = True
                    path['legs'].append(index)
                else:
                    trajectories.remove(path)
                continue
            path['location'] = destination
            path['pids'] = path['pids'] or pids
            path['legs'].append(index)
            continue
        if source[0] != 'chosen_other' or destination != ('play', side):
            continue
        pids = chosen_card_pids(record, side)
        cid = gm.title_to_cid(obj)
        if cid is None or not cid.startswith('FREE_PEOPLES_' if side == 'FP' else 'SHADOW_'):
            continue
        location = gm.zone_key(str(record.get('from', '')).rsplit(' (chosen)', 1)[0])
        deck = 'strategy' if '_Strategy_' in cid else 'character'
        for path in trajectories[:]:
            if (pids and path['pids'] == pids and path.get('cid')
                    and path['cid'] != cid):
                trajectories.remove(path)
        candidates = [p for p in trajectories if not p.get('closed') and compatible(p, location, deck, pids)
                      and p.get('cid', cid) == cid]
        if len(candidates) != 1:
            continue
        path = candidates[0]
        path.update(cid=cid, public_index=index, location=destination)
        path['legs'].append(index)
    for path in trajectories:
        if not path.get('cid'):
            continue
        index = path['public_index']
        records[index] = dict(records[index], **{'from': hand + ' (chosen)',
                              'to': 'Current FP Card' if side == 'FP' else 'Current SA Card',
                              '_staged_origin': path['legs'][0]})
        consumed.update(path['legs'])
        consumed.discard(index)
    return [r for i, r in enumerate(records) if i not in consumed]
