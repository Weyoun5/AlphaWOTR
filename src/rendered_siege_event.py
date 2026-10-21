"""Exact co-located target evidence for Grond / Fighting Uruk-hai events."""
def match(options, batch):
    import rendered_extract as rx
    import rendered_replay as rr
    candidates=[]
    for index,action in options:
        choice=action.params.get('choices') or {}
        source=rr.sanitize(str(choice.get('attack_from','')))
        target=rr.sanitize(str(choice.get('attack_target','')))
        if not source or source!=target:
            return None,'cardint_siege_not_colocated'
        candidates.append((index,target))
    if len(candidates)==1:
        return candidates[0][0],None  # Forced target, no learned row emitted.
    first=min((int(chunk['line']) for chunk in batch.get('rng',())
               if chunk.get('kind')=='roll_num'),default=None)
    if first is None:
        return None,'cardint_siege_no_recorded_roll'
    witnessed=set()
    for record in batch.get('records', ()):
        if int(record.get('line', 0)) <= first:
            continue
        origin=rr.zone_kind(record.get('from',record.get('zone','')))
        dest=rr.zone_kind(record.get('to',''))
        if origin[0] not in ('region','stronghold'):
            continue
        if record.get('op')=='kill':
            tokens=record.get('pieces') or ()
        elif record.get('op')=='move' and dest[0] in ('cas','reinf'):
            tokens=rx._group_tokens(record.get('obj',''))
        else:
            continue
        # A simultaneous mixed casualty group proves location without
        # claiming an ordering or hit cost for its individual members.
        if any(piece and piece[0]=='unit' for piece in (rr.parse_piece(t) for t in tokens)):
            witnessed.add(rr.sanitize(origin[1]))
    matched=[index for index,target in candidates if target in witnessed]
    return (matched[0],None) if len(matched)==1 else (None,'cardint_siege_target_unproven')
