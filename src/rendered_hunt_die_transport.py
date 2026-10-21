"""Physical-ID proof for repeated parking of a spent Fellowship die.

Repeated notifications and brief Hunt-to-Current-to-Hunt restaging do not spend
another die. Printed equal faces alone are never enough to identify a repeat.
"""
def _aligned_hunt_moves(batches, rendered_path, raw_path, side):
    import log_census as lc
    import rendered_extract as rx
    import rendered_replay as rr
    from raw_log_reader import read_raw_log_text
    from location_codes_authoritative import AUTH_LOCATION_CODES
    own_ids = (lc.BASE_FP_DIE_PIDS | lc.KNOWN_BONUS_FP if side == 'FREE_PEOPLES'
               else lc.BASE_SH_DIE_PIDS | lc.KNOWN_BONUS_SH)
    norm = lambda value: str(value).strip().lower()
    box = 'the hunt for the ring'
    rendered = []
    for batch in batches:
        for record in batch['records']:
            if record.get('op') != 'move':
                continue
            origin, target = norm(record.get('from')), norm(record.get('to'))
            if box not in (origin, target):
                continue
            faces = [face for owner, face in rx._die_tokens(record.get('obj', '')) if owner == side]
            if faces:
                rendered.append((record, batch, (origin, target, len(faces)), tuple(sorted(faces))))
    if not rendered:
        return []
    try:
        paired = rr._paired_raw_path(rendered_path, raw_path, lambda reason: None)
        raw = rx._paired_physical_raw_view(batches, read_raw_log_text(paired))
    except (OSError, ValueError, TypeError):
        return []
    physical = []
    for line, text in enumerate(raw, 1):
        match = lc.MOVE_RE.search(text)
        if not match:
            continue
        ids = match.group(1).split()
        own = [pid for pid in ids if pid in own_ids]
        if not own:
            continue
        origin, target = (norm(AUTH_LOCATION_CODES.get(match.group(k), '')) for k in (2, 3))
        if box not in (origin, target):
            continue
        if len(own) != len(set(own)):
            return []
        physical.append((line, tuple(sorted(own)), (origin, target, len(own))))
    if [entry[2] for entry in rendered] != [entry[2] for entry in physical]:
        return []
    return list(zip(rendered, physical))


def mark_repeated_deposits(batches, rendered_path, raw_path=None):
    box = 'the hunt for the ring'
    aligned = _aligned_hunt_moves(batches, rendered_path, raw_path, 'FREE_PEOPLES')
    count = 0
    previous = {}
    for (record, batch, signature, faces), (raw_line, ids, _) in aligned:
        if signature[1] != box:
            continue
        key = id(batch), ids
        prior = previous.get(key)
        if prior is not None:
            first, first_raw, first_faces = prior
            between = [r for r in batch['records'] if int(first['line']) < int(r['line']) < int(record['line'])]
            if faces == first_faces and not any(r.get('die_stage') or r.get('op') in ('roll_dice', 'turn') for r in between):
                record['_rx_hunt_repeat_deposit'] = dict(first_line=int(first['line']),
                    repeated_line=int(record['line']), physical_ids=list(ids), raw_lines=[first_raw, raw_line])
                count += 1
                continue
        previous[key] = record, raw_line, faces
    return count


def mark_shadow_roundtrips(batches, rendered_path, raw_path=None):
    """An authenticated die leaves and returns before a new action/chance."""
    import rendered_replay as rr
    aligned = _aligned_hunt_moves(batches, rendered_path, raw_path, 'SHADOW')
    records = sorted((r for b in batches for r in b['records']), key=lambda r: r['line'])
    count = 0
    for first, second in zip(aligned, aligned[1:]):
        (a, ba, sa, fa), (raw_a, ids_a, _) = first
        (b, bb, sb, fb), (raw_b, ids_b, _) = second
        if a.get('_rx_shadow_hunt_roundtrip') or b.get('_rx_shadow_hunt_roundtrip'):
            continue
        if ids_a != ids_b or fa != fb or sa != (sb[1], sb[0], sb[2]) or ba['turn'] != bb['turn']:
            continue
        between = [r for r in records if int(a['line']) < int(r['line']) < int(b['line'])]
        if any(r.get('die_stage') or r.get('op') in ('turn', 'roll_dice', 'alloc_eye', 'add_eye')
               or (r.get('op') == 'move' and str(r.get('obj', '')).lower() in ('fsp', 'the fellowship'))
               or (r.get('op') == 'draw' and (piece := rr.parse_piece(r.get('what', ''))) is not None
                   and piece[0] == 'tile') for r in between):
            continue
        proof = dict(physical_ids=list(ids_a), rendered_lines=[int(a['line']), int(b['line'])],
                     raw_lines=[raw_a, raw_b], original_zone=sa[0], temporary_zone=sa[1])
        a['_rx_shadow_hunt_roundtrip'] = b['_rx_shadow_hunt_roundtrip'] = proof
        count += 1
    return count


def mark_lidless_original_faces(batches, rendered_path, raw_path=None):
    """Recover original selected faces from unchanged rolled physical IDs."""
    import log_census as lc
    import rendered_extract as rx
    import rendered_replay as rr
    from raw_log_reader import read_raw_log_text
    import re
    aligned = _aligned_hunt_moves(batches, rendered_path, raw_path, 'SHADOW')
    wanted = {}
    for (record, batch, signature, printed), (line, ids, _) in aligned:
        if (signature[:2] == ('sa dice', 'the hunt for the ring') and printed and set(printed) == {'E'}
                and not record.get('_rx_shadow_hunt_roundtrip')
                and str(rx._card_play_title(batch) or '').split('/')[0].lower().strip() == 'the lidless eye'):
            wanted[line] = record, batch, ids
    if not wanted:
        return 0
    paired = rr._paired_raw_path(rendered_path, raw_path, lambda reason: None)
    raw = rx._paired_physical_raw_view(batches, read_raw_log_text(paired))
    sh_ids = lc.BASE_SH_DIE_PIDS | lc.KNOWN_BONUS_SH
    faces, zones, rolled_at, changed = {}, {}, {}, set()
    turn = 1
    count = 0
    for line, text in enumerate(raw, 1):
        marker = re.search(r'<~Game\.7~>(\d+)\b', text)
        if marker:
            turn = int(marker[1])
        rolled = lc.ROLL_RE.search(text)
        if rolled:
            ids, values = rolled[1].split(), rolled[2].split()
            if len(ids) == len(values) and ids and set(ids).issubset(sh_ids) and all(value.isdigit() and int(value) % 10 < 6 for value in values):
                for pid, value in zip(ids, values):
                    faces[pid] = 'MHACPE'[int(value) % 10]
                    zones[pid] = 'Game.2043'
                    rolled_at[pid] = (turn, line)
                    changed.discard(pid)
        reface = lc.CHANGE_RE.search(text)
        if reface and reface[1] in sh_ids:
            faces[reface[1]] = reface[2]
            changed.add(reface[1])
        allocation = lc.EYE_BTN_RE.search(text)
        if allocation:
            faces[allocation[1]], zones[allocation[1]] = 'E', 'Game.1997'
            changed.add(allocation[1])
        move = lc.MOVE_RE.search(text)
        if move:
            if line in wanted:
                record, batch, ids = wanted[line]
                if all(pid in faces and faces[pid] != 'E' and pid not in changed
                       and zones.get(pid) == 'Game.2043' and rolled_at.get(pid, (None,))[0] == batch['turn']
                       for pid in ids):
                    record['_rx_lidless_original_faces'] = [faces[pid] for pid in ids]
                    record['_rx_lidless_face_proof'] = dict(physical_ids=list(ids), deposit_raw_line=line,
                        roll_raw_lines=[rolled_at[pid][1] for pid in ids], original_faces=[faces[pid] for pid in ids])
                    count += 1
            for pid in move[1].split():
                if pid in sh_ids:
                    zones[pid] = move[3]
        killed = lc.KILL_RE.search(text)
        if killed:
            for pid in killed[1].split():
                if pid in sh_ids:
                    zones[pid] = 'used'
    return count


def mark_restored_shadow_allocation(batches, rendered_path, raw_path=None):
    """Prove a mistaken allocation restored before chance, including ID swaps.

    Physical dice are interchangeable only after their complete face/zone
    multiset is restored. This is separate from exact-ID transport roundtrips.
    """
    from collections import Counter
    import re
    import log_census as lc
    import rendered_extract as rx
    import rendered_replay as rr
    from raw_log_reader import read_raw_log_text
    records = sorted((r for b in batches for r in b['records']), key=lambda r:r['line'])
    allocations = [r for r in records if r.get('op') == 'alloc_eye']
    changes = []
    for r in records:
        match = rx._DIE_CHANGE_RE.match(str(r.get('raw','')))
        if match and match[1].lower() == 'shadow':
            changes.append((r,match[3]))
    aligned = _aligned_hunt_moves(batches,rendered_path,raw_path,'SHADOW')
    moves = {line:record for (record,_,_,_),(line,_,_) in aligned}
    paired = rr._paired_raw_path(rendered_path,raw_path,lambda reason:None)
    raw = rx._paired_physical_raw_view(batches,read_raw_log_text(paired))
    raw_alloc = [(n,t,lc.EYE_BTN_RE.search(t)) for n,t in enumerate(raw,1) if lc.EYE_BTN_RE.search(t)]
    raw_changes = [(n,t,lc.CHANGE_RE.search(t)) for n,t in enumerate(raw,1)
                   if lc.CHANGE_RE.search(t) and lc.CHANGE_RE.search(t)[1] in lc.BASE_SH_DIE_PIDS|lc.KNOWN_BONUS_SH]
    actor = lambda t: re.match(r'^\$([^<]+?)(?=<~| changes )',t).group(1).strip() if re.match(r'^\$([^<]+?)(?=<~| changes )',t) else None
    if ([actor(t) for _,t,_ in raw_alloc] != [r.get('player') for r in allocations]
        or [(actor(t),m[2]) for _,t,m in raw_changes] != [(str(r.get('raw','')).split(' changes ',1)[0].lstrip('$%'),f) for r,f in changes]):
        return 0
    alloc_map={n:r for (n,_,_),r in zip(raw_alloc,allocations)}
    change_map={n:r for (n,_,_),(r,_) in zip(raw_changes,changes)}
    ids=lc.BASE_SH_DIE_PIDS|lc.KNOWN_BONUS_SH
    faces={};zones={};rolled_at={};turn=1;pending=None;count=0
    def signature():return Counter((zones.get(pid),face) for pid,face in faces.items())
    for n,t in enumerate(raw,1):
        marker=re.search(r'<~Game\.7~>(\d+)\b',t)
        if marker:turn=int(marker[1]);pending=None
        allocation=lc.EYE_BTN_RE.search(t)
        move=lc.MOVE_RE.search(t)
        change=lc.CHANGE_RE.search(t)
        comment=not t.strip() or ('<~' not in t and re.match(r'^<[^<>]+>',t))
        if pending and not comment:
            allowed_return=(move and n in moves and (move[2],move[3])==('Game.1997','Game.2043')
                and len(move[1].split())==1 and move[1].strip() in pending['hunt_ids']
                and pending['returned'] is None and actor(t)==pending['actor'])
            allowed_change=(change and n in change_map and change[1]==pending['returned']
                and actor(t)==pending['actor'])
            if not (allowed_return or allowed_change):pending=None
            elif allowed_return:
                pending['returned']=move[1].strip();pending['records'].append(moves[n])
            else:pending['records'].append(change_map[n])
        rolled=lc.ROLL_RE.search(t)
        if rolled:
            ps,vs=rolled[1].split(),rolled[2].split()
            if len(ps)==len(vs) and ps and set(ps)<=ids:
                for pid,value in zip(ps,vs):
                    if value.isdigit() and int(value)%10<6:
                        faces[pid]='MHACPE'[int(value)%10];zones[pid]='Game.2043';rolled_at[pid]=turn
        if allocation:
            pid=allocation[1]
            hunt_ids={p for p in faces if zones.get(p)=='Game.1997' and faces[p]=='E'}
            if (n in alloc_map and faces.get(pid) in set('MHACP') and zones.get(pid)=='Game.2043'
                    and rolled_at.get(pid)==turn and hunt_ids):
                pending=dict(actor=actor(t),before=signature(),hunt_ids=hunt_ids,
                    original_face=faces[pid],allocated=pid,returned=None,raw_start=n,
                    records=[alloc_map[n]])
            faces[pid]='E';zones[pid]='Game.1997'
        if move:
            for pid in move[1].split():
                if pid in ids:
                    if (n in moves and turn not in rolled_at.values()
                            and (move[2],move[3])==('Game.2043','Game.1997')
                            and rx._die_tokens(moves[n].get('obj',''))==[('SHADOW','E')]):
                        faces[pid]='E'
                    zones[pid]=move[3]
        if change and change[1] in ids:faces[change[1]]=change[2]
        killed=lc.KILL_RE.search(t)
        if killed:
            for pid in killed[1].split():
                if pid in ids:zones[pid]='used'
        if pending and pending['returned'] and signature()==pending['before']:
            proof=dict(kind='restored_allocation_permutation',physical_ids=[pending['allocated'],pending['returned']],
                raw_lines=[pending['raw_start'],n],original_face=pending['original_face'],
                hunt_ids_before=sorted(pending['hunt_ids']),
                hunt_ids_after=sorted(p for p in faces if zones.get(p)=='Game.1997' and faces[p]=='E'),
                original_multiset=[dict(zone=z,face=f,count=c) for (z,f),c in sorted(pending['before'].items())])
            for r in pending['records']:r['_rx_shadow_hunt_roundtrip']=proof
            pending=None;count+=1
    return count


def annotate_direct_fellowship_ring(batches, rendered_path, raw_path=None):
    """A die parked in Hunt may fund a Ring before the Fellowship moves."""
    import copy,re
    import log_census as lc
    import rendered_extract as rx
    import rendered_replay as rr
    from raw_log_reader import read_raw_log_text
    from location_codes_authoritative import AUTH_LOCATION_CODES
    aligned=_aligned_hunt_moves(batches,rendered_path,raw_path,'FREE_PEOPLES')
    candidates=[]
    for (deposit,batch,signature,faces),(raw_line,ids,_) in aligned:
        tail=[r for r in batch['records'] if r['line']>=batch['start_line']]
        if (len(tail)<3 or tail[0] is not deposit or len(ids)!=1
                or signature[:2]!=('fp dice','the hunt for the ring')
                or batch.get('die_face') in ('C','W') or not rx._is_paid_fellowship_move(batch)):
            continue
        ring,advance=tail[1:3]
        if (rx._ring_rec_kind(ring)!='fp_use' or not rx._ring_use_is_settled(ring,tail)
                or advance.get('op')!='move' or str(advance.get('obj','')).lower()!='the fellowship'
                or any(r.get('player')!=deposit.get('player') for r in (ring,advance))
                or not isinstance(batch.get('truth_before'),dict)
                or not isinstance(batch.get('_rx_pools'),dict)):
            continue
        if any(not r.get('line') or r['line']<=advance['line'] for r in batch.get('rng',[])
               if r.get('kind') not in ('action_roll','alloc_eye','add_eye')):
            continue
        candidates.append((deposit,ring,advance,batch,raw_line,ids[0]))
    if not candidates:return 0
    paired=rr._paired_raw_path(rendered_path,raw_path,lambda reason:None)
    raw=rx._paired_physical_raw_view(batches,read_raw_log_text(paired));count=0
    for deposit,ring,advance,batch,raw_line,pid in candidates:
        meaningful=[(n,t) for n,t in enumerate(raw[raw_line:],raw_line+1) if t.strip()
                    and not ('<~' not in t and re.match(r'^<[^<>]+>',t))][:2]
        if len(meaningful)!=2:continue
        matched=[]
        for (n,t),record in zip(meaningful,(ring,advance)):
            move=lc.MOVE_RE.search(t)
            expected=next((p for p,name in lc.RING_PIECES.items() if name==ring.get('obj')),None) if record is ring else '247'
            if (not move or move[1].split()!=[expected] or t.split('<~',1)[0].lstrip('$')!=deposit['player']
                    or AUTH_LOCATION_CODES.get(move[2])!=record.get('from')
                    or AUTH_LOCATION_CODES.get(move[3])!=record.get('to')):break
            matched.append(n)
        if len(matched)!=2:continue
        original=None;roll_line=None
        for n,t in enumerate(raw[:raw_line-1],1):
            roll=lc.ROLL_RE.search(t)
            if roll and pid in roll[1].split():
                ps,vs=roll[1].split(),roll[2].split()
                if len(ps)!=len(vs):original=None;continue
                value=vs[ps.index(pid)]
                original='WCPMH'[int(value)%10] if value.isdigit() and int(value)%10<5 else None
                roll_line=n
            change=lc.CHANGE_RE.search(t)
            if change and change[1]==pid:original=change[2]
        face=batch.get('die_face')
        if original!=face or roll_line is None or face not in batch['_rx_pools']['FREE_PEOPLES']['unused']:continue
        ring['_rx_fellowship_payment_witness']=dict(kind='direct_hunt_deposit_before_ring',physical_die_id=pid,
            deposit_line=deposit['line'],counter_line=ring['line'],advance_line=advance['line'],
            raw_lines=[raw_line]+matched,roll_raw_line=roll_line,face=face)
        ring['decision_truth']=copy.deepcopy(batch['truth_before'])
        ring['_rx_pools']=copy.deepcopy(batch['_rx_pools'])
        ring['_rx_staged']={'FREE_PEOPLES':[],'SHADOW':[]}
        count+=1
    return count


def annotate_refaced_fellowship_ring(batches, rendered_path, raw_path=None):
    """Use the original die when its explicit reface precedes Ring transfer."""
    import copy,re
    import log_census as lc
    import rendered_extract as rx
    import rendered_replay as rr
    from raw_log_reader import read_raw_log_text
    from location_codes_authoritative import AUTH_LOCATION_CODES
    candidates=[]
    for batch in batches:
        tail=[r for r in batch['records'] if r['line']>=batch['start_line']]
        if len(tail)<3 or not rx._is_paid_fellowship_move(batch):continue
        stage,change=tail[:2]
        delayed=len(tail)>=5 and tail[2].get('op')=='move' and str(tail[2].get('obj','')).lower() in ('fsp','the fellowship')
        ring=tail[4] if delayed else tail[2]
        match=rx._DIE_CHANGE_RE.match(str(change.get('raw','')))
        if (stage.get('die_stage')!=('FREE_PEOPLES',batch.get('die_face'))
                or stage.get('from')!='FP Dice' or stage.get('to')!='Current Action'
                or not match or match[1].lower()!='free' or match[2]!=batch.get('die_face') or match[3]!='C'
                or rx._ring_rec_kind(ring)!='fp_use' or not rx._ring_use_is_settled(ring,tail)
                or ring.get('player')!=stage.get('player')
                or not isinstance(batch.get('_rx_pools'),dict)
                or batch.get('die_face') not in batch['_rx_pools']['FREE_PEOPLES']['unused']):continue
        if delayed:
            advance,deposit=tail[2:4]
            if (deposit.get('op')!='move' or rx._die_tokens(deposit.get('obj',''))!=[('FREE_PEOPLES','C')]
                    or deposit.get('from')!='Current Action' or deposit.get('to')!='The Hunt For The Ring'
                    or any(r.get('player')!=stage['player'] for r in (advance,deposit))):continue
        if any(not r.get('line') or r['line']<=ring['line'] for r in batch.get('rng',[])
               if r.get('kind') not in ('action_roll','alloc_eye','add_eye')):continue
        candidates.append((batch,stage,change,ring,tail[2:4] if delayed else []))
    if not candidates:return 0
    paired=rr._paired_raw_path(rendered_path,raw_path,lambda reason:None)
    raw=rx._paired_physical_raw_view(batches,read_raw_log_text(paired))
    meaningful=[(n,t) for n,t in enumerate(raw,1) if t.strip()
                and not ('<~' not in t and re.match(r'^<[^<>]+>',t))]
    faces={};turn=1;witnesses=[]
    for i,(n,t) in enumerate(meaningful):
        marker=re.search(r'<~Game\.7~>(\d+)\b',t)
        if marker:turn=int(marker[1])
        roll=lc.ROLL_RE.search(t)
        if roll:
            ps,vs=roll[1].split(),roll[2].split()
            if len(ps)==len(vs) and ps and set(ps)<=lc.BASE_FP_DIE_PIDS|lc.KNOWN_BONUS_FP:
                for pid,value in zip(ps,vs):
                    if value.isdigit() and int(value)%10<5:faces[pid]='WCPMH'[int(value)%10]
        stage=lc.MOVE_RE.search(t)
        if stage and i+2<len(meaningful) and (stage[2],stage[3])==('Game.2042','Game.2035'):
            ps=stage[1].split();nc,tc=meaningful[i+1]
            change=lc.CHANGE_RE.search(tc)
            player=t.split('<~',1)[0].lstrip('$')
            if (len(ps)==1 and ps[0] in faces and change and change[1]==ps[0] and change[2]=='C'
                    and tc.startswith('$'+player+' changes ')):
                for offset in (2,4):
                    if i+offset>=len(meaningful):continue
                    nr,tr=meaningful[i+offset];ring=lc.MOVE_RE.search(tr)
                    if not ring or len(ring[1].split())!=1 or ring[1].strip() not in lc.RING_PIECES or tr.split('<~',1)[0].lstrip('$')!=player:continue
                    middle=[]
                    if offset==4:
                        for nm,tm in meaningful[i+2:i+4]:
                            mm=lc.MOVE_RE.search(tm)
                            if not mm or tm.split('<~',1)[0].lstrip('$')!=player:break
                            middle.append(dict(ids=mm[1].split(),origin=AUTH_LOCATION_CODES.get(mm[2]),target=AUTH_LOCATION_CODES.get(mm[3])))
                        if len(middle)!=2 or not set(middle[0]['ids'])<=lc.FSP_MARKER_PIDS or middle[1]['ids']!=ps:continue
                    witnesses.append(dict(turn=turn,player=player,physical_die_id=ps[0],old_face=faces[ps[0]],
                        ring=lc.RING_PIECES[ring[1].strip()],origin=AUTH_LOCATION_CODES.get(ring[2]),
                        target=AUTH_LOCATION_CODES.get(ring[3]),raw_lines=[q[0] for q in meaningful[i:i+offset+1]],middle=middle))
        change=lc.CHANGE_RE.search(t)
        if change and change[1] in faces:faces[change[1]]=change[2]
    count=0
    for batch,stage,change,ring,middle in candidates:
        matches=[w for w in witnesses if w['turn']==batch['turn'] and w['player']==stage['player']
                 and w['old_face']==batch['die_face'] and w['ring']==ring.get('obj')
                 and (w['origin'],w['target'])==(ring.get('from'),ring.get('to'))
                 and len(w['middle'])==len(middle) and all((raw_move['origin'],raw_move['target'])==(record.get('from'),record.get('to')) for raw_move,record in zip(w['middle'],middle))]
        if len(matches)!=1:continue
        ring['_rx_fellowship_payment_witness']=dict(matches[0],kind='explicit_reface_before_ring',
            stage_line=stage['line'],reface_line=change['line'],counter_line=ring['line'])
        ring['decision_truth']=copy.deepcopy(batch['truth_before'])
        ring['_rx_pools']=copy.deepcopy(batch['_rx_pools'])
        ring['_rx_staged']={'FREE_PEOPLES':[],'SHADOW':[]}
        if middle:ring['_rx_fellowship_payment_line']=float(middle[0]['line'])-0.5
        count+=1
    return count
