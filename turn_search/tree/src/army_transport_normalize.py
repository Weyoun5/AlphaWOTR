"""Normalize two adjacent, fully paired physical army transport shapes.

The original records remain in provenance. This proves only a no-op repeated
move or a Shadow casualty cleanup; it does not forgive arbitrary stale origins.
"""
from collections import Counter
import re


def normalize(records, all_records, raw_lines, rr):
    import log_census as lc
    from client_snapshot_net import snapshot_kind,rendered_snapshot_kind,net_raw,validate_rendered_pair
    from piece_table_authoritative import AUTH_ARMY_PIECES
    from location_codes_authoritative import AUTH_LOCATION_CODES as locations
    net=net_raw(raw_lines)
    kill_re=re.compile(r'<~Controls\.(?:296|304)~>([\d\s]+)<~Controls\.(?:297|305)~><~([\w.]+)~>')
    candidates=[]
    for number in range(2,len(net)+1):
        first=lc.MOVE_RE.search(net[number-2]);second=lc.MOVE_RE.search(net[number-1]);kill=kill_re.search(net[number-1])
        if not first or (not second and not kill):continue
        ids=first.group(1).split();origin,target=first.group(2),first.group(3)
        ids2=(second or kill).group(1).split()
        if not ids or set(ids)!=set(ids2):continue
        if origin not in locations or target not in locations:continue
        if not all(pid in AUTH_ARMY_PIECES and AUTH_ARMY_PIECES[pid][1] in ('REGULAR','ELITE','LEADER') for pid in ids):continue
        origin_kind=rr.zone_kind(locations[origin]);target_kind=rr.zone_kind(locations[target])
        if origin_kind[0] not in ('region','stronghold'):continue
        repeat=bool(second and second.group(2)==origin and second.group(3)==target
                    and origin_kind[0]=='stronghold' and target_kind[0]=='region'
                    and origin_kind[1]==target_kind[1])
        cleanup=bool(kill and target=='Game.2041' and kill.group(2)==origin and all(AUTH_ARMY_PIECES[pid][0] in ('SAURON','ISENGARD','SOUTHRON_EASTERLINGS') for pid in ids))
        if repeat or cleanup:candidates.append(number)
    if not candidates:return records,net
    validate_rendered_pair(all_records,raw_lines)
    visible=[];silent=False
    for number,line in enumerate(raw_lines,1):
        command=line.startswith('$') or line.startswith('<auto>')
        if command and ' silent ' in line:silent=True
        if command and ' noisy ' in line:silent=False
        empty=bool(re.fullmatch(r'\$[^<]+<~Controls\.323~>\s*(?:[A-F0-9]{6}|null)?',line))
        if snapshot_kind(line) and not silent and not empty:visible.append(number)
    rendered=[r for r in all_records if rendered_snapshot_kind(r)]
    if len(visible)!=len(rendered):raise AssertionError('Validated snapshot pairing changed')
    paired=dict(zip(visible,rendered))
    retained={r['line']:i for i,r in enumerate(records)}
    updates={};drop=set()

    def counter(tokens):
        result=Counter()
        for token in tokens:
            piece=rr.parse_piece(token)
            if not piece or piece[0] not in ('unit','leader'):return None
            kind,nation,unit,count=piece
            result[(nation,'LEADER' if kind=='leader' else unit)]+=count
        return result

    def payload(line,ids,origin,target,op):
        rec=paired.get(line)
        if rec is None or rec.get('op')!=op:return False
        actor=net[line-1].split('<~',1)[0].lstrip('$').replace('<auto>','auto>')
        if rec.get('player')!=actor:return False
        if op=='move':
            cargo=rec.get('obj','');tokens=rr.split_group(cargo[7:-1]) if cargo.startswith('group (') and cargo.endswith(')') else [cargo]
            if rr.zone_kind(rec.get('from',''))!=rr.zone_kind(locations[origin]) or rr.zone_kind(rec.get('to',''))!=rr.zone_kind(locations[target]):return False
        else:
            tokens=rec.get('pieces',[])
            if rr.zone_kind(rec.get('zone',''))!=rr.zone_kind(locations[origin]):return False
        return counter(tokens)==Counter(AUTH_ARMY_PIECES[pid] for pid in ids)

    for number in candidates:
        first=lc.MOVE_RE.search(net[number-2]);second=lc.MOVE_RE.search(net[number-1]);kill=kill_re.search(net[number-1])
        if not first or (not second and not kill):continue
        ids=first.group(1).split();origin,target=first.group(2),first.group(3)
        ids2=(second or kill).group(1).split()
        if not ids or set(ids)!=set(ids2):continue
        if origin not in locations or target not in locations:continue
        if not all(pid in AUTH_ARMY_PIECES and AUTH_ARMY_PIECES[pid][1] in ('REGULAR','ELITE','LEADER') for pid in ids):continue
        if rr.zone_kind(locations[origin])[0] not in ('region','stronghold'):continue
        repeat=bool(second and second.group(2)==origin and second.group(3)==target
                    and rr.zone_kind(locations[origin])[0]=='stronghold'
                    and rr.zone_kind(locations[target])[0]=='region'
                    and rr.zone_kind(locations[origin])[1]==rr.zone_kind(locations[target])[1])
        cleanup=bool(kill and target=='Game.2041' and kill.group(2)==origin and all(AUTH_ARMY_PIECES[pid][0] in ('SAURON','ISENGARD','SOUTHRON_EASTERLINGS') for pid in ids))
        if not (repeat or cleanup):continue
        rec1,rec2=paired.get(number-1),paired.get(number)
        def refuse(detail):
            raise rr.PhysicalRecruitIdentityError({'detail':detail,'raw_line':number,'line':rec2.get('line') if rec2 else None,'physical_ids':ids})
        if len(ids)!=len(set(ids)) or len(ids2)!=len(set(ids2)):refuse('adjacent_army_duplicate_physical_id')
        if not rec1 or not rec2 or rec1['line'] not in retained or rec2['line'] not in retained:refuse('adjacent_army_pair_not_retained')
        if retained[rec2['line']]!=retained[rec1['line']]+1:refuse('adjacent_army_rendered_interposition')
        if not payload(number-1,ids,origin,target,'move') or not payload(number,ids2,origin,target,'move' if repeat else 'kill'):refuse('adjacent_army_payload_mismatch')
        proof={'kind':'same_target_noop' if repeat else 'casualty_cleanup','raw_lines':[number-1,number],
            'physical_ids':ids,'original_records':[dict(rec1),dict(rec2)],'original_raw':[net[number-2],net[number-1]]}
        if repeat:
            update=dict(updates.get(rec1['line'],rec1));update['_physical_adjacent_transport']=proof
            updates[rec1['line']]=update;drop.add(rec2['line']);net[number-1]=''
        else:
            update=dict(rec2);update['zone']=locations[target];update['_physical_adjacent_transport']=proof
            updates[rec2['line']]=update
            text=net[number-1];lo,hi=kill.span(2);net[number-1]=text[:lo]+target+text[hi:]
    return [updates.get(r['line'],r) for r in records if r['line'] not in drop],net
