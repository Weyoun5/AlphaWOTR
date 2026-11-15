"""Conservative physical origin correction after native Undo.

Only board moves with independently paired payloads, a fully known common
physical source, and a preceding undone gesture justify an origin correction.
Mixed origins are refused rather than merged or inventory-clamped.
"""
from collections import Counter
import re


def normalize(records, all_records, raw_lines, rr):
    import log_census as lc
    from client_snapshot_net import snapshot_kind, rendered_snapshot_kind, net_raw, validate_rendered_pair
    from piece_table_authoritative import AUTH_ARMY_PIECES, AUTH_PIECE_LABELS
    from location_codes_authoritative import AUTH_LOCATION_CODES as locations
    validate_rendered_pair(all_records, raw_lines)
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
    retained={r['line']:r for r in records}
    net,events=net_raw(raw_lines,with_events=True)
    undo_targets=[(event['line'],target) for event in events if event['op']=='undo' for target in event['targets']]
    positions={};updates={}

    def counter(tokens):
        result=Counter()
        for token in tokens:
            parsed=rr.parse_piece(token)
            if not parsed or parsed[0] not in ('unit','leader','nazgul','named'):return None
            result[parsed[:3]]+=parsed[3]
        return result

    def raw_counter(ids):
        raw_pieces=Counter()
        for pid in ids:
            if pid in AUTH_ARMY_PIECES:
                nation,kind=AUTH_ARMY_PIECES[pid]
                key=('nazgul','SAURON',None) if kind=='NAZGUL' else ('leader',nation,None) if kind=='LEADER' else ('unit',nation,kind)
                raw_pieces[key]+=1
            else:
                piece=counter([AUTH_PIECE_LABELS.get(pid,'')])
                if piece is None:
                    raw_pieces=None
                    break
                raw_pieces.update(piece)
        return raw_pieces

    def paired_payload(line,text,ids,origin,target):
        record=paired.get(line)
        if not record or record.get('op')!='move':return False
        if len(ids)!=len(set(ids)):return False
        obj=record.get('obj','')
        tokens=rr.split_group(obj[7:-1]) if obj.startswith('group (') and obj.endswith(')') else [obj]
        raw_pieces=raw_counter(ids)
        player=text.split('<~',1)[0].lstrip('$').replace('<auto>','auto>')
        return raw_pieces is not None and raw_pieces==counter(tokens) and record.get('player')==player and rr.zone_kind(record.get('from',''))==rr.zone_kind(locations.get(origin,'')) and rr.zone_kind(record.get('to',''))==rr.zone_kind(locations.get(target,''))

    for line,text in enumerate(net,1):
        match=lc.MOVE_RE.search(text)
        if not match:
            if snapshot_kind(text):
                # Any unsupported operation touching an ID destroys its proof.
                for pid in re.findall(r'(?<!\d)\b\d{3}\b(?!\d)',text):positions.pop(pid,None)
            continue
        ids=match.group(1).split();origin,target=match.group(2),match.group(3)
        actual={pid:positions.get(pid) for pid in ids}
        authenticated=paired_payload(line,text,ids,origin,target)
        for pid in ids:
            if authenticated:positions[pid]=target
            else:positions.pop(pid,None)
        if not ids or not any(pid in AUTH_ARMY_PIECES for pid in ids):continue
        origin_name=locations.get(origin,'');target_name=locations.get(target,'')
        if rr.zone_kind(origin_name)[0] not in ('region','stronghold') or rr.zone_kind(target_name)[0] not in ('region','stronghold'):continue
        if not all(actual.values()) or all(value==origin for value in actual.values()):continue
        witnesses=[]
        for undo_line,target_line in undo_targets:
            if undo_line>=line:continue
            undone=lc.MOVE_RE.search(raw_lines[target_line-1])
            if undone and undone.group(3)==origin and set(ids).intersection(undone.group(1).split()):
                witnesses.append({'undo_raw_line':undo_line,'target_raw_line':target_line})
        if not witnesses:continue
        record=paired.get(line)
        def refuse(detail):
            raise rr.ClientUndoHistoryError({'detail':detail,'raw_line':line,
                'line':record.get('line') if record else None,'physical_ids':ids,
                'claimed_origin':origin_name,'known_origins':{pid:locations.get(value,value) for pid,value in actual.items()}})
        # The rendered duplicate filter may already have removed an adjacent
        # repeated drag. Prove its physical no-op before accepting that omission.
        previous = lc.MOVE_RE.search(net[line-2]) if line > 1 else None
        previous_record = paired.get(line-1)
        if (record and record.get('line') not in retained and authenticated
                and all(value == target for value in actual.values())
                and previous and previous.groups() == match.groups()
                and paired_payload(line-1, net[line-2], ids, origin, target)
                and previous_record and previous_record.get('line') in retained
                and previous_record.get('player') == record.get('player')):
            previous_line = previous_record['line']
            update = dict(updates.get(previous_line, retained[previous_line]))
            update['_physical_duplicate_noop'] = {
                'kept_raw_line': line-1, 'dropped_raw_line': line,
                'dropped_rendered_line': record['line'], 'ids': ids,
                'target': target_name, 'undo_witnesses': witnesses,
            }
            updates[previous_line] = update
            continue
        if not record or record.get('line') not in retained or record.get('op')!='move':refuse('post_undo_move_pair_missing')
        if len(set(ids))!=len(ids):refuse('post_undo_duplicate_physical_id')
        obj=record.get('obj','');tokens=rr.split_group(obj[7:-1]) if obj.startswith('group (') and obj.endswith(')') else [obj]
        if not authenticated:refuse('post_undo_move_identity_or_payload_mismatch')
        sources=set(actual.values())
        if len(sources)!=1:refuse('post_undo_mixed_physical_origins')
        source=next(iter(sources));source_name=locations.get(source,'')
        if rr.zone_kind(source_name)[0] not in ('region','stronghold'):refuse('post_undo_origin_not_on_board')
        update=dict(retained[record['line']]);update['from']=source_name
        update['_physical_undo_origin']={'raw_line':line,'ids':ids,'reported_from':origin_name,'actual_from':source_name,'undo_witnesses':witnesses}
        updates[record['line']]=update
    return [updates.get(record['line'],record) for record in records]
