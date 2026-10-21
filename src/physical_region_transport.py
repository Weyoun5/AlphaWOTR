"""Correct physical figure transport with paired native identities and origins.

The client assigns destinations to physical figures; its printed source can
remain the source of an earlier drag. Counting both printed moves duplicates
units. This module preserves both gestures and their provenance while using
the last proved physical location as the second gesture's source.
"""
from collections import Counter
import re


def normalize(records, all_records, raw_lines, rr):
    from client_snapshot_net import net_raw, snapshot_kind, rendered_snapshot_kind, validate_rendered_pair
    from piece_table_authoritative import AUTH_ARMY_PIECES, AUTH_PIECE_LABELS
    from physical_initial_locations import INITIAL_FIGURE_LOCATIONS, proves_base2_opening
    from location_codes_authoritative import AUTH_LOCATION_CODES as locations
    import log_census as lc
    descriptors={}
    def identity_key(piece):
        kind,name,unit=piece[:3]
        if kind=='named':
            name=rr.COMP_ALIAS.get(str(name).lower(),name)
            # Crowning/return changes the caption, not the physical piece ID.
            name={'Aragorn':'Strider','Gandalf the White':'Gandalf the Grey'}.get(name,name)
        return kind,name,unit
    for pid,(nation,kind) in AUTH_ARMY_PIECES.items():
        descriptors[pid]=('nazgul','SAURON',None) if kind=='NAZGUL' else (
            ('leader',nation,None) if kind=='LEADER' else ('unit',nation,kind))
    for pid,label in AUTH_PIECE_LABELS.items():
        if pid not in descriptors:
            label={'Gandalf':'Gandalf the Grey','Fellowship':'The Fellowship'}.get(label,label)
            piece=rr.parse_piece(label)
            if label=='The Fellowship':descriptors[pid]=('fellowship',None,None)
            elif piece and piece[0]=='named':descriptors[pid]=identity_key(piece)
    net=net_raw(raw_lines)
    known={};seen=set();candidates=[];initial_setup_line=None
    kill_re=re.compile(r'<~Controls\.(?:296|304|644)~>([\d\s]+)<~Controls\.(?:297|305|645)~><~([\w.]+)~>')
    commands={}
    for number,line in enumerate(net,1):
        if '<~Controls.408~>' in line or '<~Controls.72~>' in line:
            known.clear()
            initial_setup_line=number if proves_base2_opening(net,number-1) else None
            seen=set() if initial_setup_line else set(descriptors)
            continue
        command=line.startswith(('$','<auto>'))
        move=lc.MOVE_RE.search(line) if command else None
        kill=kill_re.search(line) if command else None
        if move is None and kill is None:
            if snapshot_kind(line):
                # Card/die snapshots do not move army figures. Unknown commands
                # mentioning a figure invalidate that figure, never reconstruct
                # its location from an older caption or initial setup afterward.
                payload=line.split('~>',1)[-1].split('<~',1)[0]
                affected=set(re.findall(r'(?<!\S)\d{3}(?!\S)',payload))
                for pid in affected:known.pop(pid,None)
                seen.update(affected)
            elif command and '<~' in line and not re.search(r'<~Controls\.(?:345|840|464|479)~>',line):
                # Unrecognized state commands may replace the whole board.
                # Java Controls 345 rolls dice, 840 selects a card; 464/479
                # transfer private deck tokens. None moves physical figures.
                known.clear();seen.update(descriptors);initial_setup_line=None
            continue
        ids=(move or kill).group(1).split()
        origin=(move or kill).group(2);target=move.group(3) if move else None
        actor=line.split('<~',1)[0].lstrip('$').replace('<auto>','auto>')
        commands[number]=(ids,origin,target,actor)
        supported=bool(ids and len(set(ids))==len(ids) and all(pid in descriptors for pid in ids)
                       and origin in locations and (target is None or target in locations))
        previous={pid:known.get(pid) for pid in ids}
        if move:previous={pid:p if p and not p.get('killed') else None for pid,p in previous.items()}
        changed=any(p and p['target']!=origin for p in previous.values())
        duplicate_kill=bool(kill and any(p and p.get('killed') for p in previous.values()))
        fellowship_group=bool(move and len(ids)>1 and any(descriptors.get(pid)==('fellowship',None,None) for pid in ids))
        # The existing physical recruit-run normalizer owns reserve-origin
        # transcripts and their count pairing. Do not rewrite its input census.
        if supported and rr.zone_kind(locations[origin])[0]!='reinf' and (changed or fellowship_group):
            # Unmoved starting figures can join already moved figures in a
            # group. Only untouched IDs at their exact canonical starting box
            # qualify; previously invalidated/unknown histories never do.
            for pid in ids:
                if previous[pid] is None and initial_setup_line and pid not in seen and INITIAL_FIGURE_LOCATIONS.get(pid)==origin:
                    previous[pid]={'target':origin,'line':0,'actor':None}
            if all(p is not None for p in previous.values()) or (fellowship_group and not changed) or duplicate_kill:
                groups={};prior_lines=set();initial=[]
                for pid in ids:
                    prior=previous[pid]
                    source=prior['target'] if prior and not prior.get('killed') else origin
                    family='killed' if prior and prior.get('killed') else ('fellowship' if descriptors[pid][0]=='fellowship' else 'army')
                    groups.setdefault((source,family),[]).append(pid)
                    if prior and prior['line']:prior_lines.add(prior['line'])
                    elif prior:initial.append(pid)
                candidates.append((number,ids,origin,target,groups,actor,sorted(prior_lines),initial))
        if move:
            for pid in ids:known[pid]={'target':target,'line':number,'actor':actor}
        else:
            for pid in ids:known[pid]={'target':None,'line':number,'actor':actor,'killed':True}
        seen.update(ids)
    if not candidates:return records,net
    # Kind sequence alone is insufficient for payload identity, so every
    # changed record below additionally matches actor, zones and exact cargo.
    validate_rendered_pair(all_records,raw_lines)
    visible=[];silent=False
    for number,line in enumerate(raw_lines,1):
        command=line.startswith(('$','<auto>'))
        if command and ' silent ' in line:silent=True
        if command and ' noisy ' in line:silent=False
        empty=bool(re.fullmatch(r'\$[^<]+<~Controls\.323~>\s*(?:[A-F0-9]{6}|null)?',line))
        if snapshot_kind(line) and not silent and not empty:visible.append(number)
    rendered=[r for r in all_records if rendered_snapshot_kind(r)]
    paired=dict(zip(visible,rendered));retained={r['line']:r for r in records}
    updates={}
    def pieces(record):
        obj=record.get('obj','')
        tokens=(record.get('pieces',[]) if record.get('op')=='kill' else
                rr.split_group(obj[7:-1]) if obj.startswith('group (') and obj.endswith(')') else [obj])
        actual=Counter();captions={}
        for token in tokens:
            clean=re.sub(r'^\[\d+\]\s*','',token).strip()
            piece=rr.parse_piece(token)
            if clean.lower() in ('the fellowship','fellowship'):
                key=('fellowship',None,None);count=1
            elif piece and piece[0] in ('unit','leader','nazgul','named'):
                key=identity_key(piece);count=piece[3]
            else:return None,None
            actual[key]+=count;captions[key]=clean
        return actual,captions
    def payload(record,ids,origin,target,actor):
        if record is None or record.get('op')!=('kill' if target is None else 'move') or record.get('player')!=actor:return False
        if rr.zone_kind(record.get('zone' if target is None else 'from',''))!=rr.zone_kind(locations[origin]):return False
        if target is not None and rr.zone_kind(record.get('to',''))!=rr.zone_kind(locations[target]):return False
        actual,_=pieces(record)
        return actual==Counter(descriptors[pid] for pid in ids)
    for number,ids,origin,target,groups,actor,prior_lines,initial in candidates:
        record=paired.get(number)
        if record is None or record['line'] not in retained:continue
        if not payload(record,ids,origin,target,actor):
            raise rr.PhysicalRecruitIdentityError({'detail':'region_transport_pair_payload','raw_line':number})
        for prior in prior_lines:
            prior_ids,prior_origin,prior_target,prior_actor=commands[prior]
            if (prior_origin not in locations or (prior_target is not None and prior_target not in locations)
                    or not all(pid in descriptors for pid in prior_ids)
                    or not payload(paired.get(prior),prior_ids,prior_origin,prior_target,prior_actor)):
                raise rr.PhysicalRecruitIdentityError({'detail':'region_transport_prior_payload','raw_line':prior})
        _,captions=pieces(record);parts=[]
        for (source,family),group in groups.items():
            if source not in locations:raise rr.PhysicalRecruitIdentityError({'detail':'region_transport_unknown_actual_zone','raw_line':number})
            update=dict(retained[record['line']])
            update['zone' if target is None else 'from']=locations[source]
            counts=Counter(descriptors[pid] for pid in group)
            tokens=[('['+str(count)+']' if count>1 else '')+captions[key] for key,count in counts.items()]
            if target is None:update['pieces']=tokens
            else:update['obj']=tokens[0] if len(tokens)==1 else 'group ('+', '.join(tokens)+')'
            update['_physical_region_transport']={'raw_line':number,'physical_ids':group,
                'prior_raw_lines':prior_lines,'initial_physical_ids':[pid for pid in group if pid in initial],
                'printed_origin':locations[origin],'actual_origin':locations[source],
                'destination':locations[target] if target is not None else None,
                'original_record':dict(record),'kind':'physical_split_origin_correction' if len(groups)>1 else 'physical_same_origin_correction'}
            if family=='killed':update['op']='physical_kill_noop'
            elif source==target:update['op']='physical_move_noop'
            parts.append(update)
        updates[record['line']]=parts
    return [part for r in records for part in updates.get(r['line'],[r])],net
