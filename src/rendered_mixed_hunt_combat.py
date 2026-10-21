"""Keep a proved Fellowship action separate from an unproved combat tail."""
from copy import deepcopy


def _shadow_die_evidence(records, after_line):
    """Literal tail gestures only; availability never proves a die was spent."""
    import rendered_replay as rr
    evidence=[]
    for record in records:
        if int(record.get('line',0))<=after_line or record.get('op') not in ('move','kill'):
            continue
        captions=([record.get('obj','')] if record['op']=='move' else record.get('pieces',()))
        for caption in captions:
            text=str(caption).strip()
            tokens=(rr.split_group(text[text.find('(')+1:text.rfind(')')])
                    if text.lower().startswith('group (') else [text])
            for token in tokens:
                piece=rr.parse_piece(token)
                if not piece or piece[0]!='die' or 'shadow action die' not in token.lower():
                    continue
                origin=record.get('from',record.get('zone',''))
                target=record.get('to')
                paid=(record['op']=='kill' and str(origin).lower() in ('current action','sa dice')) or (
                    record['op']=='move' and str(origin).lower() in ('current action','sa dice')
                    and rr.zone_kind(target or '')[0]=='cas')
                evidence.append({'line':int(record['line']),'op':record['op'],
                    'caption':token,'from':origin,'to':target,
                    'kind':'payment_gesture' if paid else 'unresolved_die_gesture'})
    return evidence


def project_hunt_prefix(batch):
    """Return None unless an FP movement batch also contains later combat.

    No action die is invented for the combat. A returned result either proves
    the completed ordinary-tile Hunt prefix or refuses the mixed batch. The
    original parsed records stay intact for public history and later states.
    """
    import rendered_replay as rr
    if batch.get('kind') not in ('combat', 'combat?') or batch.get('die_side') != 'FREE_PEOPLES':
        return None
    records=batch.get('records',())
    moves=[r for r in records if r.get('op')=='move'
           and str(r.get('obj','')).strip().lower() in ('the fellowship','fsp')]
    tiles=[r for r in batch.get('rng',()) if r.get('kind')=='draw_tile']
    rolls=[r for r in batch.get('rng',()) if r.get('kind')=='roll_num']
    if not moves or not tiles or not rolls or not (int(moves[0]['line'])<int(tiles[0]['line'])<int(rolls[0]['line'])):
        return None
    losses=[r for r in records if r.get('op')=='kill' and int(r['line'])>int(rolls[0]['line'])
            and rr.zone_kind(r.get('zone',''))[0] in ('region','stronghold')
            and any((p:=rr.parse_piece(t)) and p[0]=='unit' for t in r.get('pieces',()))]
    if not losses:
        return None
    result={'batch':None,'reason':'mixed_hunt_combat_prefix_unproven',
            'source_line':int(batch['start_line']),'tile_line':int(tiles[0]['line']),
            'combat_first_roll':int(rolls[0]['line'])}
    commits=[r for r in records if r.get('op')=='move'
             and '(chosen)' in str(r.get('from','')).lower()
             and str(r.get('to','')).lower() in ('current fp card','current sa card')
             and int(tiles[0]['line'])<int(r['line'])<int(rolls[0]['line'])]
    if not commits or batch.get('die_face')!='C' or len(tiles)!=1:
        return result
    boundary=int(commits[0]['line']);truth=commits[0].get('decision_truth')
    if not truth or not str(tiles[0].get('value','')).isdigit():
        return result
    prefix_records=[r for r in records if int(r['line'])<boundary]
    staged=[r for r in prefix_records if r.get('op')=='move'
            and int(r['line'])==int(batch['start_line'])
            and r.get('obj')=='Free Action Die (C)' and r.get('from')=='FP Dice'
            and r.get('to')=='Current Action']
    parked=[r for r in prefix_records if r.get('op')=='move'
            and r.get('obj')=='Free Action Die (C)' and r.get('from')=='Current Action'
            and r.get('to')=='The Hunt For The Ring'
            and int(moves[0]['line'])<int(r['line'])<int(tiles[0]['line'])]
    cleaned=[r for r in prefix_records if r.get('op')=='kill'
             and r.get('pieces')==[str(tiles[0]['value'])]
             and int(r['line'])>int(tiles[0]['line'])]
    if len(staged)!=1 or len(parked)!=1 or len(cleaned)!=1:
        return result
    die_evidence=_shadow_die_evidence(records,int(cleaned[0]['line']))
    if any(item['kind']=='payment_gesture' for item in die_evidence):
        tail_reason='mixed_hunt_combat_tail_payment_observed'
        payment_status='shadow_die_payment_observed_replay_unproven'
    elif die_evidence:
        tail_reason='mixed_hunt_combat_tail_die_gesture_unresolved'
        payment_status='shadow_die_gesture_unresolved'
    else:
        tail_reason='mixed_hunt_combat_unfunded_tail'
        payment_status='no_shadow_die_gesture_observed'
    result.update(tail_payment_status=payment_status,tail_die_evidence=die_evidence)
    # All physical changes after tile cleanup would need a separate owner.
    if any(r.get('op') not in ('choose','chat','route_hint')
           for r in prefix_records if int(r['line'])>int(cleaned[0]['line'])):
        return result
    before=batch.get('truth_before') or {}
    fel0,fel1=before.get('fel',{}),truth.get('fel',{})
    corruption=[r for r in prefix_records if r.get('op')=='move' and r.get('obj')=='Corruption'
                and int(tiles[0]['line'])<int(r['line'])<int(cleaned[0]['line'])]
    if len(corruption)!=1:
        return result
    for field,value in (('from',fel0.get('corruption')),('to',fel1.get('corruption'))):
        if str(corruption[0].get(field,'')) != 'Fellowship Track '+str(value):
            return result
    if not (fel0.get('mordor') and fel1.get('mordor')
            and int(fel1.get('step',-10))==int(fel0.get('step',-10))+1
            and int(fel1.get('corruption',-10))-int(fel0.get('corruption',-10))==int(tiles[0]['value'])
            and fel1.get('hidden')==fel0.get('hidden') and fel1.get('comps')==fel0.get('comps')):
        return result
    prefix=deepcopy(batch)
    prefix['records']=deepcopy(prefix_records)
    prefix['rng']=[deepcopy(r) for r in batch['rng'] if int(r['line'])<boundary]
    prefix['end_line']=boundary
    prefix['truth']=deepcopy(truth)
    prefix['truth']['_rx_hunt_observation']=(boundary,'snapshot')
    prefix['kind']=rr.classify_batch(prefix)
    if prefix['kind']!='hunt':
        return result
    # A separate spent-die gesture still needs its own action replay. It may
    # not authorize borrowing the FP die or assigning the siege to FP.
    result.update(batch=prefix,reason=tail_reason,
                  prefix_end=boundary,post_hunt_fellowship=deepcopy(fel1),
                  original_funding={'side':batch['die_side'],'face':batch['die_face']},
                  combat_rows_emitted=False)
    return result
