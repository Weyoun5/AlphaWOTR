"""Prove redundant tile drags by matching rendered moves to physical IDs.

Raw transport is used only to distinguish the same object dragged twice from
another equal-faced tile. It never supplies a draw, bag count, or game choice.
"""
from collections import Counter
import re


def mark_duplicates(records, rendered_path, raw_path=None, raw_lines=None):
    import rendered_replay as rr
    from rendered_mordor_pool import _pieces, _norm, key, tile
    from raw_log_reader import read_raw_log_text
    import log_census as lc
    from log_replay import HUNT_TILE_PIECES
    from location_codes_authoritative import AUTH_LOCATION_CODES
    rendered=[]
    for record in records:
        if record.get('op')!='move':continue
        values=_pieces(record.get('obj',''))
        if not values:continue
        sig=(_norm(record.get('from','')),_norm(record.get('to','')),
             tuple(sorted(Counter(key(t) for t in values).items())))
        rendered.append((record,sig))
    if not any(rendered[i][1] == rendered[i-1][1] for i in range(1,len(rendered))):
        return records
    try:
        paired=rr._paired_raw_path(rendered_path,raw_path,lambda reason:None)
        lines=lc.undo_redo_net(read_raw_log_text(paired).splitlines()) if raw_lines is None else raw_lines
    except (OSError,ValueError,TypeError):
        return records
    physical=[]
    for line,text in enumerate(lines,1):
        match=lc.MOVE_RE.search(text)
        if not match:continue
        ids=match.group(1).split()
        if not ids or any(pid not in HUNT_TILE_PIECES for pid in ids):continue
        values=[tile(HUNT_TILE_PIECES[pid]) for pid in ids]
        if any(t is None for t in values):return records
        origin,target=(AUTH_LOCATION_CODES.get(match.group(i)) for i in (2,3))
        if origin is None or target is None:return records
        sig=(_norm(origin),_norm(target),tuple(sorted(Counter(key(t) for t in values).items())))
        physical.append((line,tuple(sorted(ids)),sig))
    # An unmatched caption or transport gap is no proof; leave ordinary strict
    # history rejection in place instead of guessing an alignment.
    if [r[1] for r in rendered]!=[p[2] for p in physical]:return records
    for i in range(1,len(physical)):
        line,ids,sig=physical[i];prior_line,prior_ids,prior_sig=physical[i-1]
        if line!=prior_line+1 or ids!=prior_ids or sig!=prior_sig:continue
        record=rendered[i][0]
        record['_rx_pool_duplicate_of']=int(rendered[i-1][0]['line'])
        record['_rx_pool_physical_ids']=list(ids)
        record['_rx_pool_raw_lines']=[prior_line,line]
    return records
