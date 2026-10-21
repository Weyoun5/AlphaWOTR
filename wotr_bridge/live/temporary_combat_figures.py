"""Exact physical Companion parking witnessed by a committed combat choice."""
import copy


def _zone(value):
    from board import sanitize
    value=sanitize(str(value or ''))
    return {'fp card play':'current fp card','sa card play':'current sa card'}.get(value,value)


def _location(state, name):
    for region, army in state.armies.items():
        if any(getattr(leader, 'name', None)==name for leader in army.leaders):
            return _zone(region)
    return None


def _context(state, binding):
    if int(state.turn)!=binding['turn']:
        return False
    pc=state.pending_combat or {}
    return not pc or tuple(_zone(pc.get(key)) for key in ('attacker_region','defender_region'))==tuple(binding['battle'])


def prepare(view, state, side, window, chain):
    """Resolve a proved gesture to one actual native PID before committing."""
    if window is None:
        return []
    from .mighty_gestures import park_witness
    from .sidecar import meta_pids
    from . import client_board
    import rendered_replay as rr
    import log_replay as lr
    result=[]
    for action in chain:
        witness=park_witness(state,side,window,action)
        if witness is None:
            continue
        record=witness['record'];name=witness['name']
        candidates=[str(pid) for pid in lr.COMPANIONS
                    if name in {rr.COMP_ALIAS.get(str(alias).lower(),str(alias))
                                for alias in lr.names_for_pid(str(pid))}]
        pids=[str(pid)for pid in (meta_pids(record.get('meta')) or ())]
        if pids:
            candidates=[pid for pid in candidates if pid in pids]
        # Physical dump confirms the exact public figure is actually parked.
        # This is also the identity proof for old clients with empty sidecars.
        dump=client_board.read_board(view.dump_path) if view.dump_path else None
        if dump is None:
            dump=getattr(view,'native_piece_region',None) or view.piece_region
        candidates=[pid for pid in candidates if _zone(dump.get(pid))==_zone(witness['destination_zone'])]
        if len(candidates)!=1:
            raise ValueError('Temporary Companion placement has no unique native PID witness')
        pc=state.pending_combat or {}
        binding={key:witness[key] for key in ('name','target','source_region','source_zone',
            'destination_zone','card_id','rounds_fought','owner')}
        binding.update(pid=candidates[0],turn=int(state.turn),
            battle=[_zone(pc.get(key))for key in ('attacker_region','defender_region')],
            record_norm=record.get('norm',''),player=record.get('player',''),
            native_zone=_zone(witness['destination_zone']))
        result.append(binding)
    return result


def register(view, bindings):
    if not bindings:
        return
    current=getattr(view,'temporary_combat_figures',None)
    if current is None:
        current=view.temporary_combat_figures={}
    for binding in bindings:
        current[binding['pid']]=copy.deepcopy(binding)
    view.piece_region=project_dump(view,view.piece_region)


def project_dump(view, dump):
    """A bound play-area PID retains its living figure's logical position."""
    result=dict(dump);state=view.state
    for pid,binding in list(getattr(view,'temporary_combat_figures',{}).items()):
        physical=_zone(dump.get(pid))
        # A projected view passed twice must not overwrite physical evidence.
        if physical in ('current fp card','current sa card'):
            binding['native_zone']=physical
        elif physical!=_zone(binding['source_zone']):
            binding['native_zone']=physical
        if (_location(state,binding['name']) is None
                and physical in ('fp casualties','fp dead','dead')):
            view.temporary_combat_figures.pop(pid,None)
            continue
        if physical!=_zone(binding['destination_zone']) or not _context(state,binding):
            continue
        current=_location(state,binding['name'])
        if current is not None:
            suffix=' stronghold' if _zone(binding['source_zone']).endswith(' stronghold') else ''
            result[pid]=current+suffix
    return result


def waiting(view, state):
    """Only a completed round requires the living parked figure back now."""
    result=[]
    for binding in getattr(view,'temporary_combat_figures',{}).values():
        if binding.get('native_zone')!=_zone(binding['destination_zone']):
            continue
        if _location(state,binding['name']) is None:
            continue  # The ordinary casualty ledger owns an actual death.
        pc=state.pending_combat or {}
        # The engine increments rounds_fought after rolls, BEFORE ordinary
        # casualties and deferred card attacks. Those are still this round.
        # Mandatory event rounds may go straight to the next card selection.
        boundaries={'combat_round_result','attacker_continue','defender_retreat',
                    'victor_advance','attacker_combat_card','defender_combat_card',
                    'fp_shine'}
        complete=(int(pc.get('rounds_fought',0))>int(binding['rounds_fought'])
                  and pc.get('phase') in boundaries
                  and not any(pc.get(key) for key in
                              ('_round_resume','_p9','effect_choice')))
        if not _context(state,binding) or not pc or complete:
            result.append(binding)
    return result


def return_zone(state,binding):
    region=_location(state,binding['name'])
    if region is None:
        return None
    return region+(' stronghold' if _zone(binding['source_zone']).endswith(' stronghold') else '')


def consume_return(view, state, record):
    if record.get('op')!='move':
        return False
    from .sidecar import meta_pids
    import rendered_replay as rr
    piece=rr.parse_piece(str(record.get('obj','')))
    if not piece or piece[0]!='named' or piece[3]!=1:
        return False
    pids=[str(pid)for pid in (meta_pids(record.get('meta')) or ())]
    for pid,binding in list(getattr(view,'temporary_combat_figures',{}).items()):
        if (piece[1]!=binding['name'] or (pids and pids!=[pid])
                or record.get('player','')!=binding['player']
                or _zone(record.get('from'))!=_zone(binding['destination_zone'])
                or not _context(state,binding)):
            continue
        expected=return_zone(state,binding)
        if expected is None or _zone(record.get('to'))!=expected:
            continue
        view.temporary_combat_figures.pop(pid)
        view.piece_region[pid]=record['to']
        return True
    return False


def remove_park_fact(view, gesture, facts):
    """Temporary parking is not payment for a subsequent genuine death."""
    norms={record.get('norm','')for record in getattr(gesture,'records',())}
    for binding in getattr(view,'temporary_combat_figures',{}).values():
        if binding.get('record_norm') and binding['record_norm'] in norms:
            key=(binding['name'],_zone(binding['source_region']))
            if facts.named.get(key,0)<0:
                facts.named[key]+=1


def snapshot(view):
    return copy.deepcopy(getattr(view,'temporary_combat_figures',{}))


def restore(view, values):
    view.temporary_combat_figures=copy.deepcopy(values or {})
