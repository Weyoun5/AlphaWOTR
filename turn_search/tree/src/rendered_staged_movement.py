"""Rendered-only TSL/TSiM movement provenance and public-menu matching."""
from collections import Counter, OrderedDict
import re
import ai,game_logic as gl,game_state as gs,rendered_extract as rx,rendered_replay as rr

NAMES={'the shadow lengthens','the shadow is moving','shadows gather','corsairs of umbar'}
class Unproven(ValueError):pass


def corsairs_trim_window(batch, region):
    """Only landing removals before combat can label the pre-battle trim."""
    plays = rx._card_play_titles(batch)
    if not plays or str(plays[0][1]).split('/')[0].strip().lower() != 'corsairs of umbar':
        raise Unproven('corsairs_trim_play_unproven')
    start = int(plays[0][0])
    commitments = [int(line) for line, _ in plays[1:] if int(line) > start]
    commitments.extend(int(r['line']) for r in batch['records']
                       if int(r.get('line', 0)) > start and rx._is_card_play_move(r))
    commitments.extend(int(r['line']) for r in batch.get('rng', ())
                       if r.get('kind') == 'roll_num' and int(r.get('line', 0)) > start)
    end = min(commitments, default=max(int(r.get('line', 0)) for r in batch['records']) + 1)
    destination = rr.sanitize(str(region))
    arrivals = [int(r['line']) for r in batch['records']
                if r.get('op') == 'move' and start < int(r.get('line', 0)) < end
                and rr.zone_kind(r.get('from', '')) == ('region', 'umbar')
                and rr.zone_kind(r.get('to', ''))[0] in ('region', 'stronghold')
                and rr.zone_kind(r.get('to', ''))[1] == destination
                and any(token_cargo(token)[0] is not None for token in rx._group_tokens(str(r.get('obj', ''))))]
    if not arrivals:
        raise Unproven('corsairs_trim_landing_unproven')
    return dict(batch, records=[r for r in batch['records'] if max(arrivals) < int(r.get('line', 0)) < end])

def named(name):
    compact=re.sub('[^a-z]','',str(name).lower())
    if compact in ('witchking','thewitchking','witchkingofangmar'):return 'Witch King'
    if compact in ('mouthofsauron','themouthofsauron'):return 'Mouth of Sauron'
    return str(name)

def army_cargo(state,region,regular=None,elite=None):
    army=ai._armies_lookup(state,region)
    cargo=Counter()
    if army is None:return cargo
    limits={gs.UnitKind.REGULAR:regular,gs.UnitKind.ELITE:elite}
    for unit in army.units:
        if unit.nation not in (gs.Nation.SAURON,gs.Nation.ISENGARD,gs.Nation.SOUTHRON_EASTERLINGS):continue
        left=limits[unit.kind]
        if left is not None:
            if left<=0:continue
            limits[unit.kind]-=1
        cargo[('unit',unit.nation.name,unit.kind.name)]+=1
    for leader in army.leaders:
        if gl._is_saruman(leader) or gl._leader_side(leader)!=gs.Side.SHADOW:continue
        if isinstance(leader,gs.Nazgul):key=('nazgul','SAURON',None)
        elif getattr(leader,'name',None):key=('named',named(leader.name),None)
        else:key=('leader',leader.nation.name,None)
        cargo[key]+=1
    return cargo

def token_cargo(token):
    piece=rr.parse_piece(token)
    if not piece:return None,0
    kind,name,subtype,count=piece
    if kind not in ('unit','nazgul','named','leader'):return None,0
    if kind=='named':name=named(name)
    return (kind,name,subtype),int(count)

def _shadow_drag(record):
    """The (origin, target) edge and the Shadow army cargo a move record drags
    between two regions, or None when it drags none."""
    if record.get('op')!='move':return None
    origin,target=rr.zone_kind(record.get('from','')),rr.zone_kind(record.get('to',''))
    if origin[0] not in ('region','stronghold') or target[0] not in ('region','stronghold'):return None
    if origin[1]==target[1]:return None
    cargo=Counter()
    for token in rx._group_tokens(record.get('obj','')):
        key,count=token_cargo(token)
        if key is not None and (key[0] == 'nazgul'
            or key[0] == 'named' and key[1] in ('Witch King','Mouth of Sauron')
            or key[0] in ('unit','leader') and key[1] in ('SAURON','ISENGARD','SOUTHRON_EASTERLINGS')):
            cargo[key]+=count
    if not cargo:return None
    return (origin[1],target[1]),cargo


def _identical_swap(timeline, state):
    """R3-TSL-IDENTICAL-SWAP: The Shadow Lengthens' forced reciprocal swap.

    USER D2 / ruling 1.279 makes both moves mandatory while a join exists and
    USER R2-TSL (2026-10-23) keeps the reciprocal swap, so 'X -> Y' followed by
    'Y -> X' with equal cargo can be Y's own army trading places, not a
    drag-back. Figure IDs cannot tell the two apart; the swap reading is taken
    only when the window moved exactly those two records and BOTH regions'
    start-of-action Shadow armies can supply that cargo (otherwise the returning
    figures can only be the provisional move dragged back). Returns the two
    plans, or None for the drag-back reading.
    """
    if len(timeline) != 2:
        return None
    (edge, info), (back, back_info) = timeline
    if back != (edge[1], edge[0]) or info['cargo'] != back_info['cargo']:
        return None
    cargo = info['cargo']
    if not any(key[0] == 'unit' and count > 0 for key, count in cargo.items()):
        return None
    plans = []
    for (source, target), line in ((edge, info['line']), (back, back_info['line'])):
        available = army_cargo(state, source)
        if any(count > available[key] for key, count in cargo.items()):
            return None
        plans.append({'from': source, 'to': target,
                      'regular': sum(n for k, n in cargo.items() if k[0] == 'unit' and k[2] == 'REGULAR'),
                      'elite': sum(n for k, n in cargo.items() if k[0] == 'unit' and k[2] == 'ELITE'),
                      'whole': cargo == available, 'cargo': Counter(cargo), 'path': (source, target),
                      'paths': ((source, target),), 'line': line})
    return tuple(plans)


def _piecewise_swap(timeline, state):
    """R4-EXTRACT-COVERAGE-DROPS (R3-TSL-IDENTICAL-SWAP FU3): the same forced
    reciprocal swap dragged in pieces (e.g. 'X -> Y' as two drags of one
    Regular each, then 'Y -> X' as one drag of two). No drag-back cancels, the
    merged walk reads the returning cargo as the first army forwarded back to
    its own origin and refuses with tsl_origin_or_through_ambiguous. The swap
    reading is taken only when EVERY movement of the window lies on the two
    reciprocal edges, both directions carry the same total cargo (with a unit)
    and both regions' start-of-action Shadow armies can supply it - the
    _identical_swap test on the per-direction totals. Returns the two plans
    (first drag of each direction gives its order), or None.
    """
    if len(timeline) < 3:
        return None
    edge = timeline[0][0]
    back = (edge[1], edge[0])
    totals, lines = {edge: Counter(), back: Counter()}, {}
    for moved, info in timeline:
        if moved not in totals:
            return None
        totals[moved].update(info['cargo'])
        lines.setdefault(moved, info['line'])
    if back not in lines or totals[edge] != totals[back]:
        return None
    return _identical_swap(((edge, {'cargo': totals[edge], 'line': lines[edge]}),
                            (back, {'cargo': totals[back], 'line': lines[back]})), state)


def literal_plan(batch,state,title):
    name=str(title).split('/')[0].strip().lower()
    if name not in NAMES:raise Unproven('wrong_card')
    plays=rx._card_play_titles(batch)
    if not plays or str(plays[0][1]).split('/')[0].strip().lower()!=name:
        raise Unproven('play_window_ambiguous')
    play_line=int(plays[0][0]);closed=[]
    for record in batch['records']:
        if record.get('op')=='kill' and int(record.get('line',0))>play_line:
            tokens=[token for piece in record.get('pieces',[]) for token in rx._group_tokens(str(piece))]
            if any(token.split('/')[0].strip().lower()==name for token in tokens):closed.append(int(record['line']))
    if not closed:raise Unproven('card_not_closed')
    first_close = min(closed)
    subsequent = [int(line) for line, _ in plays[1:] if int(line) < first_close]
    if subsequent and name != 'corsairs of umbar':
        raise Unproven('play_window_ambiguous')
    boundary = first_close
    if name == 'corsairs of umbar':
        commitments = subsequent + [int(r['line']) for r in batch.get('rng', ())
            if int(r.get('line',0)) > play_line and r.get('kind') == 'roll_num']
        boundary = min([first_close] + commitments)
    timeline=[]
    for record in batch['records']:
        if not play_line < int(record.get('line',0)) < boundary:continue
        drag=_shadow_drag(record)
        if drag is None:continue
        timeline.append((drag[0],{'cargo':drag[1],'line':int(record['line'])}))
    # R17-CORPUS-CENSUS: the window ends at the card's first closing kill, so
    # a Shadow army drag logged after the discard in the same batch, before
    # anything else commits (a later card play; for Corsairs also its battle
    # roll), is this card's movement or a correction of it that the plan
    # cannot see. Refuse the window rather than label the truncated movement.
    # A Corsairs window cut by its battle roll keeps its post-battle moves out.
    if boundary == first_close:
        later = [int(line) for line, _ in plays[1:] if int(line) > first_close]
        if name == 'corsairs of umbar':
            later += [int(r['line']) for r in batch.get('rng', ())
                      if int(r.get('line',0)) > first_close and r.get('kind') == 'roll_num']
        end = min(later, default=float('inf'))
        if any(first_close < int(r.get('line',0)) < end and _shadow_drag(r) is not None
               for r in batch['records']):
            raise Unproven('movement_after_closure')
    # Literal drag-backs cancel a provisional move only when no intervening
    # movement touched that category at either endpoint. Figure IDs are not
    # inferred across a merge or an intervening transfer.
    corrected=[]
    for edge, info in timeline:
        reverse = (edge[1], edge[0])
        matched = None
        for index in range(len(corrected)-1,-1,-1):
            prior_edge, prior = corrected[index]
            if prior_edge == reverse and prior['cargo'] == info['cargo']:
                matched = index
                break
            if set(prior_edge) & set(edge) and set(prior['cargo']) & set(info['cargo']):
                break
        if matched is None:
            corrected.append((edge,info))
        else:
            corrected.pop(matched)
    edges=OrderedDict()
    for edge,info in corrected:
        combined=edges.setdefault(edge,{'cargo':Counter(),'line':info['line']})
        combined['cargo'].update(info['cargo'])
    if not edges:
        if name == 'the shadow lengthens':
            swap = _identical_swap(timeline, state)
            if swap:
                return swap
        if name in ('the shadow lengthens','the shadow is moving'):
            return ()
        raise Unproven('no_movement')
    if name == 'corsairs of umbar':
        landing = [(source, target, edge) for (source, target), edge in edges.items() if source == 'umbar']
        if len(landing) != 1:
            raise Unproven('corsairs_no_unique_departure')
        source, target, edge = landing[0]
        cargo = edge['cargo']
        available = army_cargo(state, source)
        if not any(k[0] == 'unit' for k in cargo) or any(n > available[k] for k,n in cargo.items()):
            raise Unproven('corsairs_cargo_unproven')
        # Post-battle survivor movement cannot establish the initial cargo.
        first_commit = min((int(r.get('line', 0)) for r in batch['records']
            if r.get('op') == 'move' and 'hand' in str(r.get('from','')).lower()
            and int(r.get('line',0)) > play_line), default=10**9)
        if edge['line'] > first_commit:
            raise Unproven('corsairs_postbattle_cargo_unproven')
        return ({'from':source,'to':target,'cargo':cargo,'line':edge['line'],
                 'whole':cargo == available,'path':(source,target)},)
    join = name != 'the shadow is moving'
    limit = 1 if name == 'shadows gather' else 2 if join else 4
    hops = 3 if name == 'shadows gather' else 2

    def _walk(movements):
        # Uses only fresh locals, so a second walk starts from a clean slate.
        routes=[];used_sources=set();reserved=Counter()
        for (source,target),edge in movements:
            cargo=edge['cargo']
            if not any(key[0]=='unit' and count>0 for key,count in cargo.items()):
                raise Unproven('leader_route_without_army')
            if source==target:raise Unproven('return_correction')
            available=army_cargo(state,source)
            if join:
                # Reserve only original cargo. Arriving figures cannot become a
                # new source army merely because the client drags them again.
                can_supply=all(count<=available[key]-reserved[(source,key)] for key,count in cargo.items())
            else:
                can_supply=source not in used_sources and all(count<=available[key] for key,count in cargo.items())
            incoming=[route for route in routes if route['path'][-1]==source and route['cargo']==cargo and len(route['path'])<hops + 1]
            if name=='the shadow is moving':
                # One-region movements cannot legally forward an arriving group.
                if not can_supply:raise Unproven('tsim_original_cargo_unproven')
                routes.append({'path':[source,target],'cargo':cargo,'line':edge['line']});used_sources.add(source)
            elif can_supply and incoming:
                # Once two distinct original armies are committed, selecting a
                # third original source is illegal. A single matching incoming
                # route can therefore prove the second hop without unit IDs.
                if source not in used_sources and len(used_sources)==limit and len(incoming)==1 and target!=incoming[0]['path'][0]:
                    incoming[0]['path'].append(target)
                else:
                    raise Unproven('tsl_origin_or_through_ambiguous')
            elif can_supply:
                routes.append({'path':[source,target],'cargo':cargo,'line':edge['line']});used_sources.add(source)
                for key,count in cargo.items():reserved[(source,key)]+=count
            elif len(incoming)==1 and target!=incoming[0]['path'][0]:
                incoming[0]['path'].append(target)
            else:
                raise Unproven('tsl_cargo_route_unproven')
        return routes
    try:
        # Merged edges stay primary: they explain one army dragged in pieces.
        routes=_walk(edges.items())
    except Unproven as error:
        # R2 (2026-10-22): two armies converging on one edge (East Rhun and
        # North Rhun both Vale of the Carnen -> Dale) merge into one edge
        # carrying both cargoes before the second army's first hop exists,
        # so no source or incoming route explains it. For that refusal only,
        # on a join card, re-walk the drag-back-corrected timeline in literal
        # order (the Sep-13 chronological TSL walk). Every other refusal and
        # every shape the merged walk plans is unchanged.
        if name == 'the shadow lengthens' and str(error) == 'tsl_origin_or_through_ambiguous':
            # R4-EXTRACT-COVERAGE-DROPS: a forced swap dragged in pieces
            swap = _piecewise_swap(timeline, state)
            if swap:
                return swap
        if not join or str(error)!='tsl_cargo_route_unproven':raise
        try:
            routes=_walk(corrected)
        except Unproven:
            # The timeline walk is a recovery attempt only: a window it cannot
            # plan either keeps the merged walk's refusal reason.
            raise error from None
    if join:
        # The client may drag a selected army in pieces. Merge only original
        # source fragments proven to finish at the same destination. Keep
        # every literal route for audit; distinct original armies stay apart.
        combined=OrderedDict()
        for route in routes:
            source,target=route['path'][0],route['path'][-1]
            prior=combined.get(source)
            if prior is None:
                combined[source]={'path':route['path'],'paths':[tuple(route['path'])],
                                  'cargo':Counter(route['cargo']),'line':route['line']}
            else:
                if prior['path'][-1]!=target:
                    raise Unproven('tsl_original_source_multiple_destinations')
                prior['cargo'].update(route['cargo'])
                prior['paths'].append(tuple(route['path']))
        routes=list(combined.values())
    if len(routes)>limit:raise Unproven('too_many_armies')
    plans=[]
    for route in routes:
        source,target=route['path'][0],route['path'][-1];cargo=route['cargo']
        regular=sum(count for key,count in cargo.items() if key[0]=='unit' and key[2]=='REGULAR')
        elite=sum(count for key,count in cargo.items() if key[0]=='unit' and key[2]=='ELITE')
        whole=cargo==army_cargo(state,source)
        plans.append({'from':source,'to':target,'regular':regular,'elite':elite,'whole':whole,'cargo':cargo,'path':tuple(route['path']),'paths':tuple(route.get('paths',[tuple(route['path'])])),'line':route['line']})
    return tuple(plans)


def match_plan(copts,ctx):
    """Match the next proven original-army movement to exactly one public option."""
    plan=ctx['_staged_army_plan']
    cursor=int(ctx.get('_staged_army_i',0))
    if cursor>=len(plan):
        matches=[i for i,a in copts if any(a.params.get('choices',{}).get(key) for key in ('_tsim_done','_tsl_done'))]
        return (matches[0],None) if len(matches)==1 else (None,'cardint_staged:no_unique_done')
    expected=plan[cursor]
    matches=[]
    for index,action in copts:
        choice=action.params.get('choices',{})
        if choice.get('from_region', 'umbar' if 'target_region' in choice else '')!=expected['from'] or choice.get('to_region',choice.get('target_region'))!=expected['to']:continue
        if '_cargo_index' not in choice:
            matches.append(index)
    if len(matches)!=1:return None,'cardint_staged:no_unique_public_movement'
    ctx['_staged_cargo_i']=cursor
    ctx['_staged_army_i']=cursor+1
    ctx['tm_i']=cursor+1
    return matches[0],None


def match_cargo(copts, ctx):
    plan = ctx.get('_staged_army_plan')
    cursor = ctx.get('_staged_cargo_i')
    if plan is None or cursor is None or not 0 <= cursor < len(plan):
        return None, 'cardint_cargo:no_proven_route'
    expected = plan[cursor]['cargo']
    matches = []
    for index, action in copts:
        choice = action.params.get('choices', {})
        category = tuple(choice.get('_cargo_key', ()))
        if len(category) != 3:
            continue
        literal_key = (category[0], category[1], category[2] or None)
        count = expected[literal_key]
        if choice.get('count') == count:
            matches.append(index)
    return (matches[0], None) if len(matches) == 1 else (None, 'cardint_cargo:no_unique_count')
