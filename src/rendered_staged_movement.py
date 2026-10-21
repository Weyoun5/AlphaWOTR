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
    edges=OrderedDict();timeline=[];last_move=play_line
    for record in batch['records']:
        if record.get('op')!='move' or not play_line < int(record.get('line',0)) < boundary:continue
        origin,target=rr.zone_kind(record.get('from','')),rr.zone_kind(record.get('to',''))
        if origin[0] not in ('region','stronghold') or target[0] not in ('region','stronghold'):continue
        if origin[1]==target[1]:continue
        cargo=Counter()
        for token in rx._group_tokens(record.get('obj','')):
            key,count=token_cargo(token)
            if key is not None and (key[0] == 'nazgul'
                or key[0] == 'named' and key[1] in ('Witch King','Mouth of Sauron')
                or key[0] in ('unit','leader') and key[1] in ('SAURON','ISENGARD','SOUTHRON_EASTERLINGS')):
                cargo[key]+=count
        if not cargo:continue
        edge=edges.setdefault((origin[1],target[1]),{'cargo':Counter(),'line':int(record['line'])})
        edge['cargo'].update(cargo);last_move=max(last_move,int(record['line']))
        timeline.append(((origin[1],target[1]),{'cargo':cargo,'line':int(record['line'])}))
    if not any(line>last_move for line in closed):raise Unproven('movement_after_closure')
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

    routes=[];used_sources=set();reserved=Counter()
    movements=edges.items()
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
