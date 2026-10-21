"""Resolve a declared reveal route and its separate stronghold Hunt draws."""
import heapq


def reachable_regions(state):
    """Map destinations within progress along passable region borders.

    Gorgoroth and its neighboring regions are ordinary map destinations;
    the Mordor Track is separate from those regions (rulebook pp. 38, 43).
    """
    import game_logic as gl
    from board import sanitize
    steps = state.fellowship.progress
    if type(steps) is not int or steps < 0:
        raise ValueError('Invalid Fellowship progress')
    if state.fellowship.mordor_entered:
        raise ValueError('Mordor Track has no map reveal route')
    start = gl.fellowship_region(state)
    reached = frontier = {start}
    for _ in range(steps):
        frontier = {sanitize(nxt) for current in frontier
                    for nxt in state.board.adjacent(current)} - reached
        reached = reached | frontier
        if not frontier:
            break
    return reached


def _stronghold(state, region):
    import board
    import game_logic as gl
    from game_state import Side
    return (board.regions[region].get('is_stronghold') and gl.is_shadow_region(region)
            and region not in state.enemy_controlled[Side.FREE_PEOPLES])


def choose_path(state, destination, path=None, *, declaration=False):
    """Default to fewest extra draws within progress; accept an explicit route."""
    import game_logic as gl
    from board import sanitize
    start=gl.fellowship_region(state)
    destination=sanitize(destination)
    steps=state.fellowship.progress
    if type(steps) is not int or steps < 0:
        raise ValueError('Invalid Fellowship progress')
    if state.fellowship.mordor_entered:
        raise ValueError('Mordor Track has no map reveal route')
    if path is not None:
        if not isinstance(path,(list,tuple)) or not path:
            raise ValueError('Reveal route must be a nonempty list of regions')
        route=tuple(sanitize(str(r)) for r in path)
        if route[0]!=start or route[-1]!=destination or len(route)-1>steps:
            raise ValueError('Reveal route endpoints or length differ from placement')
        if any(b not in state.board.adjacent(a) for a,b in zip(route,route[1:])):
            raise ValueError('Reveal route contains a nonadjacent step')
        return route
    # Positive path length breaks ties, so no optimal route contains a loop.
    # Different step counts must stay separate: a cheaper detour might leave
    # too little progress to reach the requested destination.
    seen=getattr(state,'_sh_tile_drawn_regions',None) or set()
    frontier=[(0,0,(start,))]
    best={(start,0):0}
    while frontier:
        cost,used,route=heapq.heappop(frontier)
        current=route[-1]
        if cost!=best.get((current,used)):
            continue
        if current==destination:
            return route
        if used==steps:
            continue
        for nxt in sorted(state.board.adjacent(current)):
            added=int(bool(not declaration and _stronghold(state,nxt) and nxt not in seen))
            if nxt == 'moria' and 'moria' not in route and balrog_card(state) is not None:
                added += 1
            key=(nxt,used+1)
            new=cost+added
            if new < best.get(key,float('inf')):
                best[key]=new
                heapq.heappush(frontier,(new,used+1,route+(nxt,)))
    raise ValueError('Reveal destination is not reachable within progress')


def place(state, destination, path=None):
    import game_logic as gl
    from game_state import Side
    import board
    route=choose_path(state,destination,path)
    destination=route[-1]
    meta=board.regions[destination]
    if ((meta.get('is_city') or meta.get('is_stronghold'))
            and gl.is_fp_region(destination)
            and destination not in state.enemy_controlled[Side.SHADOW]):
        raise ValueError('Reveal cannot end in an unconquered Free Peoples city or stronghold')
    if (getattr(state,'_pending_reveal_strongholds',None)
            or getattr(state,'_fellowship_route_event',None)):
        raise ValueError('Previous reveal route has unfinished Hunt draws')
    seen=getattr(state,'_sh_tile_drawn_regions',None)
    if seen is None:
        state._sh_tile_drawn_regions=seen=set()
    owed=tuple(dict.fromkeys(r for r in route if _stronghold(state,r) and r not in seen))
    state.fellowship.region=state.board.region(destination)
    state.fellowship.progress=0
    state._reveal_place_due=False
    state._pending_reveal_strongholds=owed
    state._fellowship_route_event={
        'kind':'reveal', 'balrog_due':'moria' in route and balrog_card(state) is not None,
        'conditional_strongholds':()}
    from game_trace import emit_annotation
    emit_annotation(state,'fellowship_reveal_route',{'path':list(route),'strongholds':list(owed)})
    drain(state)


def balrog_card(state):
    from game_state import PersistentCardType
    cards=getattr(state,'persistent_cards',None)
    return cards.get_by_type(PersistentCardType.BALROG_OF_MORIA) if cards else None


def declared(state, route):
    """Register the selected route after declaration's healing and upkeep."""
    if 'moria' not in route or balrog_card(state) is None:
        return
    if (getattr(state,'_fellowship_route_event',None)
            or getattr(state,'_pending_reveal_strongholds',None)):
        raise ValueError('Previous Fellowship route is unfinished')
    # These tiles become owed only if the completed declaration is revealed.
    # A stationary Fellowship already revealed at registration creates no
    # new reveal event merely because a Balrog tile is drawn.
    conditional=tuple(dict.fromkeys(r for r in route if _stronghold(state,r))) if state.fellowship.hidden else ()
    state._fellowship_route_event={
        'kind':'declare','balrog_due':True,'conditional_strongholds':conditional}
    from game_trace import emit_annotation
    emit_annotation(state,'fellowship_declaration_route',{'path':list(route)})
    drain(state)


def drain(state):
    """Almanac p27: placement, optional Balrog, then each stronghold draw."""
    import game_logic as gl
    while (getattr(state,'_fellowship_route_event',None)
            or getattr(state,'_pending_reveal_strongholds',None)):
        gl.settle_hunt_reveal(state)
        if gl.should_game_end(gl.check_all_victory_conditions(state), end_of_turn=False):
            state._pending_reveal_strongholds=()
            state._fellowship_route_event=None
            state._reveal_place_due=False
            return
        if (gl._reveal_hunt_pending(state)
                or getattr(state, '_hunt_reveal_pending', 0)
                or any(getattr(state,key,None) for key in (
                    '_hobbit_place_queue','pending_card_followup','pending_discard'))):
            return
        event=getattr(state,'_fellowship_route_event',None)
        if event and event['balrog_due']:
            event['balrog_due']=False
            card=balrog_card(state)
            if card is not None:
                state.pending_card_option={
                    'card_id':getattr(card,'card_id','__balrog__'),
                    'card_name':'Balrog of Moria','side':'sh','decider':'sh','direct':True,
                    'options':[{'balrog_declare_draw':True},{'balrog_declare_draw':False}]}
                return
        if event and event['kind']=='declare' and not state.fellowship.hidden:
            # The figure was placed by the declaration. A revealing Balrog
            # tile activates draws on THAT route, without a second placement.
            state._reveal_place_due=False
            owed=event['conditional_strongholds']
            event['conditional_strongholds']=()
            seen=getattr(state,'_sh_tile_drawn_regions',None) or set()
            state._sh_tile_drawn_regions=seen
            state._pending_reveal_strongholds=tuple(getattr(state,'_pending_reveal_strongholds',None) or ())+tuple(r for r in owed if r not in seen)
        if not getattr(state,'_pending_reveal_strongholds',None):
            state._fellowship_route_event=None
            return
        region,*rest=state._pending_reveal_strongholds
        state._pending_reveal_strongholds=tuple(rest)
        gl._draw_shadow_stronghold_tile(state,region,'reveal route')
        if (getattr(state,'_pending_hunt_damage',0)>0
                and getattr(state,'pending_card_option',None) is None and state.pending_hunt is None):
            if not gl._offer_hunt_reduction(state):
                gl._settle_hunt_damage(state)


def pending_draws(state):
    from hunt_protection_context import STRONGHOLDS
    queue=getattr(state,'_pending_reveal_strongholds',None)
    if queue is None:
        return ()
    if (not isinstance(queue,(tuple,list)) or len(queue)>len(STRONGHOLDS)
            or any(r not in STRONGHOLDS for r in queue)
            or len(set(queue))!=len(queue)):
        raise ValueError('Malformed pending reveal stronghold draws')
    return tuple(queue)


def observation_tail(state):
    import numpy as np
    from hunt_protection_context import STRONGHOLDS
    tail=np.zeros(64,dtype=np.float32)
    for position,region in enumerate(pending_draws(state)):
        tail[position*8+STRONGHOLDS.index(region)]=1
    return tail
