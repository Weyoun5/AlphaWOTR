"""Small immutable source-computed probabilities; no native reimplementation of pow."""
def values():
    import hunt_dist as hd
    primary=[];reroll=[]
    for hits in range(6):
        p=hits/6.0
        primary.append([[([6]*h+[1]*(n-h),prob) for h,prob in hd._binom_terms(n,p)] for n in range(6)])
        reroll.append([([6],p),([1],1.0-p)])
    specials={tile.special_name:hd.tile_key(tile) for tile in hd.SPECIAL_HUNT_TILES.values() if tile.special_name}
    return dict(schema='source_hunt_options_v1',primary=primary,reroll=reroll,
                shelob=[([v],1.0/6.0) for v in range(1,7)],specials=specials)
