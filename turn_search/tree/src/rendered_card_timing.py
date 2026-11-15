"""Scoped public-bag timing for card menus that precede their Hunt effect."""
from contextlib import contextmanager
import re
import unicodedata


def pool_signature(state):
    return tuple(tuple(sorted(str(t) for t in getattr(state, name, ())))
                 for name in ('hunt_pool', 'hunt_discard'))


def _norm(value):
    return re.sub('[^a-z]', '', unicodedata.normalize('NFKD', str(value)).encode('ascii','ignore').decode().lower())


@contextmanager
def pre_hunt_menu(state, batch, source_card_id, initial_pool):
    """Mark only an unchanged, known pre-draw card menu, then restore timing.

    Strike/Search flights precede their deferred Hunt/reveal consequences.
    Strike's hunt/discard/none menu also precedes its first roll. A consumed
    tile, post-draw marker, damage, or other pending context cannot qualify.
    """
    old = getattr(state, '_rx_hunt_observation', None)
    pending = getattr(state, 'pending_card_option', None) or {}
    options = pending.get('options') or []
    name = str(pending.get('card_name', ''))
    chain = (name == '__nazgul_moves__' and options
             and all(o.get('_naz_post') in ('strike','search') for o in options))
    final = (_norm(name) in ('nazgulstrike','thenazgulstrike') and options
             and all(o.get('action') in ('hunt','discard_table_card','none') for o in options))
    ents = (_norm(name) in ('theentsawakeentmoot','theentsawaketreebeard','theentsawakehuorns')
            and options and (pending.get('card_casualty')
                or all('ents_chain_play' in o or o.get('_ents_chain_done') for o in options)))
    safe = (isinstance(old, tuple) and len(old)==2
            and old[0]==int(batch['start_line']) and old[1] in ('snapshot','stable_batch')
            and pending.get('card_id')==source_card_id and (chain or final or ents)
            and pool_signature(state)==initial_pool
            and not any(getattr(state, key, None) for key in
                        ('pending_hunt','_pending_hunt_damage','_mithril_stash','pending_combat')))
    if safe:
        state._rx_hunt_observation = (int(batch['start_line']), 'before_draw')
    try:
        yield bool(safe)
    finally:
        if safe:
            state._rx_hunt_observation = old


def is_fp_challenge_draw(batch):
    """Challenge's explicit FP tile draw cannot offer Mithril (Almanac p11)."""
    if batch.get('die_side') != 'FREE_PEOPLES':
        return False
    draws = sorted((r for r in batch.get('rng', ()) if r.get('kind') == 'draw_tile'),
                   key=lambda r: int(r.get('line') or 0))
    stages = [r for r in batch.get('records', ()) if r.get('die_stage')]
    if not draws or not stages:
        return False
    actor = stages[0].get('player')
    if not actor or draws[0].get('player') != actor:
        return False
    return any(r.get('op') == 'move' and int(r.get('line') or 0) < int(draws[0]['line'])
               and _norm(str(r.get('obj', '')).split('/')[0]) == 'challengeoftheking'
               and str(r.get('from', '')).strip().lower() == 'free peoples hand'
               and _norm(r.get('to', '')) in ('currentfpcard','currentsacard')
               for r in batch.get('records', ()))
