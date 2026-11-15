"""Public draw source, known Hunt successes and already-resolved strongholds."""
import numpy as np

SOURCES=('hunt','mordor','balrog','orc_patrol','foul_thing','isildur','stronghold')
STRONGHOLDS=('barad dur','dol guldur','minas morgul','morannon','moria',
             'mount gundabad','orthanc','umbar')
TAIL_WIDTH=25

def _source(stash):
    kind=stash.get('apply_kind')
    if kind is not None:
        if kind not in SOURCES[3:] + ('breaking',):
            raise ValueError('Unknown Hunt protection draw source')
        return kind
    if stash.get('mordor_track'):
        return 'mordor'
    return 'balrog' if stash.get('eye_no_effect') else 'hunt'

def _write(tail,stash,start):
    if stash is None:
        return
    if not isinstance(stash,dict) or not stash:
        raise ValueError('Malformed Hunt protection continuation')
    kind=_source(stash)
    count=stash.get('successes',0)
    if type(count) is not int or not 0<=count<=5:
        raise ValueError('Hunt successes must be an integer from zero to five')
    # Breaking uses separate v51 flags; keep all seven legacy slots fixed.
    if kind in SOURCES:
        tail[start+SOURCES.index(kind)]=1
    tail[start+7]=count/5.

def observation_tail(state):
    tail=np.zeros(TAIL_WIDTH,dtype=np.float32)
    mithril=getattr(state,'_mithril_stash',None)
    _write(tail,mithril,0)
    if mithril is not None:
        returning=mithril.get('return_pending',False)
        if type(returning) is not bool:
            raise ValueError('Malformed Mithril return status')
        tail[8]=float(returning)
    _write(tail,getattr(state,'_wizardstaff_stash',None),9)
    if getattr(state,'_reveal_place_due',False):
        seen=getattr(state,'_sh_tile_drawn_regions',None) or set()
        if not isinstance(seen,(set,frozenset,list,tuple)):
            raise ValueError('Malformed reveal-event stronghold history')
        for region in seen:
            if region not in STRONGHOLDS:
                raise ValueError('Unknown stronghold in reveal-event history')
            tail[17+STRONGHOLDS.index(region)]=1
    return tail


def breaking_tail(state):
    """Breaking source flags and remaining Companion removals, normalized by eight."""
    observation_tail(state)
    return np.asarray([
        float(bool(getattr(state, name, None)) and
              _source(getattr(state, name)) == 'breaking')
        for name in ('_mithril_stash', '_wizardstaff_stash')] +
        [breaking_remaining(state) / 8.0], dtype=np.float32)


def breaking_remaining(state):
    """Count of required remaining choices; zero means no Breaking chain."""
    pending = getattr(state, 'pending_card_option', None)
    if pending is None:
        return 0
    if not isinstance(pending, dict):
        raise ValueError('Malformed pending card context')
    if not pending.get('bof_separation'):
        if 'bof_n_left' in pending:
            raise ValueError('Breaking count without a separation context')
        return 0
    count = pending.get('bof_n_left')
    if pending.get('bof_separation') is not True or type(count) is not int or not 1 <= count <= 8:
        raise ValueError('Breaking remaining choices must be an integer from one to eight')
    companions = getattr(state.fellowship, 'companions', ())
    movable = sum(c.name not in ('Frodo', 'Sam', 'Gollum') for c in companions)
    if count > movable:
        raise ValueError('Breaking remaining choices exceed eligible Companions')
    return count
