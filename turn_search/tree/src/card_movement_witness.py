"""Gross public cargo witness for staged army movement rendering."""
from collections import Counter

def row(source,target,units,leaders):
    unit_counts=Counter((unit.nation.name,unit.kind.name) for unit in units)
    leader_counts=Counter((type(leader).__name__,getattr(getattr(leader,'nation',None),'name',''),str(getattr(leader,'name',''))) for leader in leaders)
    return (source,target,tuple((*key,count) for key,count in sorted(unit_counts.items())),tuple((*key,count) for key,count in sorted(leader_counts.items())))
