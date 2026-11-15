"""Reference opponent reply allocation; utilities always belong to the search root.

Statistics here must come from opponent-information-conditioned belief evaluations,
not from the main tree's evaluations of the root's actual private hand. This small
component does not change production MCTS or call any network itself.
"""
from dataclasses import dataclass
import math
import random


def expected_utility(values, probabilities):
    """Average random/hidden outcomes before comparing the opponent's actions."""
    values=tuple(float(v) for v in values)
    probabilities=tuple(float(p) for p in probabilities)
    if not values or len(values)!=len(probabilities):
        raise ValueError('One probability is required for each utility')
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Reply utilities must be finite')
    if not all(math.isfinite(p) and p>=0 for p in probabilities):
        raise ValueError('Outcome probabilities must be finite and nonnegative')
    total=math.fsum(probabilities)
    if not math.isfinite(total) or total<=0:
        raise ValueError('Outcome probability mass must be positive and finite')
    return math.fsum((p/total)*v for v,p in zip(values,probabilities))


@dataclass
class ReplyStats:
    panels: int=0
    utility_sum: float=0.0

    @property
    def mean(self):
        return self.utility_sum/self.panels if self.panels else None


class ReplySelector:
    """Progressively examine legal replies using lower-confidence root utilities.

    Shuffling a complete legal menu avoids a permanent first-N/top-policy cutoff.
    Wider budgets gradually admit every action. Selection and completed values
    are separate so a failed evaluation cannot create a spurious good/bad reply.
    One panel is the weighted mean across a common set of belief/chance samples.
    """
    def __init__(self, action_keys, *, seed, exploration=1.25, widening=2.0):
        self.keys=tuple(action_keys)
        if not self.keys or len(set(self.keys))!=len(self.keys):
            raise ValueError('A nonempty unique legal action menu is required')
        if not math.isfinite(exploration) or exploration<=0:
            raise ValueError('Exploration must be positive and finite')
        if not math.isfinite(widening) or widening<=0:
            raise ValueError('Widening must be positive and finite')
        self.exploration=exploration;self.widening=widening
        self.order=list(range(len(self.keys)));random.Random(seed).shuffle(self.order)
        self.stats=[ReplyStats() for _ in self.keys]
        self.completed=0;self.pending=None

    def select(self):
        if self.pending is not None:
            raise RuntimeError('Complete or cancel the pending reply before selecting again')
        width=min(len(self.keys),max(1,math.ceil(self.widening*math.sqrt(self.completed+1))))
        active=self.order[:width]
        unvisited=[i for i in active if self.stats[i].panels==0]
        if unvisited:
            selected=unvisited[0]
        else:
            def score(i):
                arm=self.stats[i]
                return arm.mean-self.exploration*math.sqrt(math.log(self.completed+1)/arm.panels)
            selected=min(active,key=score)
        self.pending=selected
        return self.keys[selected]

    def complete(self, key, utilities, probabilities):
        if self.pending is None or key!=self.keys[self.pending]:
            raise ValueError('Completion does not identify the pending reply')
        value=expected_utility(utilities,probabilities)
        arm=self.stats[self.pending]
        updated=arm.utility_sum+value
        if not math.isfinite(updated):
            raise ValueError('Accumulated reply utility is not finite')
        arm.utility_sum=updated;arm.panels+=1;self.completed+=1;self.pending=None
        return value

    def cancel(self):
        self.pending=None

    def best(self):
        candidates=[i for i,s in enumerate(self.stats) if s.panels]
        if not candidates:
            raise RuntimeError('No completed reply evaluations')
        return self.keys[min(candidates,key=lambda i:self.stats[i].mean)]

    def report(self):
        return [{'key':key,'panels':s.panels,'root_utility':s.mean}
                for key,s in zip(self.keys,self.stats)]


class SharedReplySelector:
    """SEARCHFIX FIX R (ai._REPLY_SHARED): one reply node's statistics across all sampled hands.

    Arms are keyed by the action's semantic key (its RGK1 wire bytes), so a card row
    accumulates over every sampled hand that can play it. An arm's position is a
    64-bit priority drawn once, at first sight, from the table's own seeded stream:
    it never changes, so a key first seen in a later hand takes a stable place in one
    fixed random order and the hands' menus are restrictions of that order. Width and
    the lower-confidence rule are ReplySelector's, on this node's completed count.
    Values remain opponent-information-conditioned belief panels; never root-hand ones.
    Native twin: OwnSharedReplies (own_reply_types.inc).
    """
    def __init__(self, *, seed, exploration=1.25, widening=2.0):
        if not math.isfinite(exploration) or exploration<=0:
            raise ValueError('Exploration must be positive and finite')
        if not math.isfinite(widening) or widening<=0:
            raise ValueError('Widening must be positive and finite')
        self.exploration=exploration;self.widening=widening
        self.rng=random.Random(seed)
        self.arms={};self.priority=[];self.stats=[]
        # len(actions_bytes(all arms)): the node's cache charge (portable_reply_identity.logical_charge)
        self.wire_bytes=13
        self.completed=0;self.pending=None

    def admit(self, wires):
        for wire in wires:
            if wire not in self.arms:
                self.arms[wire]=len(self.stats)
                self.priority.append(self.rng.getrandbits(64));self.stats.append(ReplyStats())
                self.wire_bytes+=8+len(wire)


class HandReplies:
    """One visit's hand menu over a node's SharedReplySelector: ReplySelector's select /
    complete / cancel / best, restricted to the arms this hand can play."""
    def __init__(self, table, keys, wires):
        self.table=table;self.keys=tuple(keys)
        if not self.keys or len(self.keys)!=len(wires) or len(set(wires))!=len(wires):
            raise ValueError('A nonempty unique legal action menu is required')
        table.admit(wires)
        self.rows=[table.arms[wire] for wire in wires]
        self.order=sorted(range(len(self.rows)),key=lambda r:(table.priority[self.rows[r]],self.rows[r]))
        self.selected=None

    def select(self):
        table=self.table
        if table.pending is not None:
            raise RuntimeError('Complete or cancel the pending reply before selecting again')
        width=min(len(self.keys),max(1,math.ceil(table.widening*math.sqrt(table.completed+1))))
        active=self.order[:width]
        unvisited=[r for r in active if table.stats[self.rows[r]].panels==0]
        if unvisited:
            selected=unvisited[0]
        else:
            def score(r):
                arm=table.stats[self.rows[r]]
                return arm.mean-table.exploration*math.sqrt(math.log(table.completed+1)/arm.panels)
            selected=min(active,key=score)
        table.pending=self.rows[selected];self.selected=selected
        return self.keys[selected]

    def complete(self, key, utilities, probabilities):
        table=self.table
        if table.pending is None or self.selected is None or key!=self.keys[self.selected]:
            raise ValueError('Completion does not identify the pending reply')
        value=expected_utility(utilities,probabilities)
        arm=table.stats[table.pending]
        updated=arm.utility_sum+value
        if not math.isfinite(updated):
            raise ValueError('Accumulated reply utility is not finite')
        arm.utility_sum=updated;arm.panels+=1;table.completed+=1;table.pending=None
        return value

    def cancel(self):
        self.table.pending=None

    def best(self):
        """The lowest mean among the arms legal in this hand with panels>0, first in menu order on a tie."""
        found=None;value=None
        for row,arm_index in enumerate(self.rows):
            arm=self.table.stats[arm_index]
            if arm.panels and (found is None or arm.mean<value):
                found=row;value=arm.mean
        if found is None:
            raise RuntimeError('No completed reply evaluations')
        return self.keys[found]
