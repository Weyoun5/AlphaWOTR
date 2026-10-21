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
