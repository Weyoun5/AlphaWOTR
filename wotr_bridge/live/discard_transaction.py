"""Replayable physical hand-limit discard, with no opponent choice or RNG."""
from dataclasses import dataclass


@dataclass(frozen=True)
class HandDiscardTransaction:
    side: object
    card_ids: tuple
    action_type = 'hand_limit_discard_group'

    @property
    def params(self):
        return {'side':self.side.name,'card_ids':list(self.card_ids)}

    def __call__(self,state):
        import game_logic as gl
        import ai
        pending=ai.describe_pending(state)
        if pending is None or pending['kind']!='discard':
            raise ValueError('A hand-limit discard must be the current decision')
        hand=state.hands[self.side].hand
        ids=tuple(self.card_ids)
        if (not ids or len(set(ids))!=len(ids) or len(ids)>len(hand)-gl.HAND_LIMIT
                or any(cid not in hand or str(cid).startswith('HU_') for cid in ids)):
            raise ValueError('Discard identities or excess count are not proven')
        # No rule inspects face-down discard order. Canonical ordering keeps
        # replay identical for one native group whose members are unordered.
        for cid in sorted(ids):
            if not gl.discard_hand_card(state,self.side,cid):
                raise ValueError('Selected hand card disappeared during discard')

