"""Blade of Westernesse accepts a separated Hobbit under either name form.

Literal oracle: printed CSV prerequisite 'a Hobbit is in the battle'. The
full-name aliases identify the same Hobbit figures; no opponent hand or engine
output supplies the expected eligibility.
"""
import unittest
import ai
import combat
import cards
import game_state as gs
from test_combat_continuation_literal import live_battle, card_id, companion, minion


class BladeHobbitEligibility(unittest.TestCase):
    def test_exact_identity_aliases_and_duck_typed_figures(self):
        from types import SimpleNamespace
        for name in ('Meriadoc','Meriadoc Brandybuck','Peregrin','Peregrin Took',
                     'Frodo','Frodo Baggins','Sam','Samwise','Sam Gamgee','Samwise Gamgee'):
            with self.subTest(name=name):
                self.assertTrue(cards.is_hobbit_identity(name))
                self.assertTrue(cards.is_hobbit_identity(SimpleNamespace(name=name)))
        for name in (None,12,'Legolas','Not Meriadoc','Peregrin Took impostor','Samwise captain'):
            self.assertFalse(cards.is_hobbit_identity(name))
        self.assertFalse(cards.is_hobbit_identity(SimpleNamespace()))

    def position(self,name,*,on_map=True):
        state,a,d=live_battle(fp_leaders=[companion(name)] if on_map else [],
                             sh_leaders=[minion('Mouth of Sauron')])
        cid=card_id('Blade of Westernesse')
        state.hands[gs.Side.FREE_PEOPLES].hand=[cid]
        state.pending_combat={'attacker_region':a,'defender_region':d,
            'attacker_side':gs.Side.FREE_PEOPLES,'defender_side':gs.Side.SHADOW,
            'phase':'attacker_combat_card','deciding_side':gs.Side.FREE_PEOPLES,
            'rounds_fought':0,'is_siege':False,'shine_asked':True}
        return state,a,d,cid

    def test_full_name_hobbits_are_offered_in_actual_combat_card_menu(self):
        for name in ('Meriadoc Brandybuck','Peregrin Took'):
            with self.subTest(name=name):
                state,a,d,cid=self.position(name)
                plays=[action.params.get('card_id') for action in ai.get_legal_actions(state,gs.Side.FREE_PEOPLES)
                       if action.action_type=='combat_card_play']
                self.assertIn(cid,plays)

    def test_printed_requirement_accepts_full_names_at_base_and_extra_gates(self):
        for name in ('Meriadoc Brandybuck','Peregrin Took'):
            with self.subTest(name=name):
                state,a,d,cid=self.position(name)
                card=cards.get_card(cid)
                self.assertTrue(card.requires_hobbit)
                battle=combat.setup_combat(state,a,d)
                allowed,reason=cards.can_play_combat_card(card,battle,True)
                self.assertEqual((allowed,ai._extra_combat_card_ok(state,card,battle)),
                                 (True,True),reason)

    def test_short_name_control_and_nonhobbit_negative(self):
        for name,expected in (('Meriadoc',True),('Peregrin',True),('Legolas',False)):
            with self.subTest(name=name):
                state,a,d,cid=self.position(name)
                plays=[action.params.get('card_id') for action in ai.get_legal_actions(state,gs.Side.FREE_PEOPLES)
                       if action.action_type=='combat_card_play']
                self.assertEqual(cid in plays,expected)

    def test_hobbit_in_fellowship_does_not_meet_map_participation_requirement(self):
        state,a,d,cid=self.position('Meriadoc Brandybuck',on_map=False)
        # Fellowship carries Hobbits, but none participates in either army.
        self.assertTrue(any(c.name in ('Meriadoc','Peregrin') for c in state.fellowship.companions))
        self.assertNotIn(cid,[action.params.get('card_id') for action in ai.get_legal_actions(state,gs.Side.FREE_PEOPLES)
                             if action.action_type=='combat_card_play'])


if __name__=='__main__':unittest.main()
