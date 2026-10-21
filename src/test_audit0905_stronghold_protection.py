"""Almanac p.11: protection remains available on every Shadow tile draw."""
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
SOURCE=Path(os.environ.get('WOTR_TIMING_SOURCE', str(Path(__file__).resolve().parent))).resolve()
os.environ.update(WOTR_ACT_GEN='v47',WOTR_TIMING_GENERATION='v47',WOTR_NATIVE='0',CUDA_VISIBLE_DEVICES='-1')
sys.path.insert(0,str(SOURCE))
import test_audit0905_mithril_return as f
ai,gl,gs=f.ai,f.gl,f.gs
assert Path(ai.__file__).resolve().parent==SOURCE

def enter_moria(extra='3'):
    state=f.setup(original='1R',replacement=extra)
    state=f.redraw(f.move(state),yes=False)
    state=f.choose(state,'hunt_take_corruption')
    return f.place(state,'moria')

class StrongholdProtectionTests(unittest.TestCase):
    def test_gollum_reveal_on_mordor_track_does_not_draw_for_old_gateway(self):
        state=f.setup(original='1',replacement='3')
        gl.get_or_create_army(state,'dimrill dale').leaders.extend(state.fellowship.companions)
        state.fellowship.companions=[]
        state.fellowship.guide=state.gollum
        state.fellowship.region=state.board.region('morannon')
        state.fellowship.mordor_entered=True
        state.fellowship.mordor_step=2
        state.fellowship.corruption=5
        state=f.redraw(f.move(state),yes=False)
        final=f.choose(state,'hunt_gollum_reveal')
        self.assertEqual(final.fellowship.corruption,5)
        self.assertEqual(list(map(repr,final.hunt_pool)),['3'])
        self.assertEqual(len(final.hunt_discard),15)
        self.assertEqual(final.fellowship.mordor_step,3)
        self.assertFalse(final.fellowship.hidden)
        self.assertIsNone(final.pending_card_option)
        self.assertIsNone(final.pending_hunt)
        self.assertEqual(len(final.hunt_box),2)

    def test_kept_mithril_is_offered_again_for_moria_extra_tile(self):
        state=enter_moria()
        self.assertIsNone(state.pending_hunt)
        self.assertEqual(state.pending_card_option['card_name'],'Mithril Coat and Sting')
        self.assertEqual(repr(state._mithril_stash['tile']),'3')
        self.assertEqual(state.fellowship.corruption,1)
        self.assertEqual(f.inventory(state),f.ORDINARY)
        self.assertEqual(len(state.hunt_box),1)
        final=f.redraw(state)
        self.assertEqual(final.pending_hunt['damage'],1)
        self.assertFalse(final._reveal_place_due)
        final=f.choose(final,'hunt_take_corruption')
        self.assertEqual(final.fellowship.corruption,2)
        self.assertEqual(len(final.hunt_pool),15)
        self.assertEqual(len(final.hunt_discard),1)
        self.assertEqual(gl.fellowship_region(final),'moria')
        self.assertEqual(len(final.hunt_box),2)
        self.assertEqual(f.inventory(final),f.ORDINARY)

    def test_declining_extra_tile_keeps_card_and_charges_only_its_damage(self):
        state=f.redraw(enter_moria(),yes=False)
        self.assertEqual(state.pending_hunt['damage'],3)
        final=f.choose(state,'hunt_take_corruption')
        self.assertEqual(final.fellowship.corruption,4)
        self.assertEqual(final.hunt_pool,[])
        self.assertEqual(len(final.hunt_discard),16)
        self.assertTrue(final.persistent_cards.has_card_type(gs.PersistentCardType.MITHRIL_COAT))
        self.assertFalse(final._reveal_place_due)

    def test_extra_eye_and_zero_reveal_do_not_damage_or_start_a_second_reveal(self):
        for tile in ('Eye','0R'):
            with self.subTest(tile=tile):
                state=enter_moria(tile)
                self.assertIsNotNone(state.pending_card_option)
                final=f.redraw(state,yes=False)
                self.assertEqual(final.fellowship.corruption,1)
                self.assertIsNone(final.pending_hunt)
                self.assertFalse(final._reveal_place_due)
                self.assertEqual(final.hunt_pool,[])
                self.assertEqual(len(final.hunt_discard),16)
                self.assertEqual(len(final.hunt_box),2)

    def test_staff_can_prevent_extra_draw_after_declining_first_draw(self):
        state,cid=f.add_protection_through_public_play(f.setup(original='1R',replacement='3'),"Wizard's Staff")
        state=f.move(state)
        state=f.choose(state,'card_option',choices={'wizardstaff_prevent':False})
        state=f.redraw(state,yes=False)
        state=f.choose(state,'hunt_take_corruption')
        state=f.place(state,'moria')
        self.assertEqual(state.pending_card_option['card_name'],"Wizard's Staff")
        self.assertEqual(list(map(repr,state.hunt_pool)),['3'])
        final=f.choose(state,'card_option',choices={'wizardstaff_prevent':True})
        self.assertEqual(final.fellowship.corruption,1)
        self.assertEqual(list(map(repr,final.hunt_pool)),['3'])
        self.assertEqual(len(final.hunt_discard),15)
        self.assertEqual(len(final.hunt_box),2)
        self.assertFalse(final.persistent_cards.has_card_type(gs.PersistentCardType.WIZARD_STAFF))
        self.assertTrue(final.persistent_cards.has_card_type(gs.PersistentCardType.MITHRIL_COAT))

    def test_staff_decline_then_mithril_applies_to_same_extra_draw(self):
        state,cid=f.add_protection_through_public_play(f.setup(original='1R',replacement='3'),"Wizard's Staff")
        state=f.choose(f.move(state),'card_option',choices={'wizardstaff_prevent':False})
        state=f.choose(f.redraw(state,yes=False),'hunt_take_corruption')
        state=f.place(state,'moria')
        state=f.choose(state,'card_option',choices={'wizardstaff_prevent':False})
        self.assertEqual(state.pending_card_option['card_name'],'Mithril Coat and Sting')
        self.assertEqual(repr(state._mithril_stash['tile']),'3')
        self.assertEqual(state.hunt_pool,[])
        state=f.redraw(state,yes=False)
        final=f.choose(state,'hunt_take_corruption')
        self.assertEqual(final.fellowship.corruption,4)
        self.assertEqual(f.inventory(final),f.ORDINARY)

    def test_revealed_in_moria_placement_precedes_extra_damage(self):
        state=f.setup(original='1R',replacement='3')
        state.fellowship.region=state.board.region('moria')
        state=f.redraw(f.move(state),yes=False)
        state=f.choose(state,'hunt_take_corruption')
        self.assertIsNone(state.pending_card_option)
        self.assertTrue(state._reveal_place_due)
        self.assertEqual(state.fellowship.progress,1)
        self.assertEqual(list(map(repr,state.hunt_pool)),['3'])
        state=f.place(state,'dimrill dale')
        self.assertEqual(state.pending_card_option['card_name'],'Mithril Coat and Sting')
        self.assertFalse(state._reveal_place_due)
        self.assertEqual(gl.fellowship_region(state),'dimrill dale')
        state=f.redraw(state,yes=False)
        final=f.choose(state,'hunt_take_corruption')
        self.assertEqual(final.fellowship.corruption,4)
        self.assertEqual(len(final.hunt_discard),16)
        self.assertEqual(final.hunt_pool,[])
        self.assertEqual(len(final.hunt_box),2)

    def test_zero_progress_forced_placement_does_not_repeat_the_extra_draw(self):
        state=f.setup(original='1R',replacement='3')
        state.fellowship.region=state.board.region('moria')
        from cards import get_cards
        cid=next(cid for cid,card in get_cards().items() if card.card_name=="Isildur's Bane")
        state.hands[f.SH].hand=[cid]
        for name in ('sh_char_deck','sh_strat_deck'):
            deck=getattr(state,name)
            if cid in deck:
                deck.remove(cid)
        with patch.object(gl.random,'choice',side_effect=lambda values:values[-1]):
            state=f.choose(state,'play_event',side=f.SH,card_id=cid)
        state=f.redraw(state,yes=False)
        self.assertEqual(state.pending_card_option['card_name'],'Mithril Coat and Sting')
        self.assertEqual(repr(state._mithril_stash['tile']),'3')
        self.assertEqual(state.fellowship.corruption,1)
        state=f.redraw(state,yes=False)
        final=f.choose(state,'hunt_take_corruption')
        self.assertEqual(final.fellowship.corruption,4)
        self.assertEqual(final.hunt_pool,[])
        self.assertEqual(len(final.hunt_discard),16)
        self.assertFalse(final._reveal_place_due)
        self.assertIsNone(final.pending_card_option)
        self.assertEqual(len(final.hunt_box),1)

if __name__=='__main__':
    unittest.main(verbosity=2)
