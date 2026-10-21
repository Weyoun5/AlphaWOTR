"""Literal public reveal routes and required intermediate stronghold draws."""
from pathlib import Path
import os
import pickle
import sys
import unittest
from unittest.mock import patch

SOURCE=Path(os.environ.get('WOTR_TIMING_SOURCE', str(Path(__file__).resolve().parent))).resolve()
GENERATION=os.environ.get('WOTR_TRANSIT_GENERATION','v48')
os.environ.update(WOTR_ACT_GEN=GENERATION,WOTR_TIMING_GENERATION=GENERATION,WOTR_NATIVE='0',CUDA_VISIBLE_DEVICES='-1')
sys.path.insert(0,str(SOURCE))
import test_audit0905_mithril_return as f
assert Path(f.ai.__file__).resolve().parent==SOURCE

def crossing(staff=False, conquered=False, redraw=False, progress=2, extra_two=False):
    state=f.setup(original='3' if redraw else '1R',replacement='1R' if redraw else '3')
    state.fellowship.region=state.board.region('hollin')
    state.fellowship.progress=progress-1
    if extra_two:
        tile=next(t for t in state.hunt_discard if repr(t)=='2')
        state.hunt_discard.remove(tile)
        state.hunt_pool.insert(0,tile)
    if conquered:
        moria=f.gl.get_or_create_army(state,'moria')
        shadow=f.gl.get_or_create_army(state,'dol guldur')
        lorien=f.gl.get_or_create_army(state,'lorien')
        shadow.units.extend(moria.units)
        shadow.leaders.extend(moria.leaders)
        moria.units,moria.leaders=lorien.units,lorien.leaders
        lorien.units,lorien.leaders=[],[]
        state.enemy_controlled[f.FP].add('moria')
        state.political.boxes[f.gs.Nation.ELVES]=3
        state.political.active[f.gs.Nation.ELVES]=True
    if staff:
        state,_=f.add_protection_through_public_play(state,"Wizard's Staff")
    state=f.move(state)
    if staff:
        state=f.choose(state,'card_option',choices={'wizardstaff_prevent':False})
    state=f.redraw(state,yes=redraw)
    return f.choose(state,'hunt_take_corruption')

def human_route(state, region, route):
    action=next(a for a in f.ai.get_legal_actions(state,f.FP)
                if a.action_type=='fellowship_reveal_place' and a.params['region']==region)
    action=f.ai.Action(die_idx=action.die_idx,action_type=action.action_type,
                       params={**action.params,'path':route})
    with patch.object(f.gl.random,'choice',side_effect=lambda values:values[-1]), \
            patch.object(f.gl.random,'shuffle',side_effect=lambda values:None):
        return f.ai.apply_action(state,action)

class TransitTests(unittest.TestCase):
    def test_sufficient_progress_allows_a_longer_route_avoiding_moria(self):
        state=crossing(progress=6)
        route=['hollin','fords of bruinen','high pass','goblin s gate','old ford','gladden fields','dimrill dale']
        for a,b in zip(route,route[1:]):
            self.assertIn(b,state.board.adjacent(a))
        state=f.place(state,'dimrill dale')
        self.assertEqual(state.fellowship.corruption,1)
        self.assertEqual(list(map(repr,state.hunt_pool)),['3'])
        self.assertIsNone(state.pending_card_option)

    def test_human_route_charges_each_stronghold_separately_with_clone_safe_continuation(self):
        state=crossing(progress=5,extra_two=True)
        route=['hollin','moria','dimrill dale','south anduin vale','dol guldur','eastern brown lands']
        for a,b in zip(route,route[1:]):
            self.assertIn(b,state.board.adjacent(a))
        state=human_route(state,'eastern brown lands',route)
        self.assertEqual(state._pending_reveal_strongholds,('dol guldur',))
        self.assertEqual(repr(state._mithril_stash['tile']),'3')
        self.assertEqual(f.gl.fellowship_region(state),'eastern brown lands')
        self.assertEqual(state.fellowship.progress,0)
        self.assertEqual(len(state.hunt_box),1)
        before=pickle.dumps(state)
        for branch in (f.ai.clone_state(state),pickle.loads(before)):
            branch=f.choose(f.redraw(branch,yes=False),'hunt_take_corruption')
            self.assertEqual(branch.fellowship.corruption,4)
            self.assertEqual(repr(branch._mithril_stash['tile']),'2')
            self.assertEqual(branch._pending_reveal_strongholds,())
            self.assertEqual(len(branch.hunt_box),1)
            branch=f.choose(f.redraw(branch,yes=False),'hunt_take_corruption')
            self.assertEqual(branch.fellowship.corruption,6)
            self.assertEqual(branch.hunt_pool,[])
            self.assertEqual(len(branch.hunt_discard),16)
            self.assertEqual(len(branch.hunt_box),2)
            self.assertEqual(f.inventory(branch),f.ORDINARY)
        self.assertEqual(pickle.dumps(state),before)

    def test_invalid_explicit_human_route_is_refused_without_mutating_parent(self):
        state=crossing()
        for route in (['hollin','dimrill dale'],['moria','dimrill dale'],
                      ['hollin','fords of bruinen','high pass','goblin s gate','old ford','gladden fields','dimrill dale']):
            before=pickle.dumps(state)
            with self.subTest(route=route),self.assertRaises(ValueError):
                human_route(state,'dimrill dale',route)
            self.assertEqual(pickle.dumps(state),before)

    def test_only_two_step_route_requires_moria_draw_and_protection(self):
        state=crossing()
        self.assertEqual(state.fellowship.progress,2)
        self.assertIn('moria',state.board.adjacent('hollin'))
        self.assertIn('dimrill dale',state.board.adjacent('moria'))
        state=f.place(state,'dimrill dale')
        self.assertIsNotNone(state.pending_card_option)
        self.assertEqual(state.pending_card_option['card_name'],'Mithril Coat and Sting')
        self.assertEqual(repr(state._mithril_stash['tile']),'3')
        self.assertEqual(f.gl.fellowship_region(state),'dimrill dale')
        self.assertEqual(len(state.hunt_box),1)
        state=f.choose(f.redraw(state,yes=False),'hunt_take_corruption')
        self.assertEqual(state.fellowship.corruption,4)
        self.assertEqual(state.hunt_pool,[])
        self.assertEqual(len(state.hunt_discard),16)
        self.assertEqual(len(state.hunt_box),2)

    def test_captured_moria_does_not_require_shadow_draw(self):
        state=crossing(conquered=True)
        self.assertIn('moria',state.enemy_controlled[f.FP])
        state=f.place(state,'dimrill dale')
        self.assertEqual(state.fellowship.corruption,1)
        self.assertEqual(list(map(repr,state.hunt_pool)),['3'])
        self.assertIsNone(state.pending_card_option)
        self.assertEqual(len(state.hunt_box),2)

    def test_staff_prevents_transit_draw_before_tile_is_removed(self):
        state=f.place(crossing(staff=True),'dimrill dale')
        self.assertIsNotNone(state.pending_card_option)
        self.assertEqual(state.pending_card_option['card_name'],"Wizard's Staff")
        self.assertEqual(list(map(repr,state.hunt_pool)),['3'])
        state=f.choose(state,'card_option',choices={'wizardstaff_prevent':True})
        self.assertEqual(state.fellowship.corruption,1)
        self.assertEqual(list(map(repr,state.hunt_pool)),['3'])
        self.assertEqual(len(state.hunt_discard),15)
        self.assertEqual(len(state.hunt_box),2)

    def test_rejected_original_stays_out_of_required_transit_refill(self):
        state=crossing(redraw=True)
        state=f.place(state,'dimrill dale')
        self.assertIsNotNone(state.pending_hunt)
        self.assertEqual(state.pending_hunt['damage'],1)
        self.assertEqual(repr(state._mithril_stash['tile']),'3')
        self.assertTrue(state._mithril_stash['return_pending'])
        held=state._mithril_stash['tile']
        self.assertTrue(all(tile is not held for tile in state.hunt_pool))
        self.assertEqual(f.inventory(state),f.ORDINARY)
        state=f.choose(state,'hunt_take_corruption')
        self.assertEqual(state.fellowship.corruption,2)
        self.assertEqual(len(state.hunt_pool),15)
        self.assertEqual(len(state.hunt_discard),1)
        self.assertEqual(len(state.hunt_box),2)

if __name__=='__main__':unittest.main(verbosity=2)
