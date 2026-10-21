"""Literal peaceful-leader movement bounds; attack has a separate forced guard."""
from pathlib import Path
import os,sys,unittest
source=Path(os.environ.get('WOTR_GUARD_TEST_SOURCE',str(Path(__file__).resolve().parent)))
sys.path.insert(0,str(source))
os.environ.update(CUDA_VISIBLE_DEVICES='-1',WOTR_NATIVE='0',WOTR_ACT_GEN='v52')
import ai,board,game_state as gs,army_participants


class PeacefulLeaderMovementTests(unittest.TestCase):
    def position(self,face=gs.ActionDieFace.ARMY):
        state=gs.setup_initial_game(board.load_board())
        state.armies={'dale':gs.Army(state.board.region('dale'),
            [gs.Unit(gs.Nation.THE_NORTH,gs.UnitKind.REGULAR),gs.Unit(gs.Nation.THE_NORTH,gs.UnitKind.ELITE)],
            [gs.Leader(gs.Nation.THE_NORTH),gs.Leader(gs.Nation.ELVES)])}
        for nation in gs.Nation:state.political.boxes[nation]=3
        state.political.boxes[gs.Nation.ELVES]=0
        for dice in state.dice.values():
            for die in dice:die.used=True
        state.dice[gs.Side.FREE_PEOPLES][0].used=False
        state.dice[gs.Side.FREE_PEOPLES][0].current=face
        return state

    def menu(self,state,face):
        return ai._get_army_move_actions(state,gs.Side.FREE_PEOPLES,0,
            require_leader=face==gs.ActionDieFace.CHARACTER)

    def matching(self,state,destination,regular=1,elite=1,face=gs.ActionDieFace.ARMY):
        return [a for a in self.menu(state,face) if a.params.get('to')==destination
            and a.params.get('n_regular')==regular and a.params.get('n_elite')==elite
            and a.params.get('take_leaders') and 'n_leaders' not in a.params]

    def test_foreign_crossing_cannot_vacate_peaceful_leader(self):
        for face in (gs.ActionDieFace.ARMY,gs.ActionDieFace.CHARACTER):
            with self.subTest(face=face):
                state=self.position(face)
                self.assertFalse(self.matching(state,'old forest road',face=face))

    def test_leaving_regular_allows_elite_and_belligerent_leader_to_move(self):
        for face in (gs.ActionDieFace.ARMY,gs.ActionDieFace.CHARACTER):
            with self.subTest(face=face):
                state=self.position(face);actions=self.matching(state,'old forest road',0,1,face)
                self.assertEqual(len(actions),1)
                result=ai.apply_action(state,actions[0]);rear=result.armies['dale']
                self.assertEqual([(u.nation,u.kind) for u in rear.units],[(gs.Nation.THE_NORTH,gs.UnitKind.REGULAR)])
                self.assertEqual([l.nation for l in rear.leaders],[gs.Nation.ELVES])

    def test_nonmoving_peaceful_unit_can_guard_its_leader(self):
        state=self.position();state.armies['dale'].units.append(gs.Unit(gs.Nation.ELVES,gs.UnitKind.REGULAR))
        actions=self.matching(state,'old forest road');self.assertEqual(len(actions),1)
        rear=ai.apply_action(state,actions[0]).armies['dale']
        self.assertEqual([(u.nation,u.kind) for u in rear.units],[(gs.Nation.ELVES,gs.UnitKind.REGULAR)])
        self.assertEqual([l.nation for l in rear.leaders],[gs.Nation.ELVES])

    def test_at_war_leader_can_follow_full_army(self):
        state=self.position();state.political.boxes[gs.Nation.ELVES]=3
        self.assertEqual(len(self.matching(state,'old forest road')),1)

    def test_peaceful_leader_can_move_into_own_nation_or_neutral_region(self):
        for target in ('woodland realm','northern rhovanion'):
            with self.subTest(target=target):
                state=self.position();actions=self.matching(state,target);self.assertEqual(len(actions),1)
                result=ai.apply_action(state,actions[0])
                moving=next(a for k,a in result.armies.items() if board.sanitize(k)==target)
                self.assertEqual(len(moving.units),2)
                self.assertEqual([l.nation for l in moving.leaders],[gs.Nation.THE_NORTH,gs.Nation.ELVES])

    def test_attack_into_own_nation_still_reserves_peaceful_leader_guard(self):
        state=self.position()
        state.armies['woodland realm']=gs.Army(state.board.region('woodland realm'),[gs.Unit(gs.Nation.SAURON,gs.UnitKind.REGULAR)],[])
        formation=army_participants.select_forced_attack(state,state.armies['dale'],gs.Side.FREE_PEOPLES)
        self.assertTrue(formation.available)
        self.assertTrue(formation.guard_added)
        self.assertEqual([u.kind for u in formation.rearguard_units],[gs.UnitKind.REGULAR])
        self.assertEqual([l.nation for l in formation.rearguard_leaders],[gs.Nation.ELVES])
        self.assertEqual([u.kind for u in formation.attacking_units],[gs.UnitKind.ELITE])

    def test_other_peaceful_leader_nation_has_same_requirement(self):
        state=self.position();state.armies['dale'].leaders[-1]=gs.Leader(gs.Nation.DWARVES)
        state.political.boxes[gs.Nation.DWARVES]=0
        self.assertFalse(self.matching(state,'old forest road'))


if __name__=='__main__':unittest.main()
