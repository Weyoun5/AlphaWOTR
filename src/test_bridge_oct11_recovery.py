"""Recovery regressions from physical client layouts and failed I/O.

These tests assert literal player-visible outcomes, not an engine replay used
as its own oracle. Actual Java scenario evidence lives in audit/bridge_oct11.
"""
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_live_driver import Rig, fresh_state
from game_state import Nation, Side
from wotr_bridge.live import adopt, shadow_siege_slots


def test_reset_without_dump_preserves_unfinished_hunt_and_evidence():
    rig = Rig(fresh_state(), Side.SHADOW, phase='actions')
    driver = rig.drv
    hunt = {'damage': 2, 'source': 'hunt'}
    driver.es.pending_hunt = hunt
    driver.last_actor = Side.SHADOW
    driver.view.refresh_dump = Mock(return_value=False)
    driver._track_owed = {'progress': (2, 99)}
    driver.alarms = ['missing board evidence']
    driver.rand.reset_feed = Mock()
    driver.feed.poll = Mock(return_value=[])

    _, report = driver.reset_and_resync()

    assert driver.es.pending_hunt is hunt
    assert driver.last_actor is Side.SHADOW
    assert driver._track_owed == {'progress': (2, 99)}
    assert driver.alarms == ['missing board evidence']
    driver.rand.reset_feed.assert_not_called()
    driver.feed.poll.assert_not_called()
    assert any('incomplete' in line for line in report)


def test_refused_adoption_preserves_driver_buffers_and_obligations():
    rig = Rig(fresh_state(), Side.SHADOW, phase='actions')
    driver = rig.drv
    buffer = SimpleNamespace(_reveals=[])
    driver.pending_buf = buffer
    driver._track_owed = {'progress': (2, 99)}
    driver._pawn_owed = ('hollin', 99)
    echo = {'who': 'human', 'norm': 'expected native acknowledgement'}
    driver.echo_ledger = [echo]
    driver.view.refresh_dump = Mock(return_value=True)
    driver.feed.poll = Mock(return_value=[])
    with patch.object(adopt, 'adopt_client_board', return_value=(
            ['ADOPTION DECLINED: unstable board'], 1, 1)):
        driver.resync_recover(drain_feed=True)

    assert driver.pending_buf is buffer
    assert driver._track_owed == {'progress': (2, 99)}
    assert driver._pawn_owed == ('hollin', 99)
    assert driver.echo_ledger == [echo]
    driver.feed.poll.assert_not_called()


def test_fp_can_retreat_into_shared_box_of_captured_shadow_stronghold():
    state = fresh_state()
    state.pending_combat = {'phase': 'defender_siege_choice',
                            'defender_region': 'morannon',
                            'defender_side': Side.FREE_PEOPLES}
    view = SimpleNamespace(dynamic_types={'test-elf': (Nation.ELVES, 'elite')},
                           piece_region={'test-elf': 'shadow stronghold'})
    record = {'op': 'move', 'from': 'Morannon', 'to': 'Shadow Stronghold',
              'obj': 'Elven Elite', 'meta': {'pids': ['test-elf']}}
    shadow_siege_slots.bind_record(view, state, record)
    assert record['to'] == 'morannon stronghold'
    assert view.piece_region['test-elf'] == 'morannon stronghold'


def test_resync_captured_morannon_recognizes_fp_inside_and_shadow_outside():
    state = fresh_state()
    # The physical garrison is FP. A stale engine incorrectly has native
    # Shadow ownership. Ten besiegers must not become a five-unit garrison.
    dump = {'elf': 'morannon stronghold', 'orc': 'morannon'}
    types = {'elf': (Nation.ELVES, 'elite'),
             'orc': (Nation.SAURON, 'elite')}
    adopt.adopt_sieges(state, dump, types, allow_drop=True)
    assert 'morannon' in state.besieged_strongholds
    assert 'morannon' in state.enemy_controlled[Side.FREE_PEOPLES]
    assert 'morannon' not in state.enemy_controlled[Side.SHADOW]


def test_automatic_siege_detection_does_not_change_settlement_control():
    state = fresh_state()
    adopt.adopt_sieges(state, {'elf': 'morannon stronghold', 'orc': 'morannon'},
                       {'elf': (Nation.ELVES, 'elite'),
                        'orc': (Nation.SAURON, 'elite')})
    assert 'morannon' not in state.enemy_controlled[Side.FREE_PEOPLES]


def test_manual_control_markers_repair_a_missed_capture_and_stale_capture():
    state = fresh_state()
    state.enemy_controlled[Side.SHADOW].add('dale')
    adopt.adopt_control_markers(state, {'405': 'morannon',
                                      '411': 'SA Reinforcements'})
    assert state.enemy_controlled[Side.FREE_PEOPLES] == {'morannon'}
    assert state.enemy_controlled[Side.SHADOW] == set()


def test_partial_adoption_rolls_back_state_and_new_pending():
    rig = Rig(fresh_state(), Side.SHADOW, phase='actions')
    driver = rig.drv
    hunt = {'damage': 2}
    driver.es.pending_hunt = None
    original_progress = driver.es.fellowship.progress
    driver.view.refresh_dump = Mock(return_value=True)
    driver.feed.poll = Mock(return_value=[])
    def incomplete(state, *args, **kwargs):
        state.fellowship.progress = 8
        state.pending_hunt = hunt
        return ['board adopted: divergences 2 -> 1'], 2, 1
    with patch.object(adopt, 'adopt_client_board', side_effect=incomplete) as adopter, \
            patch.object(driver, '_adoption_outcome', return_value={
                'accepted': True, 'complete': False, 'remaining': ['unmatched die']}):
        driver.reset_and_resync()
    adopter.assert_called_once()
    assert driver.es.fellowship.progress == original_progress
    assert driver.es.pending_hunt is None
    driver.feed.poll.assert_not_called()


def test_manual_resync_recovers_missing_minion_and_later_elimination():
    state = fresh_state()
    adopt.adopt_minions(state, {'254': 'orthanc'}, manual=True)
    assert state.saruman.in_play
    assert state.saruman.was_ever_in_play
    leaders = [leader for army in state.armies.values() for leader in army.leaders
               if getattr(leader, 'name', None) == 'Saruman']
    assert len(leaders) == 1
    adopt.adopt_minions(state, {'254': 'SA Casualties'}, manual=True)
    assert not state.saruman.in_play
    assert state.saruman.was_ever_in_play
    assert all(getattr(leader, 'name', None) != 'Saruman'
               for army in state.armies.values() for leader in army.leaders)


def test_failed_adoption_restores_minion_flags_and_physical_roster():
    state = fresh_state()
    snapshot = adopt._snapshot_armies(state)
    adopt.adopt_minions(state, {'254': 'orthanc'}, manual=True)
    adopt._restore_armies(state, snapshot)
    assert not state.saruman.in_play
    assert not state.saruman.was_ever_in_play
    assert all(getattr(leader, 'name', None) != 'Saruman'
               for army in state.armies.values() for leader in army.leaders)


if __name__ == '__main__':
    import unittest
    suite = unittest.TestSuite(unittest.FunctionTestCase(test)
             for name, test in list(globals().items()) if name.startswith('test_'))
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
