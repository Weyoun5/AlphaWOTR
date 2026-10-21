"""Physical casualty accounting from the real Folde/Great Host failure.

Offline controls only. Native IDs, casualty destinations and remaining hit
points are asserted independently; these tests do not count as desktop games.
"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'wotr_bridge'), str(ROOT)]
import ai
import emitter
import game_state as gs
import test_live_emission as views
from test_audit0905_casualty_order import position, rolled, play, FP, SH, TARGET, SOURCE
from wotr_bridge.live import emission


def great_host_position():
    state = rolled(position(attack_units=(7, 2), defend_units=(1, 3), great_host=True),
                   attacker_hits=3, defender_hits=0)
    state = play(state, FP, 'casualty_allocate', operation='remove_regular')
    # Use the actual failed battle's nation and destination, retaining the
    # existing fixture's literal rolls and Great Host result.
    state.armies['folde'] = state.armies.pop(TARGET)
    state.armies['folde'].region = 'folde'
    for unit in state.armies['folde'].units:
        unit.nation = gs.Nation.ROHAN
    state.pending_combat['defender_region'] = 'folde'
    state.fp_casualties[gs.Nation.ROHAN] = {'regular': 1, 'elite': 0, 'leader': 0}
    state.fp_casualties[gs.Nation.GONDOR] = {'regular': 0, 'elite': 0, 'leader': 0}
    chosen = next(action for action in ai.get_legal_actions(state, FP)
                  if action.action_type == 'casualty_allocate'
                  and action.params.get('operation') == 'remove_elite')
    post = ai.apply_action(state, chosen)
    view = views.make_view([state])
    for pid in ('287', '288', '289'):
        view.piece_region[pid] = 'Folde'
    # The first normal casualty is the only correct replacement source.
    view.piece_region['290'] = 'FP Dead'
    return state, post, chosen, view


def row_set(units=None, cas=None, sources=None):
    result = {'units': units or {}, 'naz': {'minus': [], 'plus': []},
              'named': {'leaves': {}, 'joins': {}}, 'cas': cas or {}, 'unexplained': []}
    if sources is not None:
        result['downgrade_sources'] = sources
    return result


class ForcedCasualtyRows(unittest.TestCase):
    def test_selected_elite_and_great_host_downgrade_each_render_exactly_once(self):
        pre, post, action, view = great_host_position()
        self.assertEqual(sorted(unit.kind.value.lower() for unit in post.armies['folde'].units),
                         ['elite', 'regular'])
        self.assertEqual(post.fp_casualties[gs.Nation.ROHAN]['elite'], 2)
        self.assertEqual(post.fp_casualties[gs.Nation.ROHAN]['regular'], 0)
        self.assertEqual(post._dg_src_log[-1], (gs.Nation.ROHAN, 'cas'))
        selected = emission.render_decision(view, 'WOTR_AI_FP', FP,
            'combat:casualty_allocation', action, pre=pre, post=post)
        self.assertFalse(selected.hold, selected.hold)
        self.assertEqual(len(selected.lines), 1)
        voiced, failed = emission.expressed_rows(view, selected.ops, 'fp')
        self.assertFalse(failed)
        self.assertEqual(voiced['cas'], {'rohan:elite': 1})
        residual = emission.subtract_rows(emission.removal_rows(pre, post, 'fp'), voiced)
        self.assertEqual(residual['cas'].get('rohan:elite'), 1)
        self.assertEqual(residual.get('downgrade_sources'), [('rohan', 'cas')])
        extra, ledger = emission.render_forced_rows(view, 'WOTR_AI_FP', FP,
            residual, pre, post=post, label='Great Host after selected Elite')
        self.assertFalse(extra.hold, extra.hold)
        self.assertEqual(len(extra.lines), 2)
        self.assertEqual(len(ledger), 1)
        selected_pid = selected.ops[0][1]['pieces'][0]
        killed = next(fields['pieces'][0] for op, fields in extra.ops if op == 'hand_exit_b')
        self.assertNotEqual(selected_pid, killed)
        replacement = next(fields for op, fields in extra.ops
                           if fields.get('pieces') == ['290'])
        self.assertEqual(replacement['frm'], 'Game.2039')
        self.assertEqual(replacement['to'], emitter.zone_code_for('folde'))

    def test_subtracts_death_credits_without_changing_input_or_nation(self):
        total = row_set({'rohan:elite': {'minus': [('folde', 2)], 'plus': []}},
                        {'rohan:elite': 2, 'gondor:elite': 1})
        selected = row_set({'rohan:elite': {'minus': [('folde', 1)], 'plus': []}},
                           {'rohan:elite': 1})
        before = deepcopy((total, selected))
        result = emission.subtract_rows(total, selected)
        self.assertEqual(result['cas'], {'rohan:elite': 1, 'gondor:elite': 1})
        self.assertEqual((total, selected), before)

    def test_native_kill_and_explicit_casualty_move_supply_same_death_credit(self):
        pre, _, _, view = great_host_position()
        region = emitter.zone_code_for('folde')
        for op, fields in (
            ('hand_exit_b', {'pieces': ['287'], 'frm': region}),
            ('swap_place', {'pieces': ['287'], 'frm': region}),
            ('move', {'pieces': ['287'], 'frm': region, 'to': 'Game.2039'}),
        ):
            with self.subTest(op=op):
                rows, failed = emission.expressed_rows(view, [(op, fields)], 'fp')
                self.assertFalse(failed)
                self.assertEqual(rows['cas'], {'rohan:elite': 1})

    def test_return_to_reinforcements_and_army_moves_are_not_deaths(self):
        _, _, _, view = great_host_position()
        region = emitter.zone_code_for('folde')
        for target in ('Game.2038', emitter.zone_code_for('edoras')):
            rows, failed = emission.expressed_rows(view, [('move', {
                'pieces': ['287'], 'frm': region, 'to': target})], 'fp')
            self.assertFalse(failed)
            self.assertEqual(rows['cas'], {})

    def test_residual_retains_unaccounted_loss_instead_of_reusing_selected_credit(self):
        pre, post, _, view = great_host_position()
        total = row_set({'rohan:elite': {'minus': [('folde', 2)], 'plus': []}},
                        {'rohan:elite': 1})
        chosen = row_set({'rohan:elite': {'minus': [('folde', 1)], 'plus': []}},
                         {'rohan:elite': 1})
        residual = emission.subtract_rows(total, chosen)
        result, _ = emission.render_forced_rows(view, 'WOTR_AI_FP', FP,
                                               residual, pre, post=post)
        self.assertTrue(result.hold)
        self.assertEqual(result.lines, [])

    def test_downgrade_source_subtraction_preserves_remaining_source_order(self):
        total = row_set(sources=[('rohan', 'cas'), ('gondor', 'pool'), ('rohan', 'pool')])
        chosen = row_set(sources=[('rohan', 'cas')])
        result = emission.subtract_rows(total, chosen)
        self.assertEqual(result.get('downgrade_sources'), [('gondor', 'pool'), ('rohan', 'pool')])

    def test_selected_downgrade_then_forced_replacement_death_consumes_source_once(self):
        pre, post, _, view = great_host_position()
        region = emitter.zone_code_for('folde')
        chosen_ops = [
            ('swap_place', {'nick': 'WOTR_AI_FP', 'pieces': ['287'], 'frm': region}),
            ('swap_remove', {'nick': 'WOTR_AI_FP', 'pieces': ['290'],
                             'frm': 'Game.2039', 'to': region}),
        ]
        total = row_set({'rohan:elite': {'minus': [('folde', 1)], 'plus': []}},
                        {'rohan:elite': 1, 'rohan:regular': 1}, [('rohan', 'cas')])
        voiced, failed = emission.expressed_rows(view, chosen_ops, 'fp')
        self.assertFalse(failed)
        self.assertEqual(voiced.get('downgrade_sources'), [('rohan', 'cas')])
        residual = emission.subtract_rows(total, voiced)
        self.assertEqual(residual.get('downgrade_sources'), [])
        self.assertEqual(residual['cas']['rohan:regular'], 1)
        self.assertEqual(residual['units']['rohan:regular'],
                         {'minus': [('folde', 1)], 'plus': []})
        emission._retrack_ops(view, chosen_ops)
        extra, _ = emission.render_forced_rows(view, 'WOTR_AI_FP', FP,
                                               residual, pre, post=post)
        self.assertFalse(extra.hold, extra.hold)
        self.assertEqual(len(extra.ops), 1)
        self.assertEqual(extra.ops[0][0], 'hand_exit_b')
        self.assertEqual(extra.ops[0][1]['pieces'], ['290'])

    def test_two_regular_deaths_do_not_reuse_first_rendered_death(self):
        total = row_set({'rohan:regular': {'minus': [('folde', 2)], 'plus': []}},
                        {'rohan:regular': 2})
        chosen = row_set({'rohan:regular': {'minus': [('folde', 1)], 'plus': []}},
                         {'rohan:regular': 1})
        residual = emission.subtract_rows(total, chosen)
        self.assertEqual(residual['cas'], {'rohan:regular': 1})
        self.assertEqual(residual['units']['rohan:regular']['minus'], [('folde', 1)])

    def test_offboard_kill_does_not_certify_an_army_loss(self):
        _, _, _, view = great_host_position()
        for source in ('Game.2038', 'Game.2039'):
            rows, _ = emission.expressed_rows(view, [('hand_exit_b', {
                'pieces': ['287'], 'frm': source})], 'fp')
            self.assertEqual(rows['cas'], {})

    def test_shadow_kills_do_not_change_fp_death_accounting(self):
        _, _, _, view = great_host_position()
        rows, failed = emission.expressed_rows(view, [('hand_exit_b', {
            'pieces': ['006'], 'frm': emitter.zone_code_for('orthanc')})], 'shadow')
        self.assertFalse(failed)
        self.assertEqual(rows['cas'], {})


if __name__ == '__main__':
    unittest.main(verbosity=2)
