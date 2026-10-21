"""Literal physical gestures and menu identities for sequential casualties.

These are Python adapter controls, not claims about a running Java client.
Expected piece ids, kinds and remaining budgets are fixed independently.
"""
import sys
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'wotr_bridge', ROOT / 'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import ai
import game_state as gs
import emitter
import test_live_emission as TE
# Match the production server namespace: installing both package aliases
# gives engine-global randomness funnels two different active owners.
from live import gesture_match as gm, live_parser, emission
from live.live_server import LiveServer

SH, FP = gs.Side.SHADOW, gs.Side.FREE_PEOPLES
I, S = gs.Nation.ISENGARD, gs.Nation.SAURON


def position(hits=3):
    state, _ = TE._drive_to_battle()
    army = ai._armies_lookup(state, 'orthanc')
    army.units = [gs.Unit(I, gs.UnitKind.REGULAR) for _ in range(4)]
    army.units += [gs.Unit(I, gs.UnitKind.ELITE), gs.Unit(S, gs.UnitKind.ELITE)]
    state.pending_combat = dict(phase='casualty_allocation', alloc_stage='attacker',
        attacker_region='Orthanc', defender_region='Fords of Isen',
        attacker_side=SH, defender_side=FP, deciding_side=SH,
        atk_hits=hits, def_hits=0, rounds_fought=1, is_siege=False)
    return state


def action(state, operation, nation=I):
    return next(a for a in ai.get_legal_actions(state, SH)
                if a.action_type == 'casualty_allocate'
                and a.params.get('operation') == operation
                and a.params.get('nation') == nation)


def view(state):
    result = TE.make_view([state])
    result.piece_region['035'] = 'Orthanc'  # Literal Sauron Elite.
    return result


class Window:
    def __init__(self, *lines):
        self.records = [live_parser.parse(line) for line in lines]
        self.die_side = self.die_face = None
        self.closed = 'test'
    def body_records(self):
        return self.records


class CasualtyBridgeOperations(unittest.TestCase):
    def test_whole_elite_emits_selected_nation_only(self):
        state = position()
        for nation, expected in ((I, '006'), (S, '035')):
            with self.subTest(nation=nation):
                chosen = action(state, 'remove_elite', nation)
                ops = emitter.plan(chosen.action_type, chosen.params, view(state), 'Wey', 'SHADOW')
                self.assertEqual(ops, [('hand_exit_b', {'nick': 'Wey', 'pieces': [expected],
                    'frm': emitter.zone_code_for('orthanc')})])
                self.assertEqual(state.pending_combat['atk_hits'], 3)

    def test_reduction_emits_correct_nation_and_source(self):
        state = position()
        chosen = action(state, 'reduce_elite', S)
        post = ai.apply_action(state, chosen)
        result = emission.render_decision(view(state), 'Wey', SH,
            'combat:casualty_allocation', chosen, post=post, pre=state)
        self.assertFalse(result.hold, result.hold)
        ops = [emitter.classify_line(line) for line in result.lines]
        self.assertEqual([op for op, fields in ops], ['swap_place', 'swap_remove'])
        self.assertEqual(ops[0][1]['pieces'], ['035'])
        pid = ops[1][1]['pieces'][0]
        nation, kind = emitter._unit_type(pid, view(state), side_hint='shadow')
        self.assertEqual((nation, kind), (S, 'regular'))
        self.assertIn(pid, emitter._pieces_in(view(state), 'sa reinf'))

    def test_missing_wrong_or_unavailable_stamp_refuses(self):
        state = position()
        chosen = action(state, 'reduce_elite', I)
        for stamps in ([], [(S, 'pool')], [(I, 'nonsense')], [(I, 'cas')]):
            with self.subTest(stamps=stamps):
                params = dict(chosen.params, _dg_srcs=stamps)
                with self.assertRaises(LookupError):
                    emitter.plan('casualty_allocate', params, view(state), 'Wey', 'SHADOW')

    def test_source_none_removes_elite_without_replacement(self):
        state = position()
        params = dict(action(state, 'reduce_elite').params, _dg_srcs=[(I, None)])
        ops = emitter.plan('casualty_allocate', params, view(state), 'Wey', 'SHADOW')
        self.assertEqual([op for op, fields in ops], ['swap_place'])
        self.assertEqual(ops[0][1]['pieces'], ['006'])

    def test_legacy_or_wrong_owner_refuses(self):
        state = position()
        for params, owner in (({'count': 0}, 'SHADOW'),
                              (action(state, 'remove_elite').params, 'FREE_PEOPLES')):
            with self.assertRaises(LookupError):
                emitter.plan('casualty_allocate', params, view(state), 'Wey', owner)

    def test_grouped_regular_losses_use_remaining_budget_not_global_depth(self):
        state = position()
        result = gm.match_window(state, SH, Window(
            '$Wey kills group ([3]Isengard Regular) in Orthanc.'), depth=0)
        self.assertEqual(result.status, 'match', result)
        self.assertEqual([a.params['operation'] for a in result.chain], ['remove_regular'] * 3)
        army = ai._armies_lookup(result.post, 'orthanc')
        self.assertEqual(sum(u.nation == I and u.kind == gs.UnitKind.REGULAR for u in army.units), 1)
        self.assertEqual(sum(u.kind == gs.UnitKind.ELITE for u in army.units), 2)
        self.assertEqual(state.pending_combat['atk_hits'], 3)

    def test_split_losses_keep_same_owner_until_budget_spent(self):
        state = position()
        for remaining in (2, 1):
            result = gm.match_window(state, SH, Window('$Wey kills Isengard Regular in Orthanc.'))
            self.assertEqual(result.status, 'match', result)
            state = result.post
            self.assertEqual(state.pending_combat['atk_hits'], remaining)
            self.assertEqual(state.pending_combat['deciding_side'], SH)

    def test_wrong_nation_and_over_budget_gestures_do_not_match(self):
        state = position(2)
        for line in ('$Wey kills Gondor Elite in Orthanc.',
                     '$Wey kills group ([3]Isengard Regular) in Orthanc.'):
            with self.subTest(line=line):
                result = gm.match_window(state, SH, Window(line), depth=0)
                self.assertNotEqual(result.status, 'match', result)

    def test_whole_fp_elite_does_not_invent_replacement_casualty(self):
        from test_audit0905_casualty_order import position as battle
        state = battle(attacker=FP)
        state.pending_combat.update(phase='casualty_allocation', alloc_stage='attacker',
            atk_hits=2, def_hits=0, deciding_side=FP)
        result = gm.match_window(state, FP, Window('$Wey kills Gondor Elite in West Harondor.'))
        self.assertEqual(result.status, 'match', result)
        self.assertEqual([a.params['operation'] for a in result.chain], ['remove_elite'])
        self.assertEqual(result.post.fp_casualties.get(gs.Nation.GONDOR, {}).get('regular', 0), 0)
        self.assertEqual(result.post.fp_casualties[gs.Nation.GONDOR]['elite'], 1)

    def test_explicit_fp_reduction_then_kill_preserves_regular_casualty(self):
        from test_audit0905_casualty_order import position as battle
        state = battle(attacker=FP)
        state.pending_combat.update(phase='casualty_allocation', alloc_stage='attacker',
            atk_hits=2, def_hits=0, deciding_side=FP)
        result = gm.match_window(state, FP, Window(
            '$Wey kills Gondor Elite in West Harondor.',
            '$Wey moves Gondor Regular from FP Reinforcements to West Harondor.',
            '$Wey kills Gondor Regular in West Harondor.'))
        self.assertEqual(result.status, 'match', result)
        self.assertEqual(result.chain[0].params['operation'], 'reduce_elite')
        self.assertEqual(result.post.fp_casualties[gs.Nation.GONDOR]['regular'], 1)
        self.assertEqual(result.post.fp_casualties[gs.Nation.GONDOR]['elite'], 1)

    def test_parked_whole_elite_matches_without_fabricated_swap(self):
        state = position(2)
        result = gm.match_window(state, SH, Window(
            '$Wey moves Isengard Elite from Orthanc to Current SA Card.',
            '$Wey kills Isengard Elite in Current SA Card.'))
        self.assertEqual(result.status, 'match', result)
        self.assertEqual([a.params['operation'] for a in result.chain], ['remove_elite'])

    def test_global_apply_budget_still_holds(self):
        result = gm.match_window(position(), SH, Window(
            '$Wey kills group ([3]Isengard Regular) in Orthanc.'), depth=0, budget=1)
        self.assertEqual(result.status, 'hold')
        self.assertIn('budget', result.note)

    def test_preview_rows_distinguish_nation_and_operation(self):
        state = position()
        with tempfile.TemporaryDirectory() as folder:
            server = LiveServer(folder, FP, load_nets=False, seed=31)
            server.human_side = SH
            server.driver.es = state
            server.driver.hands.es = state
            server.driver.phase = 'actions'
            status = server.status_menu()['human_menu']
            self.assertNotIn('error', status, status)
            rows = [a for a in status.get('actions', []) if a['action_type'] == 'casualty_allocate']
            self.assertEqual(len(rows), 5, status)
            self.assertEqual(len({repr(a['rows']) for a in rows}), 5)
            for row in rows:
                self.assertIn('say_pick', row['params'])

    def test_pool_and_hit_contexts_do_not_collapse(self):
        state = position()
        changed = ai.clone_state(state)
        changed.pending_combat['atk_hits'] = 2
        self.assertNotEqual(gm._pending_ctx_tag(state, None), gm._pending_ctx_tag(changed, None))
        changed = ai.clone_state(state)
        changed.fp_casualties[gs.Nation.GONDOR] = {'regular': 1}
        self.assertNotEqual(gm._state_fingerprint(state), gm._state_fingerprint(changed))

    def test_signed_residual_units_preserve_cancelled_replacement(self):
        def rows(units):
            return dict(units=units, naz={'minus': [], 'plus': []},
                        named={'leaves': {}, 'joins': {}}, cas={}, unexplained=[])
        key = 'isengard:regular'
        total = rows({})  # Remove a Regular and place a replacement: net zero.
        voiced = rows({key: {'minus': [('orthanc', 1)], 'plus': []}})
        original = deepcopy(voiced)
        residual = emission.subtract_rows(total, voiced)
        self.assertEqual(residual['units'], {key: {'minus': [], 'plus': [('orthanc', 1)]}})
        self.assertEqual(voiced, original)
        reverse = emission.subtract_rows(total, rows({key: {'minus': [], 'plus': [('orthanc', 1)]}}))
        self.assertEqual(reverse['units'], {key: {'minus': [('orthanc', 1)], 'plus': []}})

    def test_signed_residual_keeps_nations_and_regions_separate(self):
        def rows(units):
            return dict(units=units, naz={'minus': [], 'plus': []},
                        named={'leaves': {}, 'joins': {}}, cas={}, unexplained=[])
        total = rows({'isengard:regular': {'minus': [('orthanc', 2)], 'plus': []},
                      'sauron:regular': {'minus': [], 'plus': [('moria', 1)]}})
        voiced = rows({'isengard:regular': {'minus': [('orthanc', 1)], 'plus': []},
                       'sauron:regular': {'minus': [], 'plus': [('moria', 1)]}})
        self.assertEqual(emission.subtract_rows(total, voiced)['units'],
                         {'isengard:regular': {'minus': [('orthanc', 1)], 'plus': []}})

    def test_forced_defender_witness_existing_literal_controls(self):
        from test_live_server import test_forced_casualty_delta_is_exact_and_opponent_only
        failures = []
        test_forced_casualty_delta_is_exact_and_opponent_only(failures)
        self.assertEqual(failures, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
