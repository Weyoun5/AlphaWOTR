"""House/Gandalf player-feed acceptance, specified before bridge changes.

AL16/46: House recruits the selected Gondor unit, draws two Strategy cards,
discards to six, then offers Gandalf's optional Character redraw. Taking it
may require a second discard. Literal input fixtures are not engine output.
The standalone pending fixture explicitly isolates a proposed contract;
the complete House tests must reach that contract through real player play.
"""
import copy
import json
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.environ.get('BRIDGE_TEST_PROJECT', ROOT))
sys.path[:0] = [str(PROJECT / 'src'), str(PROJECT / 'wotr_bridge'), str(PROJECT), str(ROOT / 'src')]
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('WOTR_ACT_GEN', 'v41')
import ai
import test_audit0904_stewards_choices as house
# Re-run against preserved live modules without replacing any source files.
if os.environ.get('WOTR_AUDIT_STEWARDS_BRIDGE_BASELINE'):
    import live
    baseline = Path(os.environ['WOTR_AUDIT_STEWARDS_BRIDGE_BASELINE'])
    for name in ('chat', 'gesture_match', 'driver'):
        if not (baseline / (name + '.py')).exists():
            continue
        spec = importlib.util.spec_from_file_location('live.' + name,
                                                       baseline / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        setattr(live, name, module)
        spec.loader.exec_module(module)
from live.driver import LiveDriver
from live.randomness import ClientRandomness
from live import chat

if os.environ.get('BRIDGE_TEST_PROJECT'):
    for name in ('ai', 'live.driver', 'live.gesture_match'):
        assert Path(sys.modules[name].__file__).resolve().is_relative_to(PROJECT.resolve())

TITLE = 'House of the Stewards/Brave Stand'
PLAY = '$Wey moves House of the Stewards/Brave Stand from Free Peoples Hand to Current FP Card.'
STRATEGY_DRAW = '$Wey draws Free Strategy Card.'
CHARACTER_DRAW = '$Wey draws Free Character Card.'
ATHELAS_DISCARD = '$Wey kills Athelas/Anduril in Free Peoples Hand.'
BILBO_DISCARD = "$Wey kills Bilbo's Song/Fateful Strike in Free Peoples Hand."
DECLINE = '<Wey> decline'
GANDALF_ID = '__gandalf_draw__'


def house_window(kind='elite', event=False):
    letter = 'P' if event else 'C'
    return [
        '$Wey moves Free Action Die (%s) from FP Dice to Current Action.' % letter,
        PLAY,
        '$Wey moves Gondor %s from FP Reinforcements to Minas Tirith.' % kind.title(),
        STRATEGY_DRAW, STRATEGY_DRAW,
        '$Wey kills group (Free Action Die (%s), %s) in Current Action.' % (letter, TITLE),
    ]


def prospective_gandalf_state():
    """Literal possible post-House position, isolating the proposed menu."""
    state = house.before_play(face=house.gs.ActionDieFace.EVENT)
    state.hands[house.FP].hand = [house.S1, house.S2]
    state.fp_strat_deck = [c for c in state.fp_strat_deck if c not in (house.S1, house.S2)]
    state.discard_face_up['fp_character'].append(house.CID)
    state.dice[house.FP][0].used = True
    state.pending_card_option = {
        'card_id': GANDALF_ID, 'card_name': 'Gandalf guide draw',
        'side': 'fp', 'decider': 'fp', 'direct': True, 'gandalf_draw': True,
        'source_card_id': house.CID, 'deck': 'character',
        'options': [{'gandalf_draw': True}, {'gandalf_draw': False}]}
    return state


class StewardsBridgeAcceptance(unittest.TestCase):
    diagnostics = []

    @classmethod
    def tearDownClass(cls):
        path = os.environ.get('WOTR_AUDIT_STEWARDS_BRIDGE_LOG')
        if path:
            Path(path).write_text(json.dumps(cls.diagnostics, indent=2,
                default=lambda obj: getattr(obj, 'name', repr(obj))) + '\n', encoding='utf-8')

    def rig(self, event=False, full_hand=False, state=None):
        temp = tempfile.TemporaryDirectory(prefix='wotr-stewards-')
        self.addCleanup(temp.cleanup)
        feed = Path(temp.name) / 'ai_feed.txt'
        feed.write_text('', encoding='utf-8')
        rand = ClientRandomness(seed=29)
        rand.install()
        if state is None:
            state = house.before_play(face=house.gs.ActionDieFace.EVENT if event
                                       else house.gs.ActionDieFace.CHARACTER,
                                      fillers=house.FILLERS if full_hand else ())
        driver = LiveDriver(state, house.SH, str(feed), randomness=rand,
                            nick='WOTR_AI_Shadow')
        driver.phase = 'actions'
        # This is an action-phase position. Leaving these at zero would
        # let an unowed bonus masquerade as an unused turn-start draw.
        driver.turn_draws = {(side, deck): 1 for side in (house.FP, house.SH)
                             for deck in ('character', 'strategy')}
        return driver, feed

    def feed(self, driver, path, *lines):
        if lines:
            with path.open('a', encoding='utf-8', newline='\n') as stream:
                stream.write('\n'.join(lines) + '\n')
        output = driver.tick()
        pending = driver.rand.pending()
        self.diagnostics.append(copy.deepcopy(dict(
            test=self.id(), lines=lines, output=list(map(repr, output)),
            hand=driver.es.hands[house.FP].hand,
            figures=house.figures(driver.es),
            strategy_left=len(driver.es.fp_strat_deck),
            character_left=len(driver.es.fp_char_deck),
            pending_card=getattr(driver.es, 'pending_card_option', None),
            pending_discard=getattr(driver.es, 'pending_discard', None),
            rng_pending=None if pending is None else dict(kind=pending.kind, kw=pending.kw),
            deferred=None if driver.deferred is None else dict(
                note=getattr(driver.deferred[0], '_starve_note', None),
                call=repr(getattr(driver.deferred[0], '_starve_call', None))),
            unconsumed=driver.rand.unconsumed(),
            alarms=driver.alarms, findings=list(map(repr, driver.findings)),
            commits=driver.n_commits_total)))
        return output

    def assert_paid(self, driver):
        self.assertNotIn(house.CID, driver.es.hands[house.FP].hand)
        self.assertEqual(driver.es.discard_face_up['fp_character'].count(house.CID), 1)
        self.assertEqual([d.used for d in driver.es.dice[house.FP]], [True, False])
        self.assertEqual([d.used for d in driver.es.dice[house.SH]], [False])
        self.assertEqual(driver.alarms, [])

    def ready_for_first_discard(self):
        driver, path = self.rig(event=True, full_hand=True)
        char_before = list(driver.es.fp_char_deck)
        self.feed(driver, path, *house_window(event=True))
        self.assertEqual(len(driver.es.hands[house.FP].hand), 7,
                         'Two required Strategy draws must commit before the first discard')
        self.assertEqual(driver.es.fp_char_deck, char_before,
                         'The first discard cannot see the later Gandalf draw')
        self.assertEqual(ai.describe_pending(driver.es)['kind'], 'discard')
        self.assertIsNone(driver.rand.pending(),
                          'Gandalf draw is not owed while the first discard is open')
        self.assert_paid(driver)
        return driver, path, char_before

    def ready_for_gandalf(self):
        driver, path, char_before = self.ready_for_first_discard()
        self.feed(driver, path, ATHELAS_DISCARD)
        self.assertEqual(len(driver.es.hands[house.FP].hand), 6)
        self.assertNotIn(house.FILLERS[0], driver.es.hands[house.FP].hand)
        self.assertEqual(driver.es.pending_card_option['card_id'], GANDALF_ID)
        self.assertEqual(driver.es.fp_char_deck, char_before)
        return driver, path, char_before

    def test_character_die_accepts_each_unit_then_exactly_two_strategy_draws(self):
        for kind, expected in [('regular', (4, 1, 1)), ('elite', (3, 2, 1))]:
            for split in (False, True):
                with self.subTest(kind=kind, split=split):
                    driver, path = self.rig()
                    char_before = list(driver.es.fp_char_deck)
                    strategy_count = len(driver.es.fp_strat_deck)
                    for lines in ([house_window(kind)] if not split
                                  else [[line] for line in house_window(kind)]):
                        self.feed(driver, path, *lines)
                    self.assertEqual(house.figures(driver.es), expected)
                    self.assertEqual(len(driver.es.hands[house.FP].hand), 2)
                    self.assertEqual(len(driver.es.fp_strat_deck), strategy_count - 2)
                    self.assertEqual(driver.es.fp_char_deck, char_before)
                    self.assertIsNone(ai.describe_pending(driver.es))
                    self.assert_paid(driver)

    def test_event_die_two_strategy_draws_precede_first_discard_and_bonus(self):
        self.ready_for_first_discard()

    def test_named_discard_not_in_hand_cannot_discard_a_placeholder(self):
        driver, path, _ = self.ready_for_first_discard()
        before = list(driver.es.hands[house.FP].hand)
        commits = driver.n_commits_total
        output = self.feed(driver, path,
                           '$Wey kills Gwaihir the Windlord/Fateful Strike in Free Peoples Hand.')
        self.assertEqual(driver.es.hands[house.FP].hand, before)
        self.assertEqual(driver.n_commits_total, commits)
        self.assertEqual(ai.describe_pending(driver.es)['kind'], 'discard')
        self.assertTrue(any(kind == 'discard_identity_stale'
                            for kind, _ in driver.findings), output)

    def test_deck_back_or_truncated_title_cannot_choose_known_discard(self):
        for title in ('Free Character Card', 'Athe'):
            with self.subTest(title=title):
                driver, path, _ = self.ready_for_first_discard()
                before = list(driver.es.hands[house.FP].hand)
                commits = driver.n_commits_total
                self.feed(driver, path,
                          '$Wey kills %s in Free Peoples Hand.' % title)
                self.assertEqual(driver.es.hands[house.FP].hand, before)
                self.assertEqual(driver.n_commits_total, commits)
                self.assertEqual(ai.describe_pending(driver.es)['kind'], 'discard')

    def test_discard_sidecar_pid_has_authority_over_rendered_title(self):
        from live import live_parser
        from live.driver import PhaseWindow
        driver, _, _ = self.ready_for_first_discard()
        driver.view.card_pid_map = {house.FILLERS[0]: '150', house.FILLERS[1]: '151'}
        rec = live_parser.parse(ATHELAS_DISCARD)
        rec['meta'] = {'pids': ['151']}
        driver.pending_buf = PhaseWindow()
        driver.pending_buf.records = [rec]
        dp = ai.describe_pending(driver.es)
        filtered = driver._discard_identity_filter(dp, dp['actions'])
        self.assertEqual([action.params['card_id'] for action in filtered],
                         [house.FILLERS[1]])

    def test_human_keeps_die_staged_until_house_and_gandalf_finish(self):
        for kind, figures in [('regular', (4, 1, 1)), ('elite', (3, 2, 1))]:
            for take in (False, True):
                with self.subTest(kind=kind, take=take):
                    driver, path = self.rig(event=True, full_hand=True)
                    char_before = len(driver.es.fp_char_deck)
                    strategy_before = len(driver.es.fp_strat_deck)
                    output = []
                    for line in house_window(kind, event=True)[:-1]:
                        output += self.feed(driver, path, line)
                    self.assertEqual(house.figures(driver.es), figures)
                    self.assertEqual(len(driver.es.hands[house.FP].hand), 7)
                    self.assertEqual(ai.describe_pending(driver.es)['kind'], 'discard')
                    self.assertEqual(len(driver.es.fp_char_deck), char_before)
                    self.assertEqual(len(driver.es.fp_strat_deck), strategy_before - 2)
                    output += self.feed(driver, path, ATHELAS_DISCARD)
                    self.assertEqual(len(driver.es.hands[house.FP].hand), 6)
                    self.assertEqual(driver.es.pending_card_option['card_id'], GANDALF_ID)
                    output += self.feed(driver, path, CHARACTER_DRAW if take else DECLINE)
                    if take:
                        self.assertEqual(len(driver.es.hands[house.FP].hand), 7)
                        self.assertEqual(ai.describe_pending(driver.es)['kind'], 'discard')
                        output += self.feed(driver, path, BILBO_DISCARD)
                    self.assertEqual(len(driver.es.hands[house.FP].hand), 6)
                    self.assertEqual(len(driver.es.fp_char_deck), char_before - int(take))
                    self.assertIsNone(ai.describe_pending(driver.es))
                    self.assertFalse(any('Controls.644' in str(item) for item in output),
                                     'The human still owns the staged die cleanup')
                    self.assert_paid(driver)
                    commits = driver.n_commits_total
                    output += self.feed(driver, path, house_window(event=True)[-1])
                    self.assertEqual(driver.n_commits_total, commits,
                                     'Final cleanup must not replay the House action')
                    self.assertIsNone(driver.seg.window)
                    self.assertIsNone(driver.deferred)
                    self.assertIsNone(driver.pending_buf)
                    self.assertIsNone(driver.chat_deferred)
                    self.assertFalse(any(driver.rand.unconsumed().values()))
                    self.assert_paid(driver)

    def test_full_hand_gandalf_decline_after_first_discard_has_no_bonus(self):
        driver, path, char_before = self.ready_for_gandalf()
        self.feed(driver, path, DECLINE)
        self.assertEqual(len(driver.es.hands[house.FP].hand), 6)
        self.assertEqual(driver.es.fp_char_deck, char_before)
        self.assertIsNone(ai.describe_pending(driver.es))
        self.assertIsNone(driver.rand.pending())
        self.assertFalse(any(driver.rand.unconsumed().values()))
        self.assert_paid(driver)

    def test_staged_house_discard_undo_restores_the_decision(self):
        driver, path = self.rig(event=True, full_hand=True)
        self.feed(driver, path, *house_window(event=True)[:-1])
        self.assertIsNotNone(ai.describe_pending(driver.es))
        self.assertEqual(ai.describe_pending(driver.es)['kind'], 'discard')
        self.feed(driver, path, ATHELAS_DISCARD)
        self.assertEqual(driver.es.pending_card_option['card_id'], GANDALF_ID)
        self.feed(driver, path, '$Wey undoes last action : Wey kills Athelas/Anduril in Free Peoples Hand.')
        self.assertEqual(len(driver.es.hands[house.FP].hand), 7)
        self.assertIn(house.FILLERS[0], driver.es.hands[house.FP].hand)
        self.assertEqual(ai.describe_pending(driver.es)['kind'], 'discard')
        self.assertIsNotNone(driver.seg.window)
        self.feed(driver, path, ATHELAS_DISCARD, DECLINE, house_window(event=True)[-1])
        self.assertEqual(len(driver.es.hands[house.FP].hand), 6)
        self.assertEqual(driver.es.hands[house.FP].discarded_face_down.count(house.FILLERS[0]), 1)
        self.assertIsNone(ai.describe_pending(driver.es))
        self.assertIsNone(driver.seg.window)
        self.assert_paid(driver)

    def test_staged_house_wrong_recruitment_cannot_commit_by_card_identity(self):
        driver, path = self.rig(event=True, full_hand=True)
        starting_deck = list(driver.es.fp_strat_deck)
        starting_hand = list(driver.es.hands[house.FP].hand)
        lines = house_window(event=True)[:-1]
        lines[2] = '$Wey moves Rohan Regular from FP Reinforcements to Minas Tirith.'
        output = self.feed(driver, path, *lines)
        self.assertEqual(house.figures(driver.es), (3, 1, 1))
        self.assertEqual(driver.es.hands[house.FP].hand, starting_hand)
        self.assertEqual(len(driver.es.fp_strat_deck), len(starting_deck))
        reserved_cards = [cid for cid, _pid in
                          driver.rand._feed['cards'][(house.FP.name, 'strategy')]]
        self.assertEqual(len(reserved_cards), 2)
        # Staging starts the legal card and reserves its die before its
        # physical effect. The incorrect recruit must never commit that
        # effect, and an unfinished gesture is not a refusal boundary.
        self.assertTrue(driver.es.dice[house.FP][0].used)
        self.assertEqual(driver.n_commits_total, 1)
        self.assertEqual(ai.describe_pending(driver.es)['kind'], 'card_option')
        self.assertFalse(any(item[0] == 'chat' and item[1] == 'refusal' for item in output))
        close = house_window(event=True)[-1]
        output = self.feed(driver, path, close)
        self.assertTrue(any(item[0] == 'chat' and item[1] == 'refusal' for item in output), output)
        self.assertEqual(driver.n_commits_total, 1)
        self.assertEqual(driver.es.hands[house.FP].hand, starting_hand)
        self.assertEqual(len(driver.es.fp_strat_deck), len(starting_deck))
        # Native Undo restores the discarded die/card. Correct the physical
        # recruitment while preserving both already supplied draw receipts.
        self.feed(driver, path, '$Wey undoes last action : ' + close[1:])
        self.assertIsNotNone(driver.seg.window)
        self.feed(driver, path,
                  '$Wey moves Rohan Regular from Minas Tirith to FP Reinforcements.',
                  house_window(event=True)[2])
        self.assertEqual(house.figures(driver.es), (3, 2, 1))
        self.assertEqual(len(driver.es.hands[house.FP].hand), 7)
        self.assertTrue(all(driver.es.hands[house.FP].hand.count(cid) == 1
                            for cid in reserved_cards))
        self.assertEqual(len(driver.es.fp_strat_deck), len(starting_deck) - 2)
        self.assertEqual(ai.describe_pending(driver.es)['kind'], 'discard')
        self.assertEqual(driver.n_commits_total, 2)
        self.feed(driver, path, ATHELAS_DISCARD, DECLINE, close)
        self.assertIsNone(ai.describe_pending(driver.es))
        self.assertIsNone(driver.seg.window)
        self.assertIsNone(driver.pending_buf)
        self.assertFalse(any(driver.rand.unconsumed().values()))
        self.assert_paid(driver)

    def test_manual_gandalf_draw_is_one_choice_one_draw_then_second_discard(self):
        driver, path, char_before = self.ready_for_gandalf()
        self.feed(driver, path, CHARACTER_DRAW)
        self.assertEqual(len(driver.es.hands[house.FP].hand), 7)
        self.assertEqual(len(driver.es.fp_char_deck), len(char_before) - 1)
        self.assertEqual(ai.describe_pending(driver.es)['kind'], 'discard')
        self.assertIsNone(driver.rand.pending(), 'The same physical draw must not be requested again')
        self.feed(driver, path, BILBO_DISCARD)
        self.assertEqual(len(driver.es.hands[house.FP].hand), 6)
        self.assertNotIn(house.FILLERS[1], driver.es.hands[house.FP].hand)
        self.assertIsNone(ai.describe_pending(driver.es))
        self.assertFalse(any(driver.rand.unconsumed().values()))
        self.assert_paid(driver)

    def test_prospective_optional_gandalf_menu_understands_explicit_words(self):
        # Isolates bridge vocabulary using the agreed literal contract rows.
        menu = [ai.Action(-1, 'card_option', {'card_id': GANDALF_ID,
                'side': 'fp', 'choices': {'gandalf_draw': take}})
                for take in (True, False)]
        self.assertIs(chat.match_choice('draw', menu), menu[0])
        self.assertIs(chat.match_choice('decline', menu), menu[1])
        self.assertIs(chat.match_choice('no draw', menu), menu[1])

    def test_prospective_optional_gandalf_prompt_explains_draw_and_decline(self):
        driver, path = self.rig(state=prospective_gandalf_state())
        output = self.feed(driver, path)
        words = ' '.join(str(item) for item in output).lower()
        self.assertIn('gandalf', words)
        self.assertIn('character', words)
        self.assertIn('decline', words)

    def test_prospective_manual_draw_selects_and_supplies_the_bonus_once(self):
        driver, path = self.rig(state=prospective_gandalf_state())
        deck_before = len(driver.es.fp_char_deck)
        self.feed(driver, path, CHARACTER_DRAW)
        self.assertEqual(len(driver.es.hands[house.FP].hand), 3)
        self.assertEqual(len(driver.es.fp_char_deck), deck_before - 1)
        self.assertIsNone(ai.describe_pending(driver.es))
        self.assertIsNone(driver.rand.pending())
        self.assertFalse(any(driver.rand.unconsumed().values()))
        commits = driver.n_commits_total
        for _ in range(3):
            self.feed(driver, path)
        self.assertEqual(driver.n_commits_total, commits)
        self.assertEqual(len(driver.es.hands[house.FP].hand), 3)
        self.assert_paid(driver)

    def test_wrong_deck_cannot_supply_gandalf_character_bonus(self):
        driver, path = self.rig(state=prospective_gandalf_state())
        deck_before = list(driver.es.fp_strat_deck)
        self.feed(driver, path, STRATEGY_DRAW)
        self.assertEqual(driver.es.hands[house.FP].hand, [house.S1, house.S2])
        self.assertEqual(driver.es.fp_strat_deck, deck_before)
        self.assertEqual(driver.es.pending_card_option['card_id'], GANDALF_ID)

    def test_bonus_draw_is_rejected_until_first_discard_finishes(self):
        driver, path, char_before = self.ready_for_first_discard()
        hand_before = list(driver.es.hands[house.FP].hand)
        output = self.feed(driver, path, CHARACTER_DRAW)
        self.assertEqual(driver.es.hands[house.FP].hand, hand_before)
        self.assertEqual(driver.es.fp_char_deck, char_before)
        self.assertEqual(ai.describe_pending(driver.es)['kind'], 'discard')
        self.assertFalse(any(driver.rand.unconsumed().values()))
        self.assertTrue(any('un-owed' in str(item) for item in output), output)

    def test_chat_take_waits_for_one_physical_draw_without_rechoosing(self):
        driver, path, char_before = self.ready_for_gandalf()
        self.feed(driver, path, '<Wey> draw')
        self.assertEqual(len(driver.es.hands[house.FP].hand), 6)
        self.assertEqual(driver.es.fp_char_deck, char_before)
        self.assertIsNotNone(driver.chat_deferred)
        commits = driver.n_commits_total
        for _ in range(3):
            self.feed(driver, path)
        self.assertEqual(driver.n_commits_total, commits)
        self.feed(driver, path, CHARACTER_DRAW)
        self.assertEqual(driver.n_commits_total, commits + 1)
        self.assertEqual(len(driver.es.hands[house.FP].hand), 7)
        self.assertEqual(len(driver.es.fp_char_deck), len(char_before) - 1)
        self.assertIsNone(driver.chat_deferred)
        self.assertIsNone(driver.rand.pending())
        self.feed(driver, path, BILBO_DISCARD)
        self.assertIsNone(ai.describe_pending(driver.es))
        self.assert_paid(driver)

    def test_other_side_draw_does_not_choose_the_human_bonus(self):
        driver, path = self.rig(state=prospective_gandalf_state())
        original = list(driver.es.hands[house.FP].hand)
        self.feed(driver, path, '$WOTR_AI_Shadow draws Shadow Character Card.')
        self.assertEqual(driver.es.hands[house.FP].hand, original)
        self.assertEqual(driver.es.pending_card_option['card_id'], GANDALF_ID)
        self.assertFalse(any(driver.rand.unconsumed().values()))

    def test_chat_take_wrong_deck_never_feeds_or_settles_the_bonus(self):
        driver, path = self.rig(state=prospective_gandalf_state())
        self.feed(driver, path, '<Wey> draw')
        strategy_before = list(driver.es.fp_strat_deck)
        count_before = dict(driver.turn_draws)
        self.feed(driver, path, STRATEGY_DRAW)
        self.assertEqual(driver.es.fp_strat_deck, strategy_before)
        self.assertEqual(driver.turn_draws, count_before)
        self.assertFalse(any(driver.rand.unconsumed().values()))
        self.assertIsNotNone(driver.chat_deferred)
        self.assertEqual(driver.rand.pending().kind, 'draws')
        self.assertEqual(driver.rand.pending().kw.get('items'), [('FREE_PEOPLES', 'character')])
        self.feed(driver, path, '$Wey undoes last action : Wey draws Free Strategy Card.')
        self.feed(driver, path, CHARACTER_DRAW)
        self.assertEqual(len(driver.es.hands[house.FP].hand), 3)
        self.assertIsNone(ai.describe_pending(driver.es))
        self.assertIsNone(driver.chat_deferred)
        self.assertFalse(any(driver.rand.unconsumed().values()))
        self.assert_paid(driver)

    def test_manual_bonus_draw_has_one_native_undo_entry(self):
        driver, path = self.rig(state=prospective_gandalf_state())
        deck_before = len(driver.es.fp_char_deck)
        self.feed(driver, path, CHARACTER_DRAW)
        self.assertEqual(len(driver.commits), 1)
        self.assertEqual(driver.commits[-1][1], ['Wey draws Free Character Card.'])
        self.feed(driver, path, '$Wey undoes last action : Wey draws Free Character Card.')
        self.assertEqual(len(driver.es.hands[house.FP].hand), 2)
        self.assertEqual(len(driver.es.fp_char_deck), deck_before)
        self.assertEqual(driver.es.pending_card_option['card_id'], GANDALF_ID)
        self.feed(driver, path, DECLINE)
        self.assertIsNone(ai.describe_pending(driver.es))
        self.assert_paid(driver)

    def test_reissued_draw_echo_does_not_choose_a_new_bonus(self):
        driver, path = self.rig(state=prospective_gandalf_state())
        driver._reissued_draws94 = {('fp', 'char'): 1}
        driver._reissued_ts94 = time.time()
        before = list(driver.es.hands[house.FP].hand)
        deck_before = list(driver.es.fp_char_deck)
        self.feed(driver, path, CHARACTER_DRAW)
        self.assertEqual(driver.es.hands[house.FP].hand, before)
        self.assertEqual(driver.es.fp_char_deck, deck_before)
        self.assertEqual(driver.es.pending_card_option['card_id'], GANDALF_ID)
        self.assertEqual(driver._reissued_draws94[('fp', 'char')], 0)
        self.feed(driver, path, CHARACTER_DRAW)
        self.assertEqual(len(driver.es.hands[house.FP].hand), 3)
        self.assertIsNone(ai.describe_pending(driver.es))
        self.assert_paid(driver)

    def test_gandalf_decline_words_do_not_decline_mandatory_palantir_draw(self):
        menu = [ai.Action(-1, 'card_option', {'card_id': '__palantir_draw__',
                'side': 'sh', 'choices': {'deck': deck}})
                for deck in ('character', 'strategy')]
        self.assertIsNone(chat.match_choice('decline', menu))
        self.assertIsNone(chat.match_choice('no draw', menu))

    def test_prospective_ai_bonus_decline_is_announced(self):
        menu = [ai.Action(-1, 'card_option', {'card_id': GANDALF_ID,
                'card_name': 'Gandalf guide draw', 'side': 'fp',
                'choices': {'gandalf_draw': take}}) for take in (True, False)]
        text = chat.declined_leg(menu[1], menu)
        self.assertIsNotNone(text, 'A declined optional draw changes no board piece; it needs a notice')
        self.assertIn('gandalf', text.lower())

    def test_ai_house_full_hand_draw_chain_and_cleanup_are_once(self):
        from live.live_server import LiveServer
        import emitter
        import log_replay
        temp = tempfile.TemporaryDirectory(prefix='wotr-stewards-ai-')
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / 'ai_feed.txt'
        path.write_text('', encoding='utf-8')
        server = LiveServer(temp.name, house.FP, load_nets=False, seed=17)
        driver = server.driver
        driver.es = house.before_play(face=house.gs.ActionDieFace.EVENT,
                                      fillers=house.FILLERS)
        driver.hands.es = driver.es
        driver.phase = 'actions'
        positions = json.loads((ROOT / 'wotr_bridge/initial_positions.json').read_text())
        driver.view.piece_region = {pid: log_replay.decode_location(zone)
                                   for pid, zone in positions.items()}
        # Card PIDs are this fixture's committed identities; human
        # inputs elsewhere are authored text, never these emissions.
        for pid, cid in zip(('130', '131', '132'), [house.CID] + house.FILLERS[:2]):
            driver.view.card_pid_map[cid] = pid
            driver.view.piece_region[pid] = 'Free Peoples Hand'
        driver.view.piece_region['210'] = 'FP Dice'
        all_output = []
        def decide(dp):
            out = server._ai_decide(dp)
            all_output.extend(out)
            return out
        play = ai.Action(0, 'play_event', {'card_id': house.CID, 'side': 'fp'})
        out = decide({'side': house.FP, 'kind': 'action', 'actions': [play]})
        self.assertEqual([s for s in out if s.startswith('!CALL')], ['!CALL play_card 130'])
        self.assertEqual(server._flush_owed_die_kills(), [])
        self.feed(driver, path,
            '$%s moves Free Action Die (P) from FP Dice to Current Action.' % server.nick,
            PLAY.replace('$Wey', '$' + server.nick))
        dp = ai.describe_pending(driver.es)
        self.assertEqual({a.params['choices']['unit_type'] for a in dp['actions']},
                         {'regular', 'elite'})
        out = decide(dp)
        self.assertEqual([s for s in out if s.startswith('!CALL')], ['!CALL draw_fp_strat'])
        self.assertFalse(any(s.startswith('!CALL') for s in decide(dp)))
        for pid, cid in [('145', house.S1), ('146', house.S2)]:
            driver.view.card_pid_map[cid] = pid
            server.rand.feed('card', ('FREE_PEOPLES', 'strategy', cid))
            out = decide(dp)
            self.feed(driver, path, STRATEGY_DRAW.replace('$Wey', '$' + server.nick))
        self.assertEqual(house.figures(driver.es), (3, 2, 1))
        self.assertEqual(len(driver.es.hands[house.FP].hand), 7)
        self.assertEqual(ai.describe_pending(driver.es)['kind'], 'discard')
        self.assertEqual(server._flush_owed_die_kills(), [])
        self.feed(driver, path,
            '$%s moves Gondor Elite from FP Reinforcements to Minas Tirith.' % server.nick)
        discard, = [a for a in ai.get_legal_actions(driver.es, house.FP)
                    if a.params.get('card_id') == house.FILLERS[0]]
        decide({'side': house.FP, 'kind': 'discard', 'actions': [discard]})
        self.feed(driver, path, ATHELAS_DISCARD.replace('$Wey', '$' + server.nick))
        self.assertEqual(len(driver.es.hands[house.FP].hand), 6)
        self.assertEqual(driver.es.pending_card_option['card_id'], GANDALF_ID)
        self.assertEqual(server._flush_owed_die_kills(), [])
        dp = ai.describe_pending(driver.es)
        out = decide(dp)
        self.assertEqual([s for s in out if s.startswith('!CALL')], ['!CALL draw_fp_char'])
        self.assertFalse(any(s.startswith('!CALL') for s in decide(dp)))
        self.assertEqual(server._flush_owed_die_kills(), [])
        driver.view.card_pid_map[house.BONUS] = '147'
        server.rand.feed('card', ('FREE_PEOPLES', 'character', house.BONUS))
        decide(dp)
        self.feed(driver, path, CHARACTER_DRAW.replace('$Wey', '$' + server.nick))
        self.assertEqual(len(driver.es.hands[house.FP].hand), 7)
        self.assertEqual(server._flush_owed_die_kills(), [])
        discard, = [a for a in ai.get_legal_actions(driver.es, house.FP)
                    if a.params.get('card_id') == house.FILLERS[1]]
        decide({'side': house.FP, 'kind': 'discard', 'actions': [discard]})
        self.feed(driver, path, BILBO_DISCARD.replace('$Wey', '$' + server.nick))
        self.assertEqual(len(driver.es.hands[house.FP].hand), 6)
        self.assertIsNone(ai.describe_pending(driver.es))
        self.assertEqual([s for s in all_output if s.startswith('!CALL draw_')],
                         ['!CALL draw_fp_strat', '!CALL draw_fp_strat', '!CALL draw_fp_char'])
        cleanup = server._flush_owed_die_kills()
        self.assertEqual(len(cleanup), 1, cleanup)
        op, fields = emitter.classify_line(emitter.strip_tail(cleanup[0]))
        self.assertEqual(op, 'kill')
        self.assertEqual(fields['pieces'], ['210', '130'])
        self.assertEqual(server._flush_owed_die_kills(), [])
        commits = driver.n_commits_total
        self.feed(driver, path, house_window(event=True)[-1].replace('$Wey', '$' + server.nick))
        self.assertEqual(driver.n_commits_total, commits)
        self.assertFalse(any(server.rand.unconsumed().values()))
        self.assert_paid(driver)


if __name__ == '__main__':
    unittest.main(verbosity=2)
