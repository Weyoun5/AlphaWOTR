"""SCENARIO_DEFECTS D5 pins (real-client reproductions
`probe_cc_refusal_p1` / `probe_cc_refusal_u_red1` and the other-seat
`ai_attacks_human_defends_p1claim`): a HUMAN combat card whose text is
not playable in the battle is refused by the engine's reveal.  USER E4
+ rule 5 + K1 step 3: the card RETURNS TO HAND - the hidden commit is
withdrawn (ai.cancel_hidden_combat_card re-opens that side's window,
the other side's commit stands), the client's reveal is undone by ONE
client-native press, the human chooses again or says `nc`, the
re-commit owes a FRESH reveal directive, and the AI attacker's owed
battle die is NEVER killed while the battle is open.  A human mistake
is a finding + notice, never an alarm.

Composed stack (driver + randomness + emission + chat), netless
(scripted picks), clientless: a UNIT pin of our own protocol - the
real-client scenario is the evidence (I4).
"""
import os
import sys
import tempfile

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
_BRIDGE = os.path.join(_ROOT, 'wotr_bridge')
for _p in (_SRC, _ROOT, _BRIDGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from game_state import Side

from wotr_bridge.live.live_server import LiveServer


def _append(path, *lines):
    with open(path, 'a', encoding='utf-8') as f:
        for ln in lines:
            f.write(ln + '\n')


def _title(cid):
    from cards import get_card
    c = get_card(cid)
    return '%s/%s' % (c.card_name, c.combat_name)


def _has_undo(out, nick):
    return any(isinstance(ln, str) and ln.startswith('$' + nick)
               and '<~Controls.819~>' in ln for ln in out)


def _has_die_kill(out):
    return any(isinstance(ln, str) and '<~Controls.644~>' in ln
               for ln in out)


def _battle_cards(cur, at, human, deck_bad, deck_good, ai_deck=None):
    """Seed the hands for the pin: an UNPLAYABLE `deck_bad` card and a
    playable `deck_good` card for the human side, optionally a
    playable card for the AI side.  Returns (bad, good, ai_cid)."""
    import ai
    import test_live_engine_fixwave as TF
    probe = ai.apply_action(cur, at)
    pc = probe.pending_combat
    is_atk = (human is Side.SHADOW)
    bad = TF._unplayable_card_for(cur, human, pc, is_atk, deck_bad)
    good = TF._hand_card_for(cur, human, pc, is_atk, deck_good)
    ai_cid = None
    if ai_deck is not None:
        other = Side.FREE_PEOPLES if human is Side.SHADOW else Side.SHADOW
        ai_cid = TF._hand_card_for(cur, other, pc, not is_atk, ai_deck)
    return bad, good, ai_cid


def test_human_attacker_refused_card_returns_to_hand(fails):
    """MEASURES (probe_cc_refusal_u_red1, human SA attacker / AI FP
    defender): the unplayable reveal -> ONE undo press, no alarm, the
    engine back in the attacker's card window with the defender's
    `nc` standing; the undo echo is bookkeeping; a second hidden
    choose owes a fresh reveal_chosen; the playable reveal resumes the
    round (starving on the client's rolls)."""
    import ai
    import test_live_emission as TE
    cur, at = TE._drive_to_battle()
    if at is None:
        fails.append(('cc_ref_control', 'NOT-MEASURED: no attack'))
        return
    bad, good, _ = _battle_cards(cur, at, Side.SHADOW, 'character',
                                 'strategy')
    if bad is None or good is None:
        fails.append(('cc_ref_control', 'NOT-MEASURED: cards bad=%r '
                      'good=%r' % (bad, good)))
        return
    td = tempfile.mkdtemp()
    feed = os.path.join(td, 'ai_feed.txt')
    open(feed, 'w').close()
    srv = LiveServer(td, Side.FREE_PEOPLES, load_nets=False, seed=31)
    drv = srv.driver
    drv.es = cur
    drv.hands.es = cur
    drv.phase = 'actions'
    drv.last_actor = None
    drv.nick_side = {'Wey': None}

    def chooser(st, sd, fp, sh, num_simulations=0, temperature=0.0,
                add_noise=False):
        d = ai.describe_pending(st, sd)
        acts = d['actions']
        nc = next((x for x in acts
                   if x.action_type == 'combat_card_none'), None)
        return (nc or acts[0]), None
    orig = ai.mcts_action_with_policy
    ai.mcts_action_with_policy = chooser
    srv.fp_net = srv.sh_net = 'SCRIPT'
    nick = srv.nick
    try:
        _append(feed, '$Wey moves Shadow Action Die (A) from SA Dice '
                      'to Current Action.')
        srv.handle({'lines': ()})
        _append(feed, '<Wey> fords of isen')
        srv.handle({'lines': ()})
        _append(feed, '$Wey chooses a card.[Shadow Character Card]')
        out = []
        for _ in range(3):
            out += srv.handle({'lines': ()})
        if not any('reveal_chosen' in ln for ln in out):
            fails.append(('cc_ref_control', 'NOT-MEASURED: no reveal '
                          'directive after both commits: %r' % (out,)))
            return
        # the refused reveal
        rev_bad = ('<auto> moves %s from Shadow Hand (chosen) to '
                   'Current SA Card.' % _title(bad))
        _append(feed, rev_bad)
        out = srv.handle({'lines': ()})
        pc = drv.es.pending_combat
        if not _has_undo(out, nick):
            fails.append(('cc_ref_undo', 'no client-native undo press '
                          'after the refusal: %r' % (out,)))
        if drv.alarms:
            fails.append(('cc_ref_alarm', 'a human mistake raised an '
                          'alarm: %r' % (drv.alarms,)))
        if not isinstance(pc, dict) \
                or pc.get('phase') != 'attacker_combat_card' \
                or pc.get('deciding_side') is not Side.SHADOW:
            fails.append(('cc_ref_window', 'the attacker window did '
                          'not re-open: %r' % (
                              (pc or {}).get('phase'),)))
            return
        if 'atk_hidden_cc' in pc or 'atk_combat_card' in pc:
            fails.append(('cc_ref_hidden', 'hidden commit survived '
                          'the refusal: %r' % (
                              {k: pc.get(k) for k in
                               ('atk_hidden_cc', 'atk_combat_card')},)))
        if 'def_combat_card' not in pc:
            fails.append(('cc_ref_other', "the defender's nc did not "
                          'stand'))
        if drv.human_cc_committed:
            fails.append(('cc_ref_flag', 'human_cc_committed still set'))
        if bad not in drv.es.hands[Side.SHADOW].hand:
            fails.append(('cc_ref_hand', 'the refused card left the '
                          'engine hand'))
        # the undo echo: bookkeeping
        _append(feed, '$%s undoes last action : %s' % (nick, rev_bad))
        out = srv.handle({'lines': ()})
        if drv.alarms or drv.uncommitted:
            fails.append(('cc_ref_echo', 'undo echo not consumed: '
                          'alarms=%r uncommitted=%r'
                          % (drv.alarms, drv.uncommitted)))
        # choose again: a FRESH reveal directive is owed
        _append(feed, '$Wey chooses a card.[Shadow Strategy Card]')
        out = []
        for _ in range(2):
            out += srv.handle({'lines': ()})
        if not any('reveal_chosen' in ln for ln in out):
            fails.append(('cc_ref_re_reveal', 'the re-commit got no '
                          'reveal directive (once-per-round dedup '
                          'swallowed it): %r' % (out,)))
        if drv.alarms:
            fails.append(('cc_ref_alarm2', repr(drv.alarms)))
        # the playable reveal resumes the round
        _append(feed, '<auto> moves %s from Shadow Hand (chosen) to '
                      'Current SA Card.' % _title(good))
        out = srv.handle({'lines': ()})
        resumed = (drv.chat_deferred is not None
                   or drv.es.pending_combat is None
                   or int(drv.es.pending_combat.get('rounds_fought', 0)
                          or 0) > 0
                   or any('roll' in str(ln).lower() for ln in out))
        if not resumed:
            fails.append(('cc_ref_resume', 'the playable reveal did '
                          'not resume the round: pc=%r out=%r'
                          % ((drv.es.pending_combat or {}).get('phase'),
                             out)))
        if drv.alarms:
            fails.append(('cc_ref_alarm3', repr(drv.alarms)))
    finally:
        ai.mcts_action_with_policy = orig


def test_ai_attacker_refusal_keeps_die_and_reshows_own_card(fails):
    """MEASURES (ai_attacks_human_defends_p1claim, AI SA attacker with
    its own card committed / human FP defender): the human's unplayable
    reveal -> undo press, the AI's OWED BATTLE DIE is NOT killed (the
    battle is open), the defender window re-opens with the attacker's
    card standing; the client's one press also retracts the AI's own
    reveal (adjacent '<auto>' lines) - the AI re-shows it with its own
    play_card; the human's re-commit owes a fresh reveal."""
    import ai
    import test_live_emission as TE
    cur, at = TE._drive_to_battle()
    if at is None:
        fails.append(('cc_ai_control', 'NOT-MEASURED: no attack'))
        return
    bad, good, ai_cid = _battle_cards(cur, at, Side.FREE_PEOPLES,
                                      'character', 'character',
                                      ai_deck='strategy')
    if bad is None or good is None or ai_cid is None:
        fails.append(('cc_ai_control', 'NOT-MEASURED: cards bad=%r '
                      'good=%r ai=%r' % (bad, good, ai_cid)))
        return
    state = ai.apply_action(cur, at)
    td = tempfile.mkdtemp()
    feed = os.path.join(td, 'ai_feed.txt')
    open(feed, 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=31)
    drv = srv.driver
    drv.es = state
    drv.hands.es = state
    drv.phase = 'actions'
    drv.last_actor = Side.FREE_PEOPLES
    drv.nick_side = {'Wey': None}
    drv.view.card_pid_map[ai_cid] = '183'
    drv.owed_die_kills.append(('SHADOW', int(at.die_idx), None, None))

    def chooser(st, sd, fp, sh, num_simulations=0, temperature=0.0,
                add_noise=False):
        d = ai.describe_pending(st, sd)
        acts = d['actions']
        ccp = next((x for x in acts
                    if x.action_type == 'combat_card_play'
                    and x.params.get('card_id') == ai_cid), None)
        return (ccp or acts[0]), None
    orig = ai.mcts_action_with_policy
    ai.mcts_action_with_policy = chooser
    srv.fp_net = srv.sh_net = 'SCRIPT'
    nick = srv.nick
    try:
        out = srv.handle({'lines': ()})         # the AI commits its card
        if not any('<~Controls.840~>183' in ln for ln in out):
            fails.append(('cc_ai_control', 'NOT-MEASURED: the AI '
                          'attacker did not commit its card: %r'
                          % (out,)))
            return
        _append(feed, '$%s chooses a card.[Shadow Strategy Card]' % nick)
        srv.handle({'lines': ()})
        _append(feed, '$Wey chooses a card.[Free Character Card]')
        out = []
        for _ in range(2):
            out += srv.handle({'lines': ()})
        if not any('reveal_chosen' in ln for ln in out):
            fails.append(('cc_ai_control', 'NOT-MEASURED: no reveal '
                          'directive: %r' % (out,)))
            return
        rev_bad = ('<auto> moves %s from Free Peoples Hand (chosen) to '
                   'Current FP Card.' % _title(bad))
        rev_ai = ('<auto> moves %s from Shadow Hand (chosen) to '
                  'Current SA Card.' % _title(ai_cid))
        _append(feed, rev_bad, rev_ai)
        out = srv.handle({'lines': ()})
        if not _has_undo(out, nick):
            fails.append(('cc_ai_undo', 'no undo press: %r' % (out,)))
        if _has_die_kill(out) or not drv.owed_die_kills:
            fails.append(('cc_ai_die', 'the owed battle die was killed '
                          'with the battle open: out=%r owed=%r'
                          % (out, drv.owed_die_kills)))
        pc = drv.es.pending_combat
        if not isinstance(pc, dict) \
                or pc.get('phase') != 'defender_combat_card' \
                or pc.get('deciding_side') is not Side.FREE_PEOPLES:
            fails.append(('cc_ai_window', 'defender window not '
                          're-opened: %r' % ((pc or {}).get('phase'),)))
            return
        if pc.get('atk_combat_card') != ai_cid:
            fails.append(('cc_ai_other', "the attacker's card did not "
                          'stand: %r' % (pc.get('atk_combat_card'),)))
        if drv.alarms:
            fails.append(('cc_ai_alarm', repr(drv.alarms)))
        # the client's one press folds both '<auto>' reveals (newest
        # first): the AI re-shows its own card
        _append(feed, '$%s undoes last action : %s' % (nick, rev_ai),
                '$%s undoes last action : %s' % (nick, rev_bad))
        out = srv.handle({'lines': ()})
        if not any('play_card 183' in str(ln) for ln in out):
            fails.append(('cc_ai_reshow', "the AI's retracted reveal "
                          'was not re-shown: %r' % (out,)))
        if drv.alarms or drv.uncommitted:
            fails.append(('cc_ai_echo', 'alarms=%r uncommitted=%r'
                          % (drv.alarms, drv.uncommitted)))
        # the re-show echo (choose + reveal of the AI's own card)
        _append(feed, '$%s chooses a card.[Shadow Strategy Card]' % nick,
                rev_ai)
        srv.handle({'lines': ()})
        if drv.alarms:
            fails.append(('cc_ai_reshow_echo', repr(drv.alarms)))
        if drv.es.pending_combat.get('atk_combat_card') != ai_cid:
            fails.append(('cc_ai_reshow_engine', 'the re-show echo '
                          'changed the engine commit'))
        # the human re-commits: a fresh reveal is owed
        _append(feed, '$Wey chooses a card.[Free Character Card]')
        out = []
        for _ in range(2):
            out += srv.handle({'lines': ()})
        if not any('reveal_chosen' in ln for ln in out):
            fails.append(('cc_ai_re_reveal', 'no fresh reveal for the '
                          're-commit: %r' % (out,)))
        _append(feed, '<auto> moves %s from Free Peoples Hand (chosen) '
                      'to Current FP Card.' % _title(good))
        out = srv.handle({'lines': ()})
        pc = drv.es.pending_combat
        resumed = (drv.chat_deferred is not None
                   or pc is None
                   or int((pc or {}).get('rounds_fought', 0) or 0) > 0
                   or srv._open_step is not None
                   or any('roll' in str(ln).lower() for ln in out))
        if not resumed:
            fails.append(('cc_ai_resume', 'the playable reveal did not '
                          'resume the round: %r' % (
                              (pc or {}).get('phase'),)))
        if drv.alarms:
            fails.append(('cc_ai_alarm2', repr(drv.alarms)))
    finally:
        ai.mcts_action_with_policy = orig


def main():
    tests = [test_human_attacker_refused_card_returns_to_hand,
             test_ai_attacker_refusal_keeps_die_and_reshows_own_card]
    fails = []
    for t in tests:
        n0 = len(fails)
        try:
            t(fails)
        except Exception as e:
            import traceback
            traceback.print_exc()
            fails.append((t.__name__, 'EXCEPTION %r' % (e,)))
        print('  [%s] %s' % ('PASS' if len(fails) == n0 else 'FAIL',
                             t.__name__))
    print('=' * 60)
    print('TOTAL %d  FAIL %d' % (len(tests), len(fails)))
    for f in fails:
        print('  FAIL:', f)
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
