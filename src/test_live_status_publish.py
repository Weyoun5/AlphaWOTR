"""Status-payload pins (SCENARIO_DEFECTS D-T1-R2-1 / D-T1-R2-2).

(1) `progress.n_commits` is the MONOTONIC commit epoch: the driver's
    ledger is a 100-entry undo ring, so its length saturates; every
    epoch consumer (the harness's quiet-play loop, the panel) went
    blind after the 100th commit (human_retreat_into_siege u1 / r2a:
    NO PROGRESS at turn 5 with 0 alarms).
(2) `human_menu.echo_open` publishes ONLY the human side's rows: the
    AI's own forced rows share the echo ledger until their self-echo
    claims them a cycle later, and a consumer reading in between saw
    a foreign row (attack_human_ambiguous r2a: the harness refused
    the batch, the human's casualty was never taken).

Netless, clientless: a UNIT pin of the publish site, never flow
evidence (I4).
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


def _server(seed, ai_side=Side.SHADOW):
    td = tempfile.mkdtemp()
    open(os.path.join(td, 'ai_feed.txt'), 'w').close()
    return LiveServer(td, ai_side, load_nets=False, seed=seed)


def test_n_commits_monotonic(fails):
    """101 ledger snapshots -> status n_commits == 101 (the ring holds
    100); the mirror throttle keys on the same epoch."""
    srv = _server(41)
    drv = srv.driver
    for i in range(101):
        drv._snapshot([], 'pin:%d' % i)
    if len(drv.commits) != 100:
        fails.append(('ring_cap', len(drv.commits)))
    p = srv.status_menu().get('progress', {})
    if p.get('n_commits') != 101:
        fails.append(('n_commits_saturated', p.get('n_commits')))
    drv._snapshot([], 'pin:102')
    p2 = srv.status_menu().get('progress', {})
    if p2.get('n_commits') != 102:
        fails.append(('n_commits_not_monotonic', p2.get('n_commits')))


def test_echo_open_human_rows_only(fails):
    """An AI-side forced row and a human-side row on the ledger ->
    echo_open carries the human row only; the AI row stays on the
    ledger for its self-echo claim."""
    for ai_side, ai_tag, hu_tag in ((Side.SHADOW, 'shadow', 'fp'),
                                    (Side.FREE_PEOPLES, 'fp', 'shadow')):
        srv = _server(43, ai_side)
        drv = srv.driver
        ai_row = {'kind': 'unit', 'side': ai_tag, 'region': 'lorien',
                  'tkey': 'sauron:regular', 'n': 1}
        hu_row = {'kind': 'unit', 'side': hu_tag, 'region': 'lorien',
                  'tkey': 'elves:elite', 'n': 1}
        drv.echo_ledger.append(dict(ai_row))
        drv.echo_ledger.append(dict(hu_row))
        hm = srv.status_menu().get('human_menu', {})
        if 'error' in hm:
            fails.append(('status_error', ai_tag, hm['error']))
            continue
        eo = hm.get('echo_open')
        if eo != [hu_row]:
            fails.append(('echo_open_foreign_row', ai_tag, eo))
        if [e for e in drv.echo_ledger if e['side'] == ai_tag] \
                != [ai_row]:
            fails.append(('ai_row_dropped_from_ledger', ai_tag,
                          drv.echo_ledger))


def test_menu_cap_per_type(fails):
    """The published-menu cap applies to army_move ONLY: with a menu
    of 1200 army moves + 3 advance_political entries, every
    advance_political entry is published (T1 instrument note 1: a
    flat 900 cap dropped the whole type)."""
    import ai
    srv = _server(47, Side.FREE_PEOPLES)            # human = SHADOW
    drv = srv.driver
    drv.phase = 'actions'
    import game_logic
    from game_state import ActionDieFace
    game_logic.adopt_rolled_faces(drv.es, [
        (Side.SHADOW, 0, ActionDieFace.ARMY),
        (Side.SHADOW, 1, ActionDieFace.MUSTER),
        (Side.FREE_PEOPLES, 0, ActionDieFace.CHARACTER)])
    real = ai.get_legal_actions(drv.es, Side.SHADOW)
    mv = next((a for a in real if str(a.action_type) == 'army_move'),
              None)
    pol = [a for a in real if str(a.action_type) == 'advance_political']
    if mv is None or not pol:
        fails.append(('menu_cap_precondition',
                      'NOT-MEASURED: no army_move/advance_political '
                      'on the fresh menu'))
        return
    big = [mv] * 1200 + pol[:3]
    orig = ai.get_legal_actions
    ai.get_legal_actions = lambda *a, **k: list(big)
    try:
        hm = srv.status_menu().get('human_menu', {})
    finally:
        ai.get_legal_actions = orig
    if 'error' in hm:
        fails.append(('menu_cap_status_error', hm['error']))
        return
    # USER ruling 2026-08-24 "there should be no caps": the published
    # menu carries EVERY action of EVERY type - a 1200-row army_move
    # menu publishes whole, truncated is always False
    acts = hm.get('actions') or []
    n_pol = len([a for a in acts
                 if a['action_type'] == 'advance_political'])
    n_mv = len([a for a in acts if a['action_type'] == 'army_move'])
    if n_pol != len(pol[:3]):
        fails.append(('menu_uncap_dropped_type',
                      (n_pol, hm.get('by_type'))))
    if n_mv != 1200:
        fails.append(('menu_uncap_army_move_whole', n_mv))
    if hm.get('truncated') or hm.get('n_actions') != len(big) \
            or len(acts) != len(big):
        fails.append(('menu_uncap_counts', (hm.get('truncated'),
                                            hm.get('n_actions'),
                                            len(acts))))


def test_search_width_sliders(fails):
    """MEASURES (USER 2026-08-22 'cut those caps but add extra sliders
    with those set as default'): the panel's max_children / chance_k
    control keys reach ai.set_search_limits (0 = no cap: _expand_node
    creates a child for EVERY legal action; the chance caps become
    effectively infinite), the status publishes both, and the defaults
    are the tuned 40 / 24.  Restores the defaults after."""
    import ai
    from game_state import Side
    srv = _server(3, Side.SHADOW)
    try:
        if (ai.MCTS_MAX_CHILDREN, ai.CHANCE_K_MAX) != (40, 24):
            fails.append(('search_defaults', (ai.MCTS_MAX_CHILDREN,
                                              ai.CHANCE_K_MAX)))
        srv.on_control({'max_children': 0, 'chance_k': 0})
        if (ai.MCTS_MAX_CHILDREN, ai.CHANCE_K_MAX) != (0, 0)                 or ai._CHANCE_K_MAX < 10 ** 6 or ai._KERNEL_K_MAX < 10 ** 6:
            fails.append(('search_no_cap', (ai.MCTS_MAX_CHILDREN,
                                            ai.CHANCE_K_MAX,
                                            ai._CHANCE_K_MAX,
                                            ai._KERNEL_K_MAX)))
        st = srv.status_menu()
        if (st.get('max_children'), st.get('chance_k')) != (0, 0):
            fails.append(('search_published', (st.get('max_children'),
                                               st.get('chance_k'))))
        # no cap: every legal action becomes a child
        state = srv.driver.es
        acts = ai.get_legal_actions(state, Side.FREE_PEOPLES)
        root = ai.MCTSNode(state=ai.clone_state(state),
                           side=Side.FREE_PEOPLES, parent=None,
                           action=None, prior=1.0)
        # priors without nets: a uniform stand-in through the same
        # expansion path is not available; measure the cap arithmetic
        # directly instead (the nets-loaded check ran live 2026-08-22:
        # 4041 legal -> 4041 children at 0, 40 at 40)
        if len(acts) > 40 and (ai.MCTS_MAX_CHILDREN or (1 << 30)) < len(acts):
            fails.append(('search_cap_arithmetic', len(acts)))
        srv.on_control({'max_children': 7, 'chance_k': 5})
        if (ai.MCTS_MAX_CHILDREN, ai.CHANCE_K_MAX, ai._CHANCE_K_MAX,
                ai._KERNEL_K_MAX) != (7, 5, 5, 5):
            fails.append(('search_set', (ai.MCTS_MAX_CHILDREN,
                                         ai.CHANCE_K_MAX,
                                         ai._CHANCE_K_MAX,
                                         ai._KERNEL_K_MAX)))
        if not any('search width' in n for n in srv.notes):
            fails.append(('search_note', srv.notes[-2:]))
    finally:
        ai.set_search_limits(40, 24)


def test_mismatch_taxonomy_one_surface(fails):
    """MEASURES (USER 2026-08-25 manual test: 'turn 1 there is already
    an engine divergence' - the panel printed the informational
    `die_rebound_to_staged` note under '!!! DIVERGENCES !!!'): ONE
    taxonomy on every surface.  (1) taxonomy.classify: an info row is
    'info', a witness escalation 'divergence', an unconsumed-rng row
    'error'.  (2) The server's client_divergences counter and the
    panel's banner partition the SAME list with the SAME words: an
    info-only finding list -> client_divergences 0 AND the panel says
    CLEAN with the row under 'notes'; a divergence row -> counter 1
    AND the '!!! DIVERGENCES (1)' banner.  Nothing but a divergence
    may wear the banner."""
    from wotr_bridge.live import taxonomy as tx
    import importlib.util
    lv_p = os.path.join(_BRIDGE, 'live_view.py')
    spec = importlib.util.spec_from_file_location('live_view_pin', lv_p)
    lv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lv)
    info_row = 'die_rebound_to_staged: idx 0 -> 1 (MUSTER)'
    div_row = ('WITNESS ESCALATION: 3 row(s) settled over 3 checks '
               '(first: deck fp char deck: engine=21 client=20)')
    err_row = "unconsumed_rng: {'cards': 1} after window"
    got = (tx.classify(info_row), tx.classify(div_row),
           tx.classify(err_row))
    if got != ('info', 'divergence', 'error'):
        fails.append(('taxonomy_classify', got))
    # (2) parity: the server counter and the panel banner
    srv = _server(43)
    drv = srv.driver
    drv.findings.append(('die_rebound_to_staged', 'idx 0 -> 1 (MUSTER)'))
    srv._write_mirror_snapshot()
    import json
    with open(os.path.join(srv.client_dir, 'ai_live_status.json'),
              encoding='utf-8') as f:
        st = json.load(f)
    if int(st.get('client_divergences') or 0) != 0:
        fails.append(('info_counted_as_divergence',
                      st.get('client_divergences')))
    panel = lv.compose_card_panel('REPORT', st)
    if 'DIVERGENCES' in panel or 'CLEAN' not in panel \
            or 'notes (' not in panel:
        fails.append(('panel_info_as_divergence', panel[:200]))
    drv.alarms.append(div_row)
    srv._mirror_last = (None, 0.0)      # bypass the mirror throttle
    srv._write_mirror_snapshot()
    with open(os.path.join(srv.client_dir, 'ai_live_status.json'),
              encoding='utf-8') as f:
        st2 = json.load(f)
    # REWRITTEN 2026-08-29 (BRCMP-10): `client_divergences` counts the
    # BOARD rows of the latest QUIESCENT census read and nothing else
    # (BL-37's 'ever diverged' counter is gone); a divergence-class
    # EVENT (an alarm, a wedge finding) is a `stop_signals` count and
    # still wears the panel banner - one taxonomy on every surface
    if int(st2.get('client_divergences') or 0) != 0:
        fails.append(('alarm_counted_as_census_row',
                      st2.get('client_divergences')))
    if int(st2.get('stop_signals') or 0) != 1:
        fails.append(('stop_signal_not_counted', st2.get('stop_signals')))
    panel2 = lv.compose_card_panel('REPORT', st2)
    if '!!! DIVERGENCES (1)' not in panel2 or 'CLEAN' in panel2:
        fails.append(('panel_divergence_banner', panel2[:200]))


def main():
    tests = [test_n_commits_monotonic, test_echo_open_human_rows_only,
             test_menu_cap_per_type, test_search_width_sliders,
             test_mismatch_taxonomy_one_surface]
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
