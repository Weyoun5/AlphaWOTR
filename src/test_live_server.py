"""LiveServer composition smoke: the D2/D3/D4/G1 turn-start walked
end-to-end through the composed stack (driver + randomness + emission
+ chat policy), netless (menu-order picks) and clientless.

This is a UNIT instrument of our own step protocol — front-end
evidence comes ONLY from Phase-3 real-client legs (I4); nothing here
claims client agreement.
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


def test_turnstart_smoke(fails):
    """MEASURES: human draws -> AI draw directives (once) -> fed cids
    commit exactly two cards -> 'fp ok' -> AI eye step commits ->
    roll_action_dice ask goes out -> the human's letters roll adopts
    and the phase reaches 'actions'."""
    td = tempfile.mkdtemp()
    feed = os.path.join(td, 'ai_feed.txt')
    open(feed, 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=99)
    drv = srv.driver
    _append(feed, '$Wey draws Free Character Card.',
            '$Wey draws Free Strategy Card.')
    out1 = srv.handle({'lines': ()})
    if [s for s in out1 if 'draw_sa_char' in s] != \
            ['!CALL draw_sa_char']:
        fails.append(('ai_draw_call_1', repr(out1)))
        return
    if drv.phase != 'declare':
        fails.append(('phase_declare', drv.phase))
    n0 = len(drv.es.hands[Side.SHADOW].hand)
    # a retry tick with nothing fed must NOT re-order the client
    out_r = srv.handle({'lines': ()})
    if any('draw_sa' in s for s in out_r):
        fails.append(('draw_dedup', repr(out_r)))
    # per-draw deterministic retry: char fed -> the STRATEGY ask
    srv.rand.feed('card', ('SHADOW', 'character',
                           drv.es.sh_char_deck[-1]))
    out1b = srv.handle({'lines': ()})
    if not any('draw_sa_strat' in s for s in out1b):
        fails.append(('ai_draw_call_2', repr(out1b)))
        return
    srv.rand.feed('card', ('SHADOW', 'strategy',
                           drv.es.sh_strat_deck[-1]))
    out2 = srv.handle({'lines': ()})
    if len(srv.driver.es.hands[Side.SHADOW].hand) != n0 + 2:
        fails.append(('ai_draws_commit',
                      '%d -> %d' % (n0,
                                    len(srv.driver.es.hands[
                                        Side.SHADOW].hand))))
    if srv._open_step is not None:
        fails.append(('draws_step_closed', repr(srv._open_step)))
    _append(feed, '<Wey> fp ok')
    out3 = srv.handle({'lines': ()})
    # the FULL zone code is load-bearing: AiOpp.findZone matches
    # '<~Game.2043~>'; a bare '2043' made the roll !CALL a silent
    # no-op on the real client (boot3 leg, 2026-08-19)
    if not any('roll_action_dice Game.2043' in s for s in out3):
        fails.append(('roll_ask', repr(out3)))
    if srv.driver.phase != 'roll':
        fails.append(('phase_roll', srv.driver.phase))
    hb0 = len(srv.driver.es.hunt_box)
    if hb0 + sum(1 for d in srv.driver.es.dice[Side.SHADOW]
                 if d.used) == 0:
        # the eye step committed either an allocation or a 0e; both
        # are legal — record which for the log
        pass
    # the AI's OWN (Shadow) pool rolls first, via its !CALL.  NOTE:
    # this leg used to stop here and assert phase == 'actions' - it
    # ENCODED the turn-order defect (USER 2026-08-20) by declaring the
    # action phase open with the human's pool still unrolled.  Action
    # resolution opens only after BOTH pools roll.
    unset = [d for d in srv.driver.es.dice[Side.SHADOW]
             if not d.used and d.current is None]
    letters = ['E'] + ['A'] * (len(unset) - 1)
    _append(feed, '$Wey rolls %s .' % ' '.join(letters))
    srv.handle({'lines': ()})
    if srv.driver.phase != 'roll':
        fails.append(('phase_still_roll', srv.driver.phase))
    # ...then the HUMAN's own FP pool
    fp_unset = [d for d in srv.driver.es.dice[Side.FREE_PEOPLES]
                if not d.used and d.current is None]
    _append(feed, '$Wey rolls %s .' % ' '.join(['A'] * len(fp_unset)))
    srv.handle({'lines': ()})
    if srv.driver.phase != 'actions':
        fails.append(('phase_actions', srv.driver.phase))
    if len(srv.driver.es.hunt_box) <= hb0:
        fails.append(('rolled_eye_hunt', 'no eye entered the box'))
    print('  [smoke] eyes_allocated=%d hunt_box=%d notes=%d'
          % (getattr(srv.driver.es, 'eyes_allocated', 0),
             len(srv.driver.es.hunt_box), len(srv.notes)))


def test_transport_seam(fails):
    """MEASURES: the composed FileBridge -> LiveServer.handle seam,
    wired exactly as main() wires it (build()): a REQ file written the
    way AiOpp writes it (LIVE_SHADOW header + raw log lines) must
    produce a RESP file carrying the AI draw directive.

    REPRODUCTION (Phase 3, 2026-08-19, scratchpad repro_seam.py against
    the unchanged module): main() built FileBridge without owner_needle
    (TypeError on entry) and handle() read request.get() on the
    transport's Request namedtuple (AttributeError swallowed by
    serve()'s catch-all on EVERY cycle -> no RESP ever written; the
    composed boot would tick forever).  This leg is the mechanized
    reproduction: it goes RED with either defect restored."""
    import tempfile as _tf
    from wotr_bridge.live.live_server import build
    td = _tf.mkdtemp()
    with open(os.path.join(td, 'ai_feed.txt'), 'w',
              encoding='utf-8') as f:
        f.write('$Wey draws Free Character Card.\n'
                '$Wey draws Free Strategy Card.\n')
    srv, bridge = build(td, 'SA', load_nets=False, seed=7)
    # the request file exactly as AiOpp.requestMove writes it
    with open(os.path.join(td, 'ai_bridge_request.log'), 'w',
              encoding='utf-8') as f:
        f.write('LIVE_SHADOW\n$Wey<~Controls.1~>setup noise line\n')
    try:
        bridge.serve(srv.handle, max_cycles=3)
    finally:
        bridge.release_lock()
    resp = os.path.join(td, 'ai_bridge_move.txt')
    if not os.path.exists(resp):
        fails.append(('seam_resp_missing',
                      'no RESP file written by the composed seam'))
        return
    txt = open(resp, encoding='utf-8').read()
    if '!CALL draw_sa_char' not in txt:
        fails.append(('seam_resp_content', repr(txt[:200])))


_PAIR_RAW = os.path.join(
    os.path.dirname(_ROOT), 'WOTR_AI_training_lab', 'audit', 'pairs',
    '2023-01-07_230908_FP_Franatleti_SP_GJB.raw.log')


def test_ai_draw_decode(fails):
    """MEASURES: sanctioned raw import #1 - the AI-draw identity
    wiring - on a REAL raw specimen (lab pairs log; its per-game
    passwords from wotr_bridge/game_passwords).  Asserts BOTH halves:
      * harvest decodes the AI side's own draws to engine cids of the
        right deck bucket and feeds rand/'card' + view.card_pid_map +
        ai_drawn_pids through handle();
      * blinding fail-closed: the HUMAN side's draws are never decoded
        and the human side's commitment tables never crack (poison
        password), even though the claim line is in the stream."""
    import io as _io
    import tempfile as _tf
    if not os.path.exists(_PAIR_RAW):
        fails.append(('draw_decode',
                      'NOT-MEASURED: specimen missing %s' % _PAIR_RAW))
        return
    import game_passwords
    from wotr_bridge.live.card_identity import CardIdentity
    stem = os.path.basename(_PAIR_RAW)[:-len('.raw.log')]
    _fp_pw, sa_pw = game_passwords.get_passwords(stem)
    raw = _io.open(_PAIR_RAW, encoding='utf-8',
                   errors='replace').read().split('\n')
    lines = raw[:60]        # SA claim + 2 SA draws + FP claim + 2 FP draws
    td = _tf.mkdtemp()
    open(os.path.join(td, 'ai_feed.txt'), 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=11)
    # this 2023 game was claimed with the players' own passwords, not
    # 'ai' - swap in an identity decoder built with the game's SA
    # password (same class, same wiring; only the password differs)
    srv.cards = CardIdentity('SHADOW', password=sa_pw)
    srv.handle({'lines': lines})
    fed = srv.rand._feed['cards']
    ch = fed.get(('SHADOW', 'character'), [])
    st = fed.get(('SHADOW', 'strategy'), [])
    if len(ch) != 1 or len(st) != 1:
        fails.append(('draw_decode_fed',
                      'char=%d strat=%d' % (len(ch), len(st))))
        return
    (cid_c, pid_c), (cid_s, pid_s) = ch[0], st[0]
    if pid_c != '208' or pid_s != '180':
        fails.append(('draw_decode_pids', (pid_c, pid_s)))
    if not (cid_c.startswith('SHADOW') and '_Character_' in cid_c):
        fails.append(('draw_decode_cid_char', ascii(cid_c)))
    if not (cid_s.startswith('SHADOW') and '_Strategy_' in cid_s):
        fails.append(('draw_decode_cid_strat', ascii(cid_s)))
    view = srv.driver.view
    if view.card_pid_map.get(cid_c) != '208' \
            or view.card_pid_map.get(cid_s) != '180':
        fails.append(('draw_decode_pidmap', ascii(view.card_pid_map)))
    if view.ai_drawn_pids != {'208', '180'}:
        fails.append(('draw_decode_drawn', view.ai_drawn_pids))
    # ---- blinding fail-closed ----
    if ('FREE_PEOPLES', 'character') in fed \
            or ('FREE_PEOPLES', 'strategy') in fed:
        fails.append(('blinding_fed_human_cards', ascii(fed)))
    dec = srv.cards.decoder
    if len(dec.fp_char_deck) or len(dec.fp_strat_deck):
        fails.append(('blinding_fp_decks_cracked',
                      (len(dec.fp_char_deck), len(dec.fp_strat_deck))))
    if len(dec.sa_char_deck) != 24 or len(dec.sa_strat_deck) != 24:
        fails.append(('sa_decks_not_cracked',
                      (len(dec.sa_char_deck), len(dec.sa_strat_deck))))
    # cursor discipline: the SAME full log resent (AiOpp resends the
    # whole log every tick) must feed NOTHING new
    srv.handle({'lines': lines})
    if len(fed.get(('SHADOW', 'character'), [])) != 1 \
            or len(fed.get(('SHADOW', 'strategy'), [])) != 1:
        fails.append(('draw_decode_refeed', ascii(fed)))


def test_die_piece_binding(fails):
    """MEASURES: pre-roll die naming + raw-roll binding hygiene.

    REPRODUCTION (Phase 3, 2026-08-19, real-net offline walk of the
    exact live trajectory):
      R3  the D3 eye-allocation EMISSION (a die drag pool->Hunt Box,
          fact 12) died 'no die piece for SHADOW die 0' - ClientView
          dropped the donor's base-die pre-seed (log_replay.py:726-730;
          fact 11: SH 099-105, FP 210-213 are fixed client pieces);
      R4  driver._bind_roll_pids wrote setdefault(pid, None) - the
          first raw action-roll line poisoned die_id_map and
          emitter.die_piece_for died unpacking None."""
    import tempfile as _tf
    import emitter
    td = _tf.mkdtemp()
    open(os.path.join(td, 'ai_feed.txt'), 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=5)
    view = srv.driver.view
    # R3: base dice resolvable BEFORE any roll
    try:
        if emitter.die_piece_for(view, 'SHADOW', 0) != '099' \
                or emitter.die_piece_for(view, 'FREE_PEOPLES', 0) != '210':
            fails.append(('die_preseed_wrong',
                          ascii(view.die_id_map)))
    except Exception as e:
        fails.append(('die_preseed', 'R3 still live: %r' % (e,)))
        return
    # R4 (superseded by the WAVE-H cbt3 finding): a raw roll line's
    # pid ORDER binds pids to the adopted engine indices - the
    # pre-seed alone mis-labels every die once the client's pool
    # order changes (cbt3 turn-2: FP pids listed REVERSED after
    # recover; the harness staged a die whose client face disagreed
    # with the engine's and an engine-legal move was refused).  The
    # binding lands at ADOPTION (letters line), never as a None
    # placeholder (old R4), and covers bonus pids (360) on the line.
    # RAW SPECIMEN IN THE REAL CLIENT GRAMMAR (cbt6 live logs:
    # Controls.345=pids, Controls.346=values + trailing checksum; the
    # old fabricated 346/347 specimen had been WRITTEN FROM the wrong
    # lifted regex - it pinned a shape no client ever emits)
    feed2 = os.path.join(td, 'ai_feed.txt')
    raw = ('$Wey<~Controls.345~>100 099 360 '
           '<~Controls.346~>15 12 13 AB12CD')
    srv.handle({'lines': [raw]})
    try:
        if emitter.die_piece_for(view, 'SHADOW', 0) != '099':
            fails.append(('die_preseed_kept_preroll',
                          ascii(view.die_id_map)))
    except Exception as e:
        fails.append(('die_after_rawline', 'poisoned: %r' % (e,)))
    if None in view.die_id_map.values():
        fails.append(('die_map_none_poison', ascii(view.die_id_map)))
    # letters adopt 3 faces onto SHADOW idx 0,1,2 -> pid order 100,
    # 099, 360 must bind those SAME indices in order
    _append(feed2, '$Wey rolls E A C .')
    srv.handle({'lines': ()})
    got = {}
    try:
        for i in range(3):
            got[i] = emitter.die_piece_for(view, 'SHADOW', i)
    except Exception as e:
        fails.append(('die_bind_adopt', repr(e)))
        return
    if got != {0: '100', 1: '099', 2: '360'}:
        fails.append(('die_bind_order', got))
    # cbt4: the OTHER channel order - letters polled BEFORE the raw
    # line arrives - must bind retroactively
    _append(feed2, '$Wey rolls C W .')     # W marks the FP pool
    srv.handle({'lines': ()})
    raw_fp = ('$Wey<~Controls.345~>211 210 <~Controls.346~>1 0 9F00AA')
    srv.handle({'lines': [raw_fp]})
    try:
        if emitter.die_piece_for(view, 'FREE_PEOPLES', 0) != '211' \
                or emitter.die_piece_for(view, 'FREE_PEOPLES', 1) \
                != '210':
            fails.append(('die_bind_retro',
                          ascii(view.die_id_map)))
    except Exception as e:
        fails.append(('die_bind_retro_err', repr(e)))


_BOOT2_RAW = os.path.join(
    _ROOT, 'wotr_bridge', 'play_client', 'logs', 'archive',
    'wotr2026_8_20_22_51-1.log')   # the USER's real 2026-08-20 session (the
                                     # boot2 leg specimen was retired 2026-08-22)


def test_ai_draw_decode_live_variants(fails):
    """MEASURES: the draw-grammar VARIANT coverage on the boot2 leg's
    OWN raw log (a real sided boot, hands claimed with 'ai' - the
    production password path).

    REPRODUCTION (boot2, 2026-08-19): every deck menu emits its own
    Controls family (491/509/0/549/557; donor _DRAW_LINE_RE knew this,
    ai_move_server.py:96-98).  The lifted 509-only shape matched NONE
    of the live draws: the AI's sdrawc line
    ($Wey<~Controls.0~>200<~Controls.549~><~Game.2006~>...) was never
    harvested, the ai_draws step starved on 'character' forever, and
    the directive dedup turned the wedge into silent empty replies."""
    import io as _io
    if not os.path.exists(_BOOT2_RAW):
        fails.append(('draw_variants',
                      'NOT-MEASURED: boot2 specimen missing %s'
                      % _BOOT2_RAW))
        return
    from wotr_bridge.live.card_identity import CardIdentity
    lines = _io.open(_BOOT2_RAW, encoding='utf-8',
                     errors='replace').read().split('\n')
    ci = CardIdentity('SHADOW')          # production default: pw 'ai'
    out = ci.harvest(lines)
    # the session's Shadow seat drew from BOTH decks (character 491/549
    # family and strategy 509/557 family): every variant must decode to
    # a real pid of the right deck, for the AI's own side only
    ai_out = [d for d in out if d['side'] == 'SHADOW']
    hu_out = [d for d in out if d['side'] == 'FREE_PEOPLES']
    decks = sorted(set(d['deck'] for d in ai_out))
    if len(ai_out) < 2 or decks != ['character', 'strategy']:
        fails.append(('variants_count',
                      [(d['pid'], d['deck']) for d in ai_out]))
        return
    for d in out:
        if not str(d['pid']).isdigit():
            fails.append(('variant_decode', (d['pid'], d['deck'],
                                             d['side'])))
            return
    # USER 2026-08-25 (the engine sees the human hand): the human FP
    # draws on the same log harvest too, and decode to real cids
    # (boot2 claimed BOTH hands with 'ai')
    if hu_out and not any(d['cid'] for d in hu_out):
        fails.append(('human_draw_decode',
                      [(d['pid'], d['name']) for d in hu_out]))


def test_step_queue_no_clobber(fails):
    """MEASURES boot blocker 3: 'fp ok' arriving in the SAME tick
    batch as the open ai_draws step must QUEUE the eye step, never
    clobber the draws step whose directives are already ordered.  The
    queued step runs only after the draws commit."""
    td = tempfile.mkdtemp()
    feed = os.path.join(td, 'ai_feed.txt')
    open(feed, 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=41)
    drv = srv.driver
    # both human draws AND fp ok land in ONE tick
    _append(feed, '$Wey draws Free Character Card.',
            '$Wey draws Free Strategy Card.',
            '<Wey> fp ok')
    out = srv.handle({'lines': ()})
    if srv._open_step is None or srv._open_step[0] != 'ai_draws':
        fails.append(('clobber_open', repr(srv._open_step)))
        return
    if not srv._step_queue or srv._step_queue[0][0] != 'ai_eyes':
        fails.append(('clobber_queue', repr(srv._step_queue)))
    if not any('draw_sa_char' in s for s in out):
        fails.append(('clobber_draw_ask', repr(out)))
    # feed both draws -> draws commit -> the QUEUED eye step runs
    srv.rand.feed('card', ('SHADOW', 'character',
                           drv.es.sh_char_deck[-1]))
    out = srv.handle({'lines': ()})
    srv.rand.feed('card', ('SHADOW', 'strategy',
                           drv.es.sh_strat_deck[-1]))
    out = srv.handle({'lines': ()})
    if not any('roll_action_dice' in s for s in out):
        fails.append(('clobber_eyes_ran', repr(out)))
    if srv._step_queue:
        fails.append(('clobber_queue_drained', repr(srv._step_queue)))


def test_status_menu_publication(fails):
    """MEASURES (PHASE3-B): the status-menu publication - the engine's
    own legal menu for the HUMAN side summarized into status_extra so a
    harness can play human board gestures FROM THE CURRENT POSITION
    (USER gesture ruling 2026-08-19).  Asserts: (1) menu withheld until
    the action phase; (2) once both pools are rolled, the payload
    carries to_act=FREE_PEOPLES, an army_move entry, JSON-serializable
    params, and the human dice with faces + client pids (base pre-seed
    210..213); (3) card-identity redaction on the DIAGNOSTIC param
    path (_menu_params_safe - the published menu itself carries
    identities since the 2026-08-25 hand-visibility ruling, pinned in
    test_human_hand_visible_in_menu); (4) the hook is wired into
    build()'s FileBridge."""
    import json
    from wotr_bridge.live.live_server import (_menu_params_safe, build)
    td = tempfile.mkdtemp()
    feed = os.path.join(td, 'ai_feed.txt')
    open(feed, 'w').close()
    srv, bridge = build(td, 'SA', load_nets=False, seed=17)
    # bound-method equality (an `is` test can never pass: each
    # attribute access mints a fresh bound-method object)
    if bridge.status_extra != srv.status_menu:
        fails.append(('status_hook_unwired',
                      repr(bridge.status_extra)))
    drv = srv.driver
    # (3) redaction unit contract (fail-closed: even placeholders)
    red = _menu_params_safe({'card_id': 'SHADOW_Character_X',
                             'from': 'gondor', 'n_regular': 2})
    if red.get('card_id') != '<hidden>' or red.get('n_regular') != 2:
        fails.append(('status_redaction', red))
    # (1) before the action phase: no menu key (draws phase)
    s0 = srv.status_menu()
    hm0 = s0.get('human_menu', {})
    if 'error' in hm0:
        fails.append(('status_error_pre', hm0['error']))
        return
    if hm0.get('phase') != 'draws' or 'actions' in hm0:
        fails.append(('status_pre_actions', (hm0.get('phase'),
                                             'actions' in hm0)))
    # walk the turn-start to the action phase (same walk as the smoke)
    _append(feed, '$Wey draws Free Character Card.',
            '$Wey draws Free Strategy Card.')
    srv.handle({'lines': ()})
    srv.rand.feed('card', ('SHADOW', 'character',
                           drv.es.sh_char_deck[-1]))
    srv.handle({'lines': ()})
    srv.rand.feed('card', ('SHADOW', 'strategy',
                           drv.es.sh_strat_deck[-1]))
    srv.handle({'lines': ()})
    _append(feed, '<Wey> fp ok')
    srv.handle({'lines': ()})
    unset = [d for d in drv.es.dice[Side.SHADOW]
             if not d.used and d.current is None]
    _append(feed, '$Wey rolls %s .'
            % ' '.join(['E'] + ['A'] * (len(unset) - 1)))
    srv.handle({'lines': ()})
    # the human's own pool roll (letters carry the FP marker 'W')
    _append(feed, '$Wey rolls W C P M .')
    srv.handle({'lines': ()})
    # (2) the action-phase publication
    s1 = srv.status_menu()
    hm = s1.get('human_menu', {})
    if 'error' in hm:
        fails.append(('status_error', hm['error']))
        return
    if hm.get('to_act') != 'FREE_PEOPLES' or hm.get('phase') != 'actions':
        fails.append(('status_to_act', (hm.get('to_act'),
                                        hm.get('phase'))))
    if not hm.get('n_actions') or not hm.get('by_type', {}).get(
            'army_move'):
        fails.append(('status_menu_empty',
                      (hm.get('n_actions'), hm.get('by_type'))))
        return
    am = [a for a in hm.get('actions', ())
          if a['action_type'] == 'army_move']
    if not am or not am[0]['params'].get('from') \
            or am[0]['die_idx'] is None:
        fails.append(('status_army_entry', am[:1]))
    dice = hm.get('dice', ())
    if len(dice) != 4 or dice[0].get('pid') != '210' \
            or not all(d.get('face') for d in dice):
        fails.append(('status_dice', dice))
    prog = s1.get('progress', {})
    if 'n_commits' not in prog or 'n_alarms' not in prog:
        fails.append(('status_progress', prog))
    # the whole payload must survive the transport's json.dumps
    try:
        json.dumps(s1)
    except Exception as e:
        fails.append(('status_not_json', repr(e)))


def test_casualty_rows_publication(fails):
    """MEASURES (WAVE-J, the wirF wedge): per-option ROWS on a
    human-owned casualty_allocation menu are the ENGINE's own
    clone-applied exact operation and forced continuation, never a
    hits-as-pieces recomputation.  Asserts: (a) options carry
    engine-resolved rows and DISTINCT options differ (hits=1 on a
    1R+1E army: remove_regular kills the Regular; reduce_elite reduces the Elite ->
    elite-1/regular+1); (b) THE ELITE=2 CONTRACT: hits=2 on the same
    army offers removing ONE whole Elite,
    never two; (c) payload survives json.dumps."""
    import json
    import ai
    from game_state import UnitKind
    td = tempfile.mkdtemp()
    open(os.path.join(td, 'ai_feed.txt'), 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=23)
    es = srv.driver.es
    target = None
    for key, a in es.armies.items():
        if len(a.units) == 2 \
                and {u.kind for u in a.units} == {UnitKind.REGULAR,
                                                  UnitKind.ELITE} \
                and all(ai._side_of_nation(u.nation)
                        is Side.FREE_PEOPLES for u in a.units):
            target = key
            break
    if target is None:
        fails.append(('no_1r1e_fp_army', sorted(es.armies)[:5]))
        return
    base_pc = {'phase': 'casualty_allocation',
               'alloc_stage': 'defender',
               'attacker_region': 'dol guldur',
               'attacker_side': Side.SHADOW,
               'defender_region': target,
               'defender_side': Side.FREE_PEOPLES,
               'atk_hits': 0, 'def_hits': 1,
               'deciding_side': Side.FREE_PEOPLES,
               'is_siege': False, 'rounds_fought': 1}
    es.pending_combat = dict(base_pc)
    s1 = srv.status_menu()
    hm = s1.get('human_menu', {})
    if 'error' in hm:
        fails.append(('rows_status_error', hm['error']))
        return
    if hm.get('pending') != 'combat:casualty_allocation':
        fails.append(('rows_pending', hm.get('pending')))
        return
    acts = {a['params'].get('operation'): a
            for a in hm.get('actions', ())
            if a['action_type'] == 'casualty_allocate'}
    if sorted(acts) != ['reduce_elite', 'remove_regular']:
        fails.append(('rows_options', sorted(acts)))
        return
    r0, r1 = acts['remove_regular'].get('rows'), acts['reduce_elite'].get('rows')
    if not r0 or not r1:
        fails.append(('rows_missing', (r0, r1)))
        return
    # (a) k=0 kills the regular; k=1 downgrades the elite
    if not (len(r0) == 1 and r0[0]['d'] == -1
            and r0[0]['tkey'].endswith(':regular')):
        fails.append(('rows_k0', r0))
    if not (any(r['d'] == -1 and r['tkey'].endswith(':elite')
                for r in r1)
            and any(r['d'] == 1 and r['tkey'].endswith(':regular')
                    for r in r1)):
        fails.append(('rows_k1', r1))
    if r0 == r1:
        fails.append(('rows_not_distinct', r0))
    # (b) the elite=2 contract at hits=2
    es.pending_combat = dict(base_pc, def_hits=2)
    s2 = srv.status_menu()
    hm2 = s2.get('human_menu', {})
    acts2 = {a['params'].get('operation'): a
             for a in hm2.get('actions', ())
             if a['action_type'] == 'casualty_allocate'}
    if 'remove_elite' not in acts2 or not acts2['remove_elite'].get('rows'):
        fails.append(('rows2_missing', sorted(acts2)))
        return
    removed = sum(-r['d'] for r in acts2['remove_elite']['rows'] if r['d'] < 0)
    if removed != 1:
        fails.append(('rows_elite2_contract',
                      'hits=2 removed %d net pieces (rows=%r) - the '
                      'wirF hits-as-pieces class'
                      % (removed, acts2['remove_elite']['rows'])))
    try:
        json.dumps(s1)
        json.dumps(s2)
    except Exception as e:
        fails.append(('rows_not_json', repr(e)))
    es.pending_combat = None


def test_status_wave_j_blocks(fails):
    """MEASURES (WAVE-J): (1) ai_open/open_step/owed_die_kills ride
    the progress block (the server's own statement replacing the
    harness's feed-scanning ai_die_open); (2) echo_open serializes
    the driver's open echo-ledger; (3) the hunt block publishes the
    engine's damage; (4) FELLOWSHIP BLINDING: the block is published
    to an FP human and NEVER to a SHADOW human (the pawn's true
    region is FP-hidden information)."""
    import json
    td = tempfile.mkdtemp()
    open(os.path.join(td, 'ai_feed.txt'), 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=29)
    drv = srv.driver
    # (1) ai_open
    p0 = srv.status_menu().get('progress', {})
    if p0.get('ai_open') or p0.get('open_step') is not None:
        fails.append(('ai_open_idle', (p0.get('ai_open'),
                                       p0.get('open_step'))))
    srv._open_step = ('ai_eyes', None)
    p1 = srv.status_menu().get('progress', {})
    if not p1.get('ai_open') or p1.get('open_step') != 'ai_eyes':
        fails.append(('ai_open_step', (p1.get('ai_open'),
                                       p1.get('open_step'))))
    srv._open_step = None
    drv.owed_die_kills.append(('SHADOW', 0))
    p2 = srv.status_menu().get('progress', {})
    if not p2.get('ai_open') or p2.get('owed_die_kills') != 1:
        fails.append(('ai_open_owed', (p2.get('ai_open'),
                                       p2.get('owed_die_kills'))))
    drv.owed_die_kills[:] = []
    # (2) echo_open
    drv.echo_ledger.append({'kind': 'unit', 'side': 'fp',
                            'region': 'osgiliath',
                            'tkey': 'gondor:regular', 'n': 2})
    hm = srv.status_menu().get('human_menu', {})
    eo = hm.get('echo_open')
    if eo != [{'kind': 'unit', 'side': 'fp', 'region': 'osgiliath',
               'tkey': 'gondor:regular', 'n': 2}]:
        fails.append(('echo_open', eo))
    drv.echo_ledger[:] = []
    # (3) hunt block
    drv.es.pending_hunt = {'phase': 'hunt_sacrifice', 'damage': 2,
                           'companion': None, 'level': None,
                           'casualty_taken': False}
    hm3 = srv.status_menu().get('human_menu', {})
    if (hm3.get('hunt') or {}).get('damage') != 2:
        fails.append(('hunt_block', hm3.get('hunt')))
    drv.es.pending_hunt = None
    # (5) WAVE-J alternation publication (the wjg1 deadlock killer):
    # phase 'actions' + a recorded last_actor -> the engine's
    # ai._next_acting_side answer rides the payload; no actor yet ->
    # None (unknown, harness may act)
    import ai as _ai
    import game_logic as _gl
    from game_state import ActionDieFace as _F
    ph0 = drv.phase
    drv.phase = 'actions'
    drv.last_actor = Side.SHADOW
    _gl.adopt_rolled_faces(drv.es, [(Side.FREE_PEOPLES, 0,
                                     _F.CHARACTER)])
    alt = srv.status_menu().get('human_menu', {}).get('alternation')
    want = _ai._next_acting_side(drv.es, Side.SHADOW)
    want = want.name if want is not None else None
    if alt is None or alt != want:
        fails.append(('alternation_pub', (alt, want)))
    drv.last_actor = None
    alt2 = srv.status_menu().get('human_menu', {}).get('alternation')
    if alt2 is not None:
        fails.append(('alternation_none', alt2))
    drv.phase = ph0
    # (4) fellowship: FP human sees it...
    hm4 = srv.status_menu().get('human_menu', {})
    fsb = hm4.get('fellowship')
    if not fsb or 'progress' not in fsb or not fsb.get('region') \
            or not fsb.get('companions'):
        fails.append(('fellowship_block', fsb))
    try:
        json.dumps(srv.status_menu())
    except Exception as e:
        fails.append(('wave_j_not_json', repr(e)))
    # (4b) WAVE-J turn-counter witness (the wjg7 never-ticking marker,
    # client 1 vs engine 6): a client turn disagreeing with the engine
    # turn is an issue; an agreeing one is not.  REWRITTEN 2026-08-29
    # (BRCMP-8): the dump exports no turn marker (0 of 55 captures
    # carry a Game.7 row), so the old '<~Game.7~>N' zone-value regex
    # was a dead leg; the witness reads the caller's `client_turn`
    # channel and is NOT-MEASURED (no row) without one.
    from wotr_bridge.live import client_board as _cb
    rows_tc = _cb.diff_engine_vs_dump(drv.es, {'null': 'Turn Track'},
                                      client_turn=3)
    if not any('turn counter' in str(r) for r in (rows_tc or [])):
        fails.append(('turn_counter_witness', rows_tc))
    rows_tc2 = _cb.diff_engine_vs_dump(
        drv.es, {'null': 'Turn Track'},
        client_turn=int(getattr(drv.es, 'turn', 1) or 1))
    if any('turn counter' in str(r) for r in (rows_tc2 or [])):
        fails.append(('turn_counter_specificity', rows_tc2))
    rows_tc3 = _cb.diff_engine_vs_dump(drv.es, {'null': '<~Game.7~>3'})
    if any('turn counter' in str(r) for r in (rows_tc3 or [])):
        fails.append(('turn_counter_dead_regex_revived', rows_tc3))
    # ... and a SHADOW human gets ONLY the physical-board-public
    # subset (BL-36, 2026-08-25): corruption/guide/hidden/progress
    # are open information; the pawn REGION and Mordor step stay
    # FP-hidden - THOSE leaking is the blinding failure
    td2 = tempfile.mkdtemp()
    open(os.path.join(td2, 'ai_feed.txt'), 'w').close()
    srv2 = LiveServer(td2, Side.FREE_PEOPLES, load_nets=False,
                      seed=31)
    hm5 = srv2.status_menu().get('human_menu', {})
    if 'error' in hm5:
        fails.append(('blinding_status_error', hm5['error']))
    else:
        fs5 = hm5.get('fellowship')
        if not fs5 or 'corruption' not in fs5:
            fails.append(('fellowship_public_block_missing', fs5))
        elif any(k in fs5 for k in ('region', 'mordor',
                                    'mordor_step', 'companions')):
            fails.append(('fellowship_blinding_leak', fs5))


def test_slot_cleanup(fails):
    """MEASURES (WAVE-J priority insert - the resurfaced slot-cleanup
    family): (1) a played event card's owed kill flushes as the E1
    GROUP kill (die + card in ONE line - the corpus shape; the
    die-only flush stranded wjg1/wjg2's played cards in the
    Current-Card slot); (2) the tile mop-up sweeps a resolved display
    tile to Used Hunt Tiles Game.2045 THROUGH handle() (the donor's
    AUTO-only-linkage lesson: the wiring is the thing under test);
    (3) the sweep HOLDS while a hunt pending is open (donor t130);
    (4) a card-added SPECIAL tile parks Removed -> Used (USER B1,
    the 8-special pin via the piece table)."""
    import json as _json
    import ai
    import emitter
    td = tempfile.mkdtemp()
    open(os.path.join(td, 'ai_feed.txt'), 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=37)
    drv = srv.driver
    # ---- (1) E1 group kill ------------------------------------------
    drv.view.card_pid_map['SHADOW_Character_Test_Card'] = '177'
    act = ai.Action(die_idx=0, action_type='play_event',
                    params={'card_id': 'SHADOW_Character_Test_Card'})
    srv._note_owed_die_kill({'side': Side.SHADOW}, act)
    if not drv.owed_die_kills \
            or len(drv.owed_die_kills[0]) < 3 \
            or drv.owed_die_kills[0][2] != '177':
        fails.append(('e1_banked', drv.owed_die_kills))
        return
    lines = srv._flush_owed_die_kills()
    if len(lines) != 1:
        fails.append(('e1_one_line', lines))
        return
    m = emitter.OPS['kill']['re'].match(lines[0])
    pids = (m.groupdict().get('pieces') or '').split() if m else []
    if not m or pids != ['099', '177']:
        fails.append(('e1_group_kill',
                      'die AND card must die in ONE kill line '
                      '(got %r)' % (lines[0],)))
    if drv.owed_die_kills:
        fails.append(('e1_cleared', drv.owed_die_kills))
    # ---- synthetic full dump (client fact 14: >=400 pieces) ---------
    root = os.path.dirname(_SRC)
    with open(os.path.join(root, 'wotr_bridge',
                           'initial_positions.json'),
              encoding='utf-8') as f:
        init = _json.load(f)
    init = dict(init)
    init['219'] = 'Game.2036'     # a drawn Eye tile in SA Card Play
    # pad to the torn-dump minimum (client fact 14: >=400 pieces;
    # initial_positions carries 391) with inert parking-box pids
    for _i in range(900, 915):
        init[str(_i)] = 'Game.2034'
    # the 8 SPECIAL tiles never move in the corpus so the modal-drag
    # derivation has no entry - the LIVE dump (wjg1 census) parks
    # them in Removed Hunt Tiles from setup
    for _i in range(232, 240):
        init[str(_i)] = 'Game.2046'

    def _write_dump():
        by_code = {}
        for pid, code in init.items():
            by_code.setdefault(code, []).append(pid)
        rows = []
        for i, (code, pids2) in enumerate(sorted(by_code.items())):
            rows.append('%d\t<~%s~>\t%s\t%s'
                        % (i, code, ' '.join(sorted(pids2)),
                           ' '.join('-' for _ in pids2)))
        p = os.path.join(td, 'ai_client_board.txt')
        with open(p, 'w', encoding='utf-8') as f2:
            f2.write('\n'.join(rows) + '\n')
        return p
    drv.view.dump_path = _write_dump()
    # a resolved Eye hunt: the engine tile sits in hunt_discard
    eye = next((t for t in drv.es.hunt_pool
                if 'eye' in str(t).lower()), None)
    if eye is None:
        fails.append(('tile_precondition', 'no eye tile in pool'))
        return
    drv.es.hunt_pool.remove(eye)
    drv.es.hunt_discard.append(eye)
    # ---- (3) gate: hunt pending open -> NO sweep --------------------
    drv.es.pending_hunt = {'phase': 'hunt_sacrifice', 'damage': 1}
    out = srv.handle({'lines': ()})
    if any('Game.2045' in ln for ln in out):
        fails.append(('tile_gate_hunt_open', out))
    drv.es.pending_hunt = None
    # ---- (2) sweep through handle() ---------------------------------
    out = srv.handle({'lines': ()})
    sweep = [ln for ln in out if '219' in ln and 'Game.2036' in ln
             and 'Game.2045' in ln]
    if not sweep:
        fails.append(('tile_sweep', out))
    # ---- (4) B1 special-in-play: Removed -> Used --------------------
    sp = type('T', (), {'is_special': True,
                        'special_name': 'The Ring is Mine!'})()
    drv.es.hunt_discard.append(sp)
    out = srv.handle({'lines': ()})
    b1 = [ln for ln in out if '236' in ln and 'Game.2046' in ln
          and 'Game.2045' in ln]
    if not b1:
        fails.append(('tile_b1_special', out))
    # idempotence: a further tick emits no duplicate B1 line
    out = srv.handle({'lines': ()})
    if any('236' in ln and 'Game.2046' in ln for ln in out):
        fails.append(('tile_b1_idempotent', out))
    # ---- (5) TABLE cards: a persistent card is NOT group-killed -
    # it seats into a Board slot, and dies IN ITS SLOT on expiry
    from game_state import PersistentCard
    tc = PersistentCard(card_id='SHADOW_Strategy_Test_Table',
                        card_name='Test Table', side=Side.SHADOW)
    drv.es.persistent_cards.sh_cards.append(tc)
    drv.view.card_pid_map['SHADOW_Strategy_Test_Table'] = '208'
    act2 = ai.Action(die_idx=1, action_type='play_event',
                     params={'card_id': 'SHADOW_Strategy_Test_Table'})
    srv._note_owed_die_kill({'side': Side.SHADOW}, act2)
    lines2 = srv._flush_owed_die_kills()
    kills2 = [ln for ln in lines2
              if emitter.OPS['kill']['re'].match(ln)]
    if any('208' in (emitter.OPS['kill']['re'].match(ln)
                     .groupdict().get('pieces') or '')
           for ln in kills2):
        fails.append(('table_not_killed', lines2))
    seats = [ln for ln in lines2 if '208' in ln
             and any(('Game.%d' % z) in ln
                     for z in range(2029, 2035))]
    if not seats:
        fails.append(('table_seated', lines2))
    # expiry -> kill IN ITS SLOT: the attested corpus grammar for a
    # kill at a Board slot is Controls.296/297 (hand_exit_b at the
    # slot; 645 with a Board token has zero corpus specimens - W1#75)
    drv.es.persistent_cards.sh_cards.remove(tc)
    out2 = srv.handle({'lines': ()})
    exp = [ln for ln in out2
           if emitter.OPS['hand_exit_b']['re'].match(ln)
           and '208' in ln
           and any(('Game.%d' % z) in ln for z in range(2029, 2035))]
    if not exp:
        fails.append(('table_expiry_kill', out2))


def test_game_end(fails):
    """MEASURES (WAVE-J, ledger A-09/A-10 - a game that cannot END
    cannot be clean): (1) ring corruption 12 ends the game at a COMMIT
    (mid-action); the closer goes out LAST and ONCE, walks the
    corruption marker to its final slot, says NO 'gg' when the AI won;
    later cycles return EMPTY replies (fact 16 autoplay termination);
    (2) RULING 1.212 timing: a MILITARY verdict does NOT end the game
    mid-action and DOES at turn end (pinned through the driver's own
    check_victory over a patched engine verdict); (3) a human
    'gg'/'concede' chat ends the game (A-10); (4) the AI takes no
    further steps after game over."""
    import json as _json
    import game_logic as gl
    td = tempfile.mkdtemp()
    feed = os.path.join(td, 'ai_feed.txt')
    open(feed, 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=41)
    drv = srv.driver
    # synthetic dump so the closer can walk the marker
    root = os.path.dirname(_SRC)
    with open(os.path.join(root, 'wotr_bridge',
                           'initial_positions.json'),
              encoding='utf-8') as f:
        init = _json.load(f)
    init = dict(init)
    for _i in range(900, 915):
        init[str(_i)] = 'Game.2034'
    init['252'] = 'Game.2019'          # corruption marker at Track 5
    by_code = {}
    for pid, code in init.items():
        by_code.setdefault(code, []).append(pid)
    p = os.path.join(td, 'ai_client_board.txt')
    with open(p, 'w', encoding='utf-8') as f2:
        f2.write('\n'.join(
            '%d\t<~%s~>\t%s\t%s' % (i, c, ' '.join(sorted(ps)),
                                    ' '.join('-' for _ in ps))
            for i, (c, ps) in enumerate(sorted(by_code.items())))
            + '\n')
    drv.view.dump_path = p
    # (2) 1.212 timing pin on a patched military verdict
    orig = gl.check_all_victory_conditions
    gl.check_all_victory_conditions = lambda st: {
        'victory': 'shadow', 'victory_type': 'military',
        'message': 'test', 'free_peoples_vp': 0, 'shadow_vp': 10}
    try:
        if drv.check_victory(end_of_turn=False) is not None:
            fails.append(('military_mid_action',
                          'RULING 1.212: military must NOT end the '
                          'game mid-action'))
        drv.game_over = None
        if drv.check_victory(end_of_turn=True) is None:
            fails.append(('military_turn_end', 'not detected'))
    finally:
        gl.check_all_victory_conditions = orig
        drv.game_over = None
    # (1) ring corruption at a REAL commit (WAVE-K R1 / PARITY-AUDIT-2:
    # the old in-place mutation could not see that check_victory ran on
    # the PRE state - a mid-action ring victory landed one commit LATE
    # and the AI's next step could emit after the game was won).  pre =
    # corruption 11 + a 1-damage hunt pending; the FP's hunt_take_
    # corruption commits through the REAL chat-commit path and the POST
    # state (12) must end the game AT that commit.
    import ai as _ai
    drv.es.fellowship.corruption = 11
    drv.es.pending_hunt = {'phase': 'hunt_sacrifice', 'damage': 1}
    with srv.rand.pin():
        _dp = _ai.describe_pending(drv.es, Side.FREE_PEOPLES)
    _act = next((a for a in (_dp['actions'] if _dp else [])
                 if a.action_type == 'hunt_take_corruption'), None)
    if _act is None:
        fails.append(('ring_mid_action', 'NOT-MEASURED: no '
                      'hunt_take_corruption in the FP hunt menu'))
        return
    drv._commit_human(_act, via='chat')
    if not drv.game_over \
            or drv.game_over.get('victory_type') != 'ring_corruption':
        fails.append(('ring_mid_action_post_state',
                      'victory not detected at the commit whose POST '
                      'state wins: game_over=%r corruption=%s'
                      % (drv.game_over, drv.es.fellowship.corruption)))
        return
    out = srv.handle({'lines': ()})
    if any(str(ln).startswith('!CALL') for ln in out):
        fails.append(('ai_emits_after_win', out))
    if not out:
        fails.append(('closer_emitted', out))
        return
    walk = [ln for ln in out if '252' in ln and 'Game.2019' in ln
            and 'Game.2026' in ln]
    if not walk:
        fails.append(('closer_marker_walk', out))
    if any(ln.endswith('gg') for ln in out):
        fails.append(('gg_on_ai_win', out))
    out2 = srv.handle({'lines': ()})
    if out2:
        fails.append(('empty_after_over', out2))
    # (3) concession on a FRESH server + (4) no further AI steps
    td2 = tempfile.mkdtemp()
    open(os.path.join(td2, 'ai_feed.txt'), 'w').close()
    srv2 = LiveServer(td2, Side.SHADOW, load_nets=False, seed=43)
    with open(os.path.join(td2, 'ai_feed.txt'), 'a',
              encoding='utf-8') as f:
        f.write('<Wey> gg\n')
    srv2.handle({'lines': ()})
    go = srv2.driver.game_over
    if not go or go.get('victory_type') != 'concession' \
            or go.get('victory') != 'shadow':
        fails.append(('concession', go))
    if srv2._open_step is not None or srv2._step_queue:
        fails.append(('ai_stopped', (srv2._open_step,
                                     srv2._step_queue)))


def test_j1_bonus_dice(fails):
    """MEASURES (WAVE-J, ledger A-11 - the J1 regression, THIRD
    occurrence): a minion in play GROWS the AI pool at the recover
    step - calculate_bonus_dice on the LIVE state + the physical
    growth drag for the AI side, with the new die pid pre-bound.
    B13-21 pin: a SECOND rollover grants NOTHING (idempotent).
    Blinding half: the HUMAN side's growth is never dragged by us."""
    import emitter
    td = tempfile.mkdtemp()
    open(os.path.join(td, 'ai_feed.txt'), 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=47)
    drv = srv.driver
    drv.phase = 'actions'
    drv.es.saruman.in_play = True           # +1 SHADOW die
    # BL-13: a recover with unspent dice is now refused - this test
    # measures the J1 grant on a LEGITIMATE rollover, so spend both
    # pools first
    from game_state import ActionDieFace
    for _s in (Side.FREE_PEOPLES, Side.SHADOW):
        for _d in drv.es.dice.get(_s, []):
            _d.current = ActionDieFace.CHARACTER
            _d.used = True
    n0 = len(drv.es.dice[Side.SHADOW])
    items = drv._on_recover({'norm': 'test recover'})
    kinds = [i[0] for i in items]
    if 'bonus_dice' not in kinds or 'recover_ai' not in kinds:
        fails.append(('j1_items', items))
        return
    n1 = len(drv.es.dice[Side.SHADOW])
    if n1 != n0 + 1:
        fails.append(('j1_pool_grown', (n0, n1)))
    payload = [i[1] for i in items if i[0] == 'bonus_dice'][0]
    lines = srv._bonus_dice_lines(payload)
    new_pid = str(emitter._SH_DIE_IDS[n0])
    if not any(new_pid in ln for ln in lines):
        fails.append(('j1_growth_line', (new_pid, lines)))
    if any(str(fp) in ln for ln in lines
           for fp in emitter._FP_DIE_IDS):
        fails.append(('j1_human_side_forged', lines))
    if drv.view.die_id_map.get(new_pid) != (Side.SHADOW, n0):
        fails.append(('j1_pid_bound',
                      drv.view.die_id_map.get(new_pid)))
    # WAVE-J (USER turn-counter report): the AI's recover is the
    # CLIENT'S OWN menu command, never hand-built bulk moves - the
    # handler is what advances the '<~Game.7~>N' turn marker
    rec_lines = srv._recover()
    if rec_lines != ['!CALL menu srecover']:
        fails.append(('menu_recover', rec_lines))
    # B13-21: second rollover -> no second grant (pools spent again
    # so the BL-13 guard does not mask the idempotency measurement)
    drv.phase = 'actions'
    for _s in (Side.FREE_PEOPLES, Side.SHADOW):
        for _d in drv.es.dice.get(_s, []):
            _d.current = ActionDieFace.CHARACTER
            _d.used = True
    items2 = drv._on_recover({'norm': 'test recover 2'})
    if 'bonus_dice' in [i[0] for i in items2]:
        fails.append(('j1_double_grant', items2))
    if len(drv.es.dice[Side.SHADOW]) != n1:
        fails.append(('j1_double_pool', len(drv.es.dice[Side.SHADOW])))


def test_mordor_entry_and_recovery(fails):
    """MEASURES (WAVE-J): (1) ledger A-29 - the Mordor entry check
    runs at 'fp ok' (after the declaration opportunity, same turn
    start) and renders the choreography ('fsp enters mordor', pawn to
    Mordor box 0, progress marker home); (2) ledger C-12 - the resync
    RECOVERY clears stuck expectation machinery loudly; (3) ledger
    F-06 - an un-blinded #HAND feed line halts fail-closed."""
    import board as board_mod
    td = tempfile.mkdtemp()
    open(os.path.join(td, 'ai_feed.txt'), 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=53)
    drv = srv.driver
    drv.phase = 'declare'
    b = board_mod.load_board()
    try:
        reg = b.region('Morannon')
    except Exception as e:
        fails.append(('mordor_precondition', repr(e)))
        return
    drv.es.fellowship.region = reg
    drv.es.fellowship.progress = 2
    with open(os.path.join(td, 'ai_feed.txt'), 'a',
              encoding='utf-8') as f:
        f.write('<Wey> fp ok\n')
    out = srv.handle({'lines': ()})
    if not drv.es.fellowship.mordor_entered:
        fails.append(('mordor_entered', False))
        return
    if not any('fsp enters mordor' in ln for ln in out):
        fails.append(('mordor_announce', out))
    if not any('248' in ln and 'Game.2016' in ln
               and 'Game.2014' in ln for ln in out):
        fails.append(('mordor_progress_home', out))
    # (2) recovery clears stuck machinery loudly
    drv.uncommitted = ['stuck line']
    items = drv.resync_recover()
    if drv.uncommitted:
        fails.append(('recovery_uncommitted', drv.uncommitted))
    if not any(k == 'resync_recovery' for k, _ in drv.findings):
        fails.append(('recovery_row', drv.findings[-3:]))
    # (3) #HAND fail-closed
    from wotr_bridge.live.driver import DriverHalt as _DH
    try:
        srv.handle({'lines': ['#HAND FREE_PEOPLES card1 card2']})
        fails.append(('hand_guard', 'no halt'))
    except _DH:
        pass


def test_randomcomp_ask(fails):
    """MEASURES (WAVE-J, the wjg10 LIVE starve): a deferred window
    starving on the random-companion draw (hunt casualty) gets its
    USER-mandated ask ordered - the chat 'random companion draw?' AND
    the '!CALL random_companion' builtin (the directive existed since
    LIFT-A; the driver's auto-order gate never let it out)."""
    import types
    td = tempfile.mkdtemp()
    open(os.path.join(td, 'ai_feed.txt'), 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=67)
    call = types.SimpleNamespace(kind='randomcomp', kw={})
    out = srv._window_ask_lines(call)
    if not any('!CALL random_companion' in ln for ln in out):
        fails.append(('randomcomp_call', out))
    if not any('random companion draw?' in ln for ln in out):
        fails.append(('randomcomp_chat', out))
    # the DRIVER gate lets it through for a starving deferral
    drv = srv.driver
    drv.deferred = types.SimpleNamespace()
    orig = srv.rand.pending
    srv.rand.pending = lambda: call
    try:
        items = drv.tick()
    finally:
        srv.rand.pending = orig
        drv.deferred = None
    if not any(i[0] == 'rng_ask' and i[1] is call for i in items):
        fails.append(('randomcomp_gate',
                      [i[0] for i in items]))
    # wjg11 LIVE: the global pending slot CLEARED by an interleaved
    # attempt - the deferral's own captured starve call must still
    # drive the ask (the starving window may never see its randomness
    # otherwise; 'DEFERRED WINDOW EXPIRED' was the visible corpse)
    drv.deferred = (types.SimpleNamespace(_starve_call=call),
                    Side.FREE_PEOPLES)
    srv.rand.pending = lambda: None
    try:
        items2 = drv.tick()
    finally:
        srv.rand.pending = orig
        drv.deferred = None
    if not any(i[0] == 'rng_ask' and i[1] is call for i in items2):
        fails.append(('randomcomp_clobbered_pending',
                      [i[0] for i in items2]))


def test_own_emission_tracking(fails):
    """MEASURES (WAVE-J, the wjg13 LIVE desync - A28 on our OWN lines):
    a kill line the server EMITS marks the pid off-board in the view at the
    emission exit, so a pick rendered BEFORE the echo arrives can never
    re-pick the dead piece (wjg13: 'kills Isengard Regular 002 in
    Orthanc' then 3s later 'moves group (... 002 ...) to Fords of
    Isen' - one real regular stranded client-side, resync desync);
    a move line updates the pids' zone. The pool is a MARK, never a pop
    (an untracked pid falls back to its setup zone)."""
    import emitter
    from emitter import render
    td = tempfile.mkdtemp()
    open(os.path.join(td, 'ai_feed.txt'), 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=71)
    view = srv.driver.view
    # control: pid 002 is pickable at Orthanc from the setup fallback
    before = emitter.pick_units(view, 'orthanc', 'shadow', n_regular=4,
                                side_hint='shadow')
    if '002' not in before:
        fails.append(('own_track_control',
                      'NOT-MEASURED: 002 not an Orthanc regular: %r'
                      % before))
        return
    kill = render('hand_exit_b', {'nick': srv.nick, 'pieces': ['002'],
                                  'frm': 'Game.1919'}, local=False)
    srv._track_own_emission([kill])
    if emitter._piece_zone_code(view, '002') != 'Game.2040':
        fails.append(('own_track_kill', view.piece_region.get('002')))
    after = emitter.pick_units(view, 'orthanc', 'shadow', n_regular=3,
                               side_hint='shadow')
    if '002' in after:
        fails.append(('own_track_repick',
                      'the dead pid was picked again (wjg13 class)'))
    mv = render('move', {'nick': srv.nick, 'pieces': ['003', '004'],
                         'frm': 'Game.1919', 'to': 'Game.1918'},
                local=False)
    srv._track_own_emission([mv])
    import log_replay as lr
    want = str(lr.decode_location('Game.1918'))
    if view.piece_region.get('003') != want:
        fails.append(('own_track_move',
                      (view.piece_region.get('003'), want)))


def test_deferral_watchdog(fails):
    """MEASURES (WAVE-J, the wjg4 silent-wedge class - live-reproduced
    2026-08-20 09:29-09:46, 16 minutes of empty RESPs), driven through
    the PRODUCTION stream (the feed FILE the driver tails -> drv.tick
    -> feed.lines), never a hand-set field:
    (1) EPOCH: a REFUSED window advances the ask epoch - the next
        window's identical ask GOES OUT (the wjg4 root: the illegal
        probe's refusal consumed the ask without a commit and the
        real fellowship move's hunt roll was falsely deduped);
    (2) WATCHDOG: an ask aged >= 25 ticks with NO rendered result in
        the feed re-sends LOUDLY (the client provably never executed
        it - re-asking cannot double-roll);
    (3) a result RENDERED in the feed since the ask (the client's own
        '<game> X rolls N' line, as the real feeds carry it) but
        unharvested is a STICKY ALARM and NO re-ask (the double-roll
        hazard) - for d6 AND for a hunt tile ('X draws Eye.');
    (4) raw REQUEST lines never count as a rendered result (the raw
        log carries the Controls grammar - the old check scanned it
        with the rendered grammar and could never fire)."""
    import types
    td = tempfile.mkdtemp()
    feed = os.path.join(td, 'ai_feed.txt')
    open(feed, 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=59)
    drv = srv.driver
    call = types.SimpleNamespace(kind='d6',
                                 kw={'need': 1, 'side': 'SHADOW',
                                     'reason': 'hunt roll'})
    _append(feed, '$Wey draws Free Character Card.')
    drv.tick([])                       # the feed stream is live
    out1 = srv._window_ask_lines(call)
    if not any('combat_roll 1' in ln for ln in out1):
        fails.append(('ask_sent', out1))
        return
    # same epoch retry tick -> deduped (the existing contract)
    if srv._window_ask_lines(call):
        fails.append(('ask_dedup', 'resent within the epoch'))
    # (1) a REFUSAL advances the epoch
    drv.window_stats['none'] += 1
    out2 = srv._window_ask_lines(call)
    if not any('combat_roll 1' in ln for ln in out2):
        fails.append(('ask_after_refusal',
                      'the wjg4 wedge: refused window did not '
                      'reopen the ask epoch'))
    # (2) watchdog re-ask on proven non-execution; RAW request lines
    # with the raw roll grammar (cbt10 raw log form) are NOT a result
    n0 = len(srv.notes)
    resent = []
    for _ in range(30):
        drv.tick(['<game> Wey<~Controls.1~> 1 5 5 5 6 0516D0'])
        resent = srv._window_ask_lines(call)
        if resent:
            break
    if not any('combat_roll 1' in ln for ln in resent):
        fails.append(('watchdog_reask', resent))
    if not any('WATCHDOG re-ask' in n for n in srv.notes[n0:]):
        fails.append(('watchdog_note', srv.notes[n0:]))
    # (3) rendered-but-unharvested -> sticky alarm, NO re-ask.  The
    # result lands in the FEED FILE, read by the production reader.
    drv.window_stats['none'] += 1
    out3 = srv._window_ask_lines(call)
    if not out3:
        fails.append(('ask3_sent', out3))
    _append(feed, '<game> Wey rolls  4 ')
    n_al0 = len(drv.alarms)
    got = []
    for _ in range(30):
        drv.tick([])
        got = srv._window_ask_lines(call)
        if got:
            break
    if got:
        fails.append(('no_reask_on_rendered',
                      'double-roll hazard: re-asked despite a '
                      'rendered result %r' % got))
    if len(drv.alarms) <= n_al0             or 'DEFERRAL WEDGE' not in str(drv.alarms[-1]):
        fails.append(('wedge_alarm', drv.alarms[-2:]))
    # (3b) the tile case: a rendered tile draw refuses the re-ask; a
    # CARD draw line is not a tile result (specificity)
    tcall = types.SimpleNamespace(kind='tile', kw={})
    drv.window_stats['none'] += 1
    if not srv._window_ask_lines(tcall):
        fails.append(('tile_ask_sent', 'no hunt_tile ask'))
    _append(feed, '$Wey draws Free Strategy Card.')
    for _ in range(26):
        drv.tick([])
        if srv._window_ask_lines(tcall):
            break
    else:
        fails.append(('tile_card_line_counted',
                      'a card draw blocked the tile re-ask'))
    drv.window_stats['none'] += 1
    srv._window_ask_lines(tcall)
    _append(feed, '$Wey draws Eye.')
    n_al1 = len(drv.alarms)
    got_t = []
    for _ in range(30):
        drv.tick([])
        got_t = srv._window_ask_lines(tcall)
        if got_t:
            break
    if got_t:
        fails.append(('tile_reask_on_rendered', repr(got_t)))
    if len(drv.alarms) <= n_al1:
        fails.append(('tile_wedge_alarm', drv.alarms[-2:]))


def test_roll_ask_nonce(fails):
    """MEASURES (BL-122, wave 26): a HUMAN-side roll ask carries a
    NONCE in its published text ('roll 2 dice #1') - text is not
    identity (the same ask recurs legitimately round after round),
    so the seat answers each nonce exactly once and a re-nag of an
    answered nonce is never double-answered.  The registry entry
    carries the demanded COUNT (BL-130: the count rides the ask).
    A new epoch's identical ask mints a FRESH nonce.  The chat
    policy admits the tail; AI-side rolls (!CALL lane) stay
    un-nonced."""
    import types
    td = tempfile.mkdtemp()
    feed = os.path.join(td, 'ai_feed.txt')
    open(feed, 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=61)
    drv = srv.driver
    call = types.SimpleNamespace(kind='d6',
                                 kw={'need': 2,
                                     'side': 'FREE_PEOPLES',
                                     'reason': 'combat roll'})
    out1 = srv._window_ask_lines(call)
    asks = [ln for ln in out1 if 'roll 2 dice' in ln]
    if not asks or '#1' not in asks[0]:
        fails.append(('nonce_stamped', out1))
        return
    ent = (getattr(srv, '_live_asks', None) or {}).get(1)
    if not ent or ent.get('need') != 2 or ent.get('kind') != 'd6':
        fails.append(('nonce_registry',
                      dict(getattr(srv, '_live_asks', {}) or {})))
    # same epoch retry -> deduped, no second nonce minted
    if srv._window_ask_lines(call):
        fails.append(('nonce_dedup', 'resent within the epoch'))
    # a NEW epoch's identical ask (next round's combat) mints #2 -
    # under text-keyed identity this ask was the falsely-deduped /
    # falsely-doubled class
    drv.window_stats['none'] += 1
    out2 = srv._window_ask_lines(call)
    if not any('#2' in ln for ln in out2 if 'roll 2 dice' in ln):
        fails.append(('nonce_fresh_epoch', out2))
    # AI-side d6 stays a !CALL directive - no nonce, no registry row
    acall = types.SimpleNamespace(kind='d6',
                                  kw={'need': 1, 'side': 'SHADOW',
                                      'reason': 'hunt roll'})
    drv.window_stats['none'] += 1
    out3 = srv._window_ask_lines(acall)
    if not any('combat_roll 1' in ln for ln in out3):
        fails.append(('nonce_ai_call_lane', out3))
    if len(getattr(srv, '_live_asks', {}) or {}) > 2:
        fails.append(('nonce_ai_registered',
                      dict(srv._live_asks)))
    # the POOL ask lane (fpgame19 finding 3: 'roll 4 dice' went out
    # un-nonced) mints through the same helper; its result is the
    # LETTERS roll, never a d6 number line
    txt = srv._nonce_ask('roll 4 dice', 'action_roll', 4)
    if '#' not in txt:
        fails.append(('nonce_pool_ask', txt))
    else:
        nid = int(txt.rsplit('#', 1)[1])
        ent = srv._live_asks.get(nid) or {}
        if ent.get('kind') != 'action_roll' or ent.get('need') != 4:
            fails.append(('nonce_pool_registry', ent))
    if srv._rng_result_rendered('action_roll',
                                ['$Wey rolls W P H H .']) is None:
        fails.append(('nonce_pool_close', 'letters roll not a result'))
    if srv._rng_result_rendered('action_roll',
                                ['<game> Wey rolls  4 ']) is not None:
        fails.append(('nonce_pool_specificity',
                      'a d6 line closed a pool ask'))


def test_gesture_reply_lane(fails):
    """MEASURES (USER 2026-08-25 physical-answer lane): the seat's
    <client>/ai_gesture_request.json {token, i} is answered in the
    published human menu with the client lines that row renders to
    (the AI's own emitter on a probe apply), token-matched, the
    request file consumed, the engine untouched.  The first cut read
    the file through an unimported name, swallowed the NameError,
    deleted the request and published null - every physical answer
    timed out (fpgame23/shgame20).  An unreadable request must be a
    LOUD hold, never a null."""
    import game_logic
    import json
    from game_state import ActionDieFace
    td = tempfile.mkdtemp()
    open(os.path.join(td, 'ai_feed.txt'), 'w').close()
    srv = LiveServer(td, Side.FREE_PEOPLES, load_nets=False, seed=5)
    drv = srv.driver
    drv.phase = 'actions'
    game_logic.adopt_rolled_faces(drv.es, [(Side.SHADOW, 0,
                                            ActionDieFace.ARMY)])
    n_commits = drv.n_commits_total
    req = os.path.join(srv.client_dir, 'ai_gesture_request.json')
    with open(req, 'w') as f:
        json.dump({'token': 'pin-1', 'i': 0}, f)
    out = srv.status_menu()
    hm = out.get('human_menu', out) if isinstance(out, dict) else {}
    gr = (hm or {}).get('gesture_reply') or {}
    if gr.get('token') != 'pin-1' or not gr.get('lines'):
        fails.append(('gesture_reply', gr))
    if os.path.exists(req):
        fails.append(('gesture_request_left', 'request not consumed'))
    if drv.n_commits_total != n_commits:
        fails.append(('gesture_probe_committed', drv.n_commits_total))
    # unreadable request -> loud hold, never null
    with open(req, 'w') as f:
        f.write('{not json')
    out2 = srv.status_menu()
    hm2 = out2.get('human_menu', out2) if isinstance(out2, dict) else {}
    gr2 = (hm2 or {}).get('gesture_reply') or {}
    if not gr2.get('hold'):
        fails.append(('gesture_unreadable_silent', gr2))


def test_waiting_on_is_explicit_when_buffered(fails):
    """USER 2026-08-28: "we should never have generic 'open pending'
    etc. it needs to be explicit about what it is waiting on".  With a
    human line buffered into an open hunt decision the status names
    the side, the kind, the line, the option count and the expected
    gesture - never 'gesture match for N buffered record(s)'; once the
    stall is signalled the status carries the self-repair offer."""
    from test_live_adopt import hunt_pending
    td = tempfile.mkdtemp()
    feed = os.path.join(td, 'ai_feed.txt')
    open(feed, 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=61)
    drv = srv.driver
    hunt_pending(drv.es)
    drv.phase = 'actions'
    _append(feed, '$Wey moves Gondor Regular from Minas Tirith to '
                  'Osgiliath.')
    srv.handle({'lines': ()})
    if drv.pending_buf is None or not drv.pending_buf.records:
        fails.append(('explicit_control', 'NOT-MEASURED: nothing buffered'))
        return
    w = srv._waiting_on()
    if 'buffered record(s)' in w or 'gesture match for' in w:
        fails.append(('generic_waiting_on', w))
    for need in ("FREE_PEOPLES's", 'hunt', 'Gondor Regular',
                 'match none of its', 'legal options', 'expected:'):
        if need not in w:
            fails.append(('waiting_on_missing', (need, w)))
    for _ in range(40):
        srv.handle({'lines': ()})
        if drv._repair is not None:
            break
    w2 = srv._waiting_on()
    if drv._repair is None or 'say yes to self-repair' not in w2 \
            or 'ai_stall_' not in w2:
        fails.append(('offer_waiting_on', w2))


def test_bonus_die_prompt_reads_the_board(fails):
    """USER manual game 2026-08-28: the human crowned Aragorn, then
    pulled the new die out of FP Reinforcements at once and killed it
    with the spent Will die.  At rollover the engine grows the FP pool
    4 -> 5; the old prompt ('drag it from FP Reinforcements') would
    have sent them after the WRONG spare piece.  CONTRACT: when the
    client already shows the human's die pieces outside Reinforcements
    for the whole new pool, the AI says the die is already on the
    table; when the new die still sits in Reinforcements, the drag
    prompt stands."""
    import game_state as gs
    from game_state import ActionDie
    td = tempfile.mkdtemp()
    feed = os.path.join(td, 'ai_feed.txt')
    open(feed, 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=61)
    drv = srv.driver
    es = drv.es
    es.dice[Side.FREE_PEOPLES].append(ActionDie(
        es.dice[Side.FREE_PEOPLES][0].faces, Side.FREE_PEOPLES))
    drv.view.refresh_dump = lambda: True
    pre = {'FREE_PEOPLES': 4, 'SHADOW': len(es.dice[Side.SHADOW])}
    # case 1: the new die already on the table (killed into FP Dead)
    drv.view.piece_region = {'210': 'FP Dice', '211': 'FP Dice',
                             '212': 'FP Dice', '213': 'FP Dice',
                             '358': 'FP Dead', '359': 'FP Reinf'}
    lines = srv._bonus_dice_lines(pre)
    txt = ' '.join(str(l) for l in lines).lower()
    if 'already on the table' not in txt or 'reinforcements to' in txt:
        fails.append(('bonus_prompt_ignored_board', txt[:200]))
    # case 2: the new die still in Reinforcements -> the drag prompt
    drv.view.piece_region = {'210': 'FP Dice', '211': 'FP Dice',
                             '212': 'FP Dice', '213': 'FP Dice',
                             '358': 'FP Reinf', '359': 'FP Reinf'}
    lines = srv._bonus_dice_lines(pre)
    txt = ' '.join(str(l) for l in lines).lower()
    if 'reinforcements to fp dice' not in txt:
        fails.append(('bonus_prompt_missing', txt[:200]))


def test_waiting_on(fails):
    """MEASURES (WAVE-J, USER panel report: the new build's panel
    said NOTHING while the old one named the blocker): waiting_on is
    NEVER EMPTY while any blocker exists (the silent-stall disease on
    the display surface), and it names the engine facts of the
    blocker - the human side and the pending kind the engine's own
    describe_pending reports - on product-built states: the fresh
    turn start, a REAL rng-deferred human window (the hunt-armed
    fellowship move fed through the feed file), a REAL engine pending
    (a placed battle applied through army_attack, the human defender's
    combat-card decision), and the alternation turn.  Also pins the
    serve_handler (lines, how) tuple the transport consumes."""
    import ai
    from test_live_driver import _hunt_armed_state, _FS_WINDOW
    from test_live_engine_fixwave import field_battle_state, pick
    td = tempfile.mkdtemp()
    feed = os.path.join(td, 'ai_feed.txt')
    open(feed, 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=61)
    drv = srv.driver
    w0 = srv._waiting_on()
    if not w0:
        fails.append(('w_turnstart_empty', repr(w0)))
    # a REAL deferral: the human FP's fellowship move with an eye in
    # the Hunt Box starves on the hunt roll the client has not rolled
    drv.es = _hunt_armed_state()
    drv.hands.es = drv.es
    drv.phase = 'actions'
    _append(feed, *_FS_WINDOW)
    srv.handle({'lines': ()})
    if drv.deferred is None:
        fails.append(('w_control', 'NOT-MEASURED: the fellowship window '
                      'did not defer (%r)' % (dict(drv.window_stats),)))
        return
    w2 = srv._waiting_on()
    if not w2:
        fails.append(('w_deferred_empty', repr(w2)))
    # a REAL engine pending owned by the human: the AI's attack walked
    # to the defender's combat-card decision
    st = field_battle_state(atk_reg=3, def_reg=3)
    nc = pick(ai.get_legal_actions(st, Side.SHADOW), 'combat_card_none')
    if nc is None:
        fails.append(('w_control', 'NOT-MEASURED: no attacker decline'))
        return
    st = ai.apply_action(st, nc)
    dp = ai.describe_pending(st)
    if dp is None or dp['side'] is not Side.FREE_PEOPLES:
        fails.append(('w_control', 'NOT-MEASURED: pending %r'
                      % (dp and (dp['kind'], dp['side']),)))
        return
    td2 = tempfile.mkdtemp()
    open(os.path.join(td2, 'ai_feed.txt'), 'w').close()
    srv = LiveServer(td2, Side.SHADOW, load_nets=False, seed=61)
    drv = srv.driver
    drv.es = st
    drv.hands.es = st
    w4 = srv._waiting_on()
    if not w4 or Side.FREE_PEOPLES.name not in w4 \
            or str(dp['kind']).split(':')[-1] not in w4:
        fails.append(('w_human_pending', (w4, dp['kind'])))
    # the alternation turn: the AI just acted, the human is to act
    st2 = ai.clone_state(st)
    st2.pending_combat = None
    drv.es = st2
    drv.hands.es = st2
    drv.phase = 'actions'
    drv.last_actor = Side.SHADOW
    w5 = srv._waiting_on()
    if not w5 or Side.FREE_PEOPLES.name not in w5:
        fails.append(('w_alternation', w5))
    out = srv.serve_handler({'lines': ()})
    if not (isinstance(out, tuple) and len(out) == 2
            and isinstance(out[1], str) and out[1]):
        fails.append(('w_serve_tuple', repr(out)[:120]))


def test_window_rng_deferral_ask(fails):
    """MEASURES (PHASE3-D port, mechanism 1 at the SERVER seam): a
    deferred human window starving on a d6 the client has not rolled
    yet makes the server ORDER it — '!CALL combat_roll 1' for the
    AI(Shadow)-rolled hunt die — exactly ONCE per commit epoch
    (SPECIFICITY: an idle retry tick re-surfaces the ask and the
    epoch dedup returns nothing; a re-sent roll the client DID
    execute would roll twice).  The '<game> ... rolls' line then
    commits the move and the published progress payload drops
    'deferred'."""
    import game_logic
    from game_state import ActionDieFace
    td = tempfile.mkdtemp()
    feed = os.path.join(td, 'ai_feed.txt')
    open(feed, 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=13)
    drv = srv.driver
    # armed position: FP C die adopted, one eye in the Hunt Box, no
    # FP hand (the fellowship move is the only rng-starving candidate)
    game_logic.adopt_rolled_faces(
        drv.es, [(Side.FREE_PEOPLES, 0, ActionDieFace.CHARACTER)])
    game_logic.allocate_eyes(drv.es, 1, indices=[0])
    drv.es.hands[Side.FREE_PEOPLES].hand[:] = []
    drv.phase = 'actions'
    _append(feed,
            '$Wey moves Free Action Die (C) from FP Dice to Current '
            'Action.',
            '$Wey moves FSP from Fellowship Track 0 to Fellowship '
            'Track 1.',
            '$Wey moves Free Action Die (C) from Current Action to '
            'The Hunt For The Ring.')
    out1 = srv.handle({'lines': ()})
    if [s for s in out1 if 'combat_roll' in s] != ['!CALL combat_roll 1']:
        fails.append(('ask_ordered', repr(out1)))
        return
    prog = srv.status_menu().get('progress', {})
    if prog.get('deferred') is not True:
        fails.append(('ask_status_deferred', prog))
    # idle retry: the SAME starving ask must not be re-ordered
    out2 = srv.handle({'lines': ()})
    if any('combat_roll' in s for s in out2):
        fails.append(('ask_dedup', repr(out2)))
    # the client's roll settles the deferral transactionally
    _append(feed, '<game> Wey rolls 1')
    srv.handle({'lines': ()})
    if drv.es.fellowship.progress != 1:
        fails.append(('ask_commit', drv.es.fellowship.progress))
    prog = srv.status_menu().get('progress', {})
    if prog.get('deferred') is not False \
            or prog.get('window_stats', {}).get('matched') != 1:
        fails.append(('ask_status_settled', prog))


def test_draw_action_single_call(fails):
    """MEASURES (WAVE-H cbt1 live reproduction): the AI's plain
    draw_card ACTION must order the client draw exactly ONCE.  The
    emission's staged prologue ('!CALL draw_sa_strat') and _dir_lines
    answering the starved draw funnel BOTH ordered it - the client
    drew TWO cards per action (cbt1: n_alarms=2 'EXCESS RANDOMNESS at
    step end: {cards: 1}', SA hand one card ahead of the engine).
    Also asserts the die kill is OWED (not packed pre-resolution) and
    flushes after the commit."""
    import ai
    import game_logic
    from game_state import ActionDieFace
    td = tempfile.mkdtemp()
    feed = os.path.join(td, 'ai_feed.txt')
    open(feed, 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=23)
    drv = srv.driver
    game_logic.adopt_rolled_faces(
        drv.es, [(Side.SHADOW, 0, ActionDieFace.EVENT)])
    drv.phase = 'actions'
    drv.last_actor = Side.FREE_PEOPLES

    # scripted pick: the draw action (netless chooser takes slot [0])
    def chooser(state, side, fp, sh, num_simulations=0,
                temperature=0.0, add_noise=False):
        with srv.rand.pin():
            menu = ai.get_legal_actions(state, side)
        act = next((a for a in menu if a.action_type == 'draw_card'),
                   None)
        return act, None
    orig = ai.mcts_action_with_policy
    ai.mcts_action_with_policy = chooser
    srv.fp_net = srv.sh_net = 'SCRIPT'
    try:
        out1 = srv.handle({'lines': ()})
        calls = [s for s in out1 if s.startswith('!CALL draw_')]
        if len(calls) != 1:
            fails.append(('draw_double_call', repr(out1)))
            return
        # idle retry must not re-order either
        out2 = srv.handle({'lines': ()})
        if any(s.startswith('!CALL draw_') for s in out2):
            fails.append(('draw_reorder_on_retry', repr(out2)))
        # the client's draw arrives; the step commits; ONE card only
        n0 = len(drv.es.hands[Side.SHADOW].hand)
        srv.rand.feed('card', ('SHADOW', calls[0].endswith('strat')
                               and 'strategy' or 'character',
                               (drv.es.sh_strat_deck[-1]
                                if calls[0].endswith('strat')
                                else drv.es.sh_char_deck[-1])))
        out3 = srv.handle({'lines': ()})
        if len(drv.es.hands[Side.SHADOW].hand) != n0 + 1:
            fails.append(('draw_commit_count',
                          len(drv.es.hands[Side.SHADOW].hand) - n0))
        if drv.alarms:
            fails.append(('draw_excess_alarm', repr(drv.alarms)))
        # the owed die kill flushed once the chain closed
        kills = [s for s in out3 if '<~Controls.644~>' in s]
        if len(kills) != 1:
            fails.append(('draw_die_kill_flush', repr(out3)))
    finally:
        ai.mcts_action_with_policy = orig


def test_draw_action_extra_draw(fails):
    """MEASURES two things on one board (Palantir of Orthanc on the
    Shadow table, an Event die spent on draw_card):
    (A) R6 (USER): the Palantir grants on Event-die event PLAYS only,
        never on a DRAW - the draw_card action orders exactly ONE
        '!CALL draw_sa_strat', commits after ONE card (hand +1), one
        owed die kill, no alarms, no second order ever;
    (B) the WAVE-K wkg1 mechanism (HARD STALL 121s): a step that
        legitimately needs the SAME directive twice ('draw N' effects,
        a second hunt tile) re-orders it only after the previous order
        was FULFILLED - measured on _dir_lines with the product's own
        fulfilment generation: same ask twice idle = one order; after
        the client delivers, the identical ask orders again; idle again
        = no third order (the cbt1 double-draw property stays)."""
    import ai
    import game_logic
    from wotr_bridge.live import randomness as rnd
    from game_state import ActionDieFace, PersistentCard,         PersistentCardType
    td = tempfile.mkdtemp()
    feed = os.path.join(td, 'ai_feed.txt')
    open(feed, 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=29)
    drv = srv.driver
    game_logic.adopt_rolled_faces(
        drv.es, [(Side.SHADOW, 0, ActionDieFace.EVENT)])
    drv.phase = 'actions'
    drv.last_actor = Side.FREE_PEOPLES
    drv.es.persistent_cards.add_card(PersistentCard(
        card_id='SHADOW_Character_The_Palantir_of_Orthanc',
        card_name='The Palantir of Orthanc', side=Side.SHADOW,
        card_type=PersistentCardType.PALANTIR_OF_ORTHANC))

    def chooser(state, side, fp, sh, num_simulations=0,
                temperature=0.0, add_noise=False):
        with srv.rand.pin():
            menu = ai.get_legal_actions(state, side)
        act = next((a for a in menu if a.action_type == 'draw_card'
                    and str(a.params.get('deck')) == 'strategy'), None)
        return act, None
    orig = ai.mcts_action_with_policy
    ai.mcts_action_with_policy = chooser
    srv.fp_net = srv.sh_net = 'SCRIPT'
    try:
        out1 = srv.handle({'lines': ()})
        calls1 = [s for s in out1 if s.startswith('!CALL draw_')]
        if calls1 != ['!CALL draw_sa_strat']:
            fails.append(('extra_draw_first_call', repr(out1)))
            return
        # idle retry = no re-order
        out1b = srv.handle({'lines': ()})
        if any(s.startswith('!CALL draw_') for s in out1b):
            fails.append(('extra_draw_idle_reorder', repr(out1b)))
        n0 = len(drv.es.hands[Side.SHADOW].hand)
        c1 = drv.es.sh_strat_deck[-1]
        srv.rand.feed('card', ('SHADOW', 'strategy', c1))
        out2 = srv.handle({'lines': ()})
        # (A) R6: ONE draw, the step commits on it
        if any(s.startswith('!CALL draw_') for s in out2):
            fails.append(('r6_palantir_granted_on_draw',
                          'a second draw was ordered on an Event-die '
                          'DRAW: %r' % (out2,)))
        if len(drv.es.hands[Side.SHADOW].hand) != n0 + 1:
            fails.append(('r6_draw_commit_count',
                          len(drv.es.hands[Side.SHADOW].hand) - n0))
        if srv._open_step is not None:
            fails.append(('r6_step_not_committed',
                          repr(srv._open_step[0])))
        if drv.alarms:
            fails.append(('extra_draw_alarms', repr(drv.alarms)))
        kills = [s for s in out2 if '<~Controls.644~>' in s]
        if len(kills) != 1:
            fails.append(('extra_draw_die_kill_flush', repr(out2)))
    finally:
        ai.mcts_action_with_policy = orig
    # (B) the same directive twice in ONE step, fulfilment-keyed
    srv.rand.begin_attempt()
    call = rnd.NeedClientCall('draws', items=[('SHADOW', 'strategy')])
    o1 = srv._dir_lines(call)
    if o1 != ['!CALL draw_sa_strat']:
        fails.append(('gen_first_order', repr(o1)))
    if srv._dir_lines(call):
        fails.append(('gen_idle_reorder', 'same ask re-ordered idle'))
    srv.rand.feed('card', ('SHADOW', 'strategy',
                           drv.es.sh_strat_deck[-1]))
    o2 = srv._dir_lines(call)
    if o2 != ['!CALL draw_sa_strat']:
        fails.append(('gen_second_order_missing',
                      'the wkg1 wedge: identical ask not re-ordered '
                      'after fulfilment: %r' % (o2,)))
    if srv._dir_lines(call):
        fails.append(('gen_third_order', 'idle retry re-ordered'))
    srv._abandon_step('test teardown')


def test_combat_card_kill(fails):
    """MEASURES (K1 / R4 two-phase combat cards + the WAVE-K R4 owed
    kill) through handle() on a real battle specimen, the AI (Shadow)
    attacking:
    (1) the AI's combat_card_play commits HIDDEN: '!CALL choose_card
        <pid>' only - no play_card, no 'cc:' chat naming the card
        while the human defender has not committed (W1#44);
    (2) NO reveal while only one side has committed;
    (3) the human's 'nc' completes the round's commitments -> exactly
        ONE '!CALL reveal_chosen' (cards revealed TOGETHER), and it
        precedes the round's roll ask in the same RESP;
    (4) the AI's own reveal echo line arrives -> no second reveal;
    (5) the owed kill: NO kill while the battle is open; after the
        combat pending CLOSES exactly one kill of that pid, ledger
        empty, never twice."""
    import ai
    import test_live_emission as TE
    cur, at = TE._drive_to_battle()
    if at is None:
        fails.append(('cck_control', 'NOT-MEASURED: no attack'))
        return
    c = ai.apply_action(cur, at)
    got = TE._walk_to(c, 'combat_card_play')
    if got is None:
        fails.append(('cck_control',
                      'NOT-MEASURED: no combat_card_play reachable'))
        return
    state, acts, side, kind = got
    if side is not Side.SHADOW:
        fails.append(('cck_control', 'NOT-MEASURED: first combat card '
                      'decision is not the AI side (%r)' % (side,)))
        return
    ccp = next(x for x in acts if x.action_type == 'combat_card_play')
    cid = ccp.params.get('card_id')
    td = tempfile.mkdtemp()
    feed = os.path.join(td, 'ai_feed.txt')
    open(feed, 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=31)
    drv = srv.driver
    drv.es = state
    drv.hands.es = state
    drv.phase = 'actions'
    drv.view.card_pid_map[cid] = '183'

    def chooser(st, sd, fp, sh, num_simulations=0, temperature=0.0,
                add_noise=False):
        return ccp, None
    orig = ai.mcts_action_with_policy
    ai.mcts_action_with_policy = chooser
    srv.fp_net = srv.sh_net = 'SCRIPT'

    def _k183(ln):
        return ('183 ' in ln and 'Game.2035' in ln
                and ('<~Controls.644~>' in ln
                     or '<~Controls.296~>' in ln))
    try:
        out1 = srv.handle({'lines': ()})
        if not any('<~Controls.840~>183' in ln for ln in out1):
            fails.append(('cck_choose_call', repr(out1)))
            return
        if any('play_card' in ln for ln in out1)                 or any('cc:' in ln for ln in out1):
            fails.append(('cck_identity_leaked_at_commit', repr(out1)))
        if list(drv.owed_card_kills) != [('SHADOW', '183', cid)]:
            fails.append(('cck_banked', repr(drv.owed_card_kills)))
            return
        # (2) one side committed: no reveal, no kill
        out2 = srv.handle({'lines': ()})
        if any('reveal_chosen' in ln for ln in out1 + out2):
            fails.append(('cck_premature_reveal', repr(out1 + out2)))
        if any(_k183(ln) for ln in out1 + out2):
            fails.append(('cck_premature_kill', repr(out1 + out2)))
        if drv.es.pending_combat is None:
            fails.append(('cck_control', 'battle closed unexpectedly'))
            return
        dp = ai.describe_pending(drv.es)
        if dp is None or dp['side'] is not Side.FREE_PEOPLES                 or not str(dp['kind']).endswith('combat_card'):
            fails.append(('cck_control', 'NOT-MEASURED: the defender '
                          'card decision is not open: %r'
                          % (dp and dp['kind'],)))
            return
        # (3) the human declines -> reveal together, once, before
        # the roll ask
        _append(feed, '<Wey> nc')
        out3 = srv.handle({'lines': ()})
        rev = [i for i, ln in enumerate(out3) if 'reveal_chosen' in ln]
        if len(rev) != 1:
            fails.append(('cck_reveal_once', 'reveal directives: %d in '
                          '%r notes=%r' % (len(rev), out3,
                                           srv.notes[-3:])))
            return
        rolls = [i for i, ln in enumerate(out3)
                 if 'combat_roll' in ln]
        if rolls and rolls[0] < rev[0]:
            fails.append(('cck_reveal_after_roll', repr(out3)))
        # (4) the AI's own reveal echo: no second reveal
        _append(feed, '<auto> moves The Shadow is Moving from Shadow '
                      'Hand (chosen) to Current SA Card.')
        out4 = srv.handle({'lines': ()})
        if any('reveal_chosen' in ln for ln in out4):
            fails.append(('cck_reveal_twice', repr(out4)))
        if any(_k183(ln) for ln in out3 + out4):
            fails.append(('cck_premature_kill_2', repr(out3 + out4)))
        # (5) the combat pending CLOSES (battle over) - engine statement
        drv.es.pending_combat = None
        drv.chat_deferred = None
        drv.deferred = None
        out5 = srv.handle({'lines': ()})
        kills = [ln for ln in out5 if _k183(ln)]
        if len(kills) != 1:
            fails.append(('cck_kill_at_close', repr(out5)))
        if drv.owed_card_kills:
            fails.append(('cck_ledger_cleared', repr(drv.owed_card_kills)))
        out6 = srv.handle({'lines': ()})
        if any(_k183(ln) for ln in out6):
            fails.append(('cck_double_kill', repr(out6)))
    finally:
        ai.mcts_action_with_policy = orig


def test_combat_d6_harvest_after_prefix(fails):
    """MEASURES (SCENARIO_DEFECTS D1, real-client reproductions
    `ai_attacks_human_defends_t7` and `attack_human_basic_t1`): in a
    ONE-CLIENT game every draw AND every '<game> X rolls' line carries
    the local seat's nick, so the roller-nick binding can never
    attribute a d6 - the driver's rule (test_live_driver bind_poison)
    poisons a nick seen drawing for both sides to UNTAGGED.  The
    prefix lane bound the same nick LAST-WRITER-WINS, the flip carried
    it into live, and every combat roll of the OTHER side starved
    forever (side-lane selection in randomness._consume_d6) -> the
    DEFERRAL WEDGE alarm.  ONE answerer: the prefix lane binds through
    driver._bind_roller_nick.
    (1) both pinned fixtures' prefixes leave the client nick POISONED
        (bound, value None);
    (2) the K1 battle specimen with the prefix lane's bindings carried
        over (the flip's contract): the AI's hidden commit, the
        human's 'nc', the reveal echo, then BOTH roll lines exactly as
        the real feed renders them ('<game> Wey rolls  1 3 2 4 3 ' for
        the attacker, the defender's line answering the AI's 'roll N
        dice' ask) are harvested into the round - nothing fed is left
        unconsumed, the defender ask is answered, and 30 idle ticks
        raise NO DEFERRAL WEDGE."""
    import re
    import ai
    import test_live_emission as TE
    from wotr_bridge.live import scenario as scen
    lab = os.path.join(os.path.dirname(_ROOT), 'WOTR_AI_training_lab',
                       'scenarios')
    bindings = {}
    for name in ('ai_attacks_human_defends', 'attack_human_basic'):
        fx = os.path.join(lab, name)
        if not os.path.exists(os.path.join(fx, 'scenario.json')):
            fails.append(('d1_control', 'NOT-MEASURED: fixture %s '
                          'missing' % name))
            return
        sc = scen.Scenario(fx)
        pdrv = scen.replay_prefix(sc.feed_path, sc.cut, sc.ai_side_name,
                                  seed=42, meta_path=sc.meta_path,
                                  expected=sc.prefix)
        probs = pdrv.prefix_problems()
        if probs:
            fails.append(('d1_control', 'NOT-MEASURED: prefix lane '
                          'unclean for %s: %r' % (name, probs[:3])))
            return
        ns = dict(pdrv.nick_side)
        bindings[name] = ns
        drew = [n for n in ns if n]
        if not drew:
            fails.append(('d1_control', 'NOT-MEASURED: no roller nick '
                          'bound by the %s prefix' % name))
            return
        bad = {n: s for n, s in ns.items() if s is not None}
        if bad:
            fails.append(('d1_prefix_nick_tagged',
                          '%s: the one-client nick left SIDE-TAGGED by '
                          'the prefix lane: %r (every later <game> '
                          'roll line of the other side starves)'
                          % (name, bad)))
    # (2) the composed live consequence on the K1 battle specimen
    cur, at = TE._drive_to_battle()
    if at is None:
        fails.append(('d1_control', 'NOT-MEASURED: no attack'))
        return
    c = ai.apply_action(cur, at)
    got = TE._walk_to(c, 'combat_card_play')
    if got is None:
        fails.append(('d1_control',
                      'NOT-MEASURED: no combat_card_play reachable'))
        return
    state, acts, side, kind = got
    if side is not Side.SHADOW:
        fails.append(('d1_control', 'NOT-MEASURED: first combat card '
                      'decision is not the AI side (%r)' % (side,)))
        return
    ccp = next(x for x in acts if x.action_type == 'combat_card_play')
    cid = ccp.params.get('card_id')
    td = tempfile.mkdtemp()
    feed = os.path.join(td, 'ai_feed.txt')
    open(feed, 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=31)
    drv = srv.driver
    drv.es = state
    drv.hands.es = state
    drv.phase = 'actions'
    drv.view.card_pid_map[cid] = '183'
    # the flip's contract: the prefix lane's bookkeeping IS the live
    # driver's (ScenarioDriver is one object) - the human-FP fixture's
    # bindings, whose seat layout this specimen shares
    drv.nick_side = dict(bindings['ai_attacks_human_defends'])

    def chooser(st, sd, fp, sh, num_simulations=0, temperature=0.0,
                add_noise=False):
        return ccp, None
    orig = ai.mcts_action_with_policy
    ai.mcts_action_with_policy = chooser
    srv.fp_net = srv.sh_net = 'SCRIPT'
    try:
        out = srv.handle({'lines': ()})
        if not any('<~Controls.840~>183' in ln for ln in out):
            fails.append(('d1_control', 'NOT-MEASURED: no hidden commit '
                          '%r' % (out,)))
            return
        _append(feed, '<Wey> nc')
        out = srv.handle({'lines': ()})
        if not any('reveal_chosen' in ln for ln in out) \
                or not any('combat_roll 5' in ln for ln in out):
            fails.append(('d1_control', 'NOT-MEASURED: no reveal + '
                          'attacker roll ask: %r' % (out,)))
            return
        _append(feed, '<auto> moves The Shadow is Moving from Shadow '
                      'Hand (chosen) to Current SA Card.')
        srv.handle({'lines': ()})
        # the attacker's 5 dice, rendered under the LOCAL nick
        _append(feed, '<game> Wey rolls  1 3 2 4 3 ')
        out = srv.handle({'lines': ()})
        ask = next((ln for ln in out
                    if re.search(r'roll \d+ (die|dice)', ln)), None)
        if ask is None:
            fails.append(('d1_attacker_roll_not_harvested',
                          'no defender roll ask after the attacker '
                          'result: out=%r notes=%r unconsumed=%r'
                          % (out, srv.notes[-3:], srv.rand.unconsumed())))
            return
        need = int(re.search(r'roll (\d+) (die|dice)', ask).group(1))
        p = srv.rand.pending()
        if p is None or p.kind != 'd6' \
                or str(p.kw.get('side')) != str(Side.FREE_PEOPLES):
            fails.append(('d1_control', 'NOT-MEASURED: the open ask is '
                          'not the defender d6: %r' % (p and p.kw,)))
            return
        # the defender's dice: the human's answer, same nick
        faces = ' '.join(str(4 if i % 2 == 0 else 1) for i in range(need))
        _append(feed, '<game> Wey rolls  %s ' % faces)
        n_al0 = len(drv.alarms)
        for _ in range(30):
            srv.handle({'lines': ()})
        wedge = [a for a in drv.alarms[n_al0:] if 'DEFERRAL WEDGE' in a]
        if wedge:
            fails.append(('d1_defender_roll_not_harvested', wedge[0]))
        left = srv.rand.unconsumed().get('d6', 0)
        if left:
            fails.append(('d1_d6_left_unconsumed',
                          '%d defender face(s) fed, never consumed; '
                          'pending=%r' % (left, srv.rand.pending()
                                          and srv.rand.pending().kw)))
        p2 = srv.rand.pending()
        if p2 is not None and p2.kind == 'd6' \
                and str(p2.kw.get('side')) == str(Side.FREE_PEOPLES) \
                and p2.kw.get('ctx') == 'combat roll':
            fails.append(('d1_defender_ask_still_open', repr(p2.kw)))
    finally:
        ai.mcts_action_with_policy = orig


def _exercise_human_casualty_followthrough(fails, downgrade=False):
    """Literal dice drive the composed stack through attacker-first losses.

    The attacker loses one Regular; then the defender either loses its
    two-Regular garrison and Leader, or its lone Elite becomes a Regular.
    This is a clientless protocol test, not native Java acceptance.
    """
    import re
    import ai
    import test_live_emission as TE
    cur, at = TE._drive_to_battle()
    if at is None:
        fails.append(('caw_wipe_control', 'NOT-MEASURED: no attack'))
        return
    if downgrade:
        import game_state as gs
        garrison = ai._armies_lookup(cur, 'fords of isen')
        garrison.units = [gs.Unit(gs.Nation.ROHAN, gs.UnitKind.ELITE)]
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
    if downgrade:
        # Explicit client setup for the synthetic Elite-garrison variant.
        drv.view.piece_region.update({'073': 'fords of isen',
                                      '069': 'FP Reinforcements',
                                      '070': 'FP Reinforcements'})

    def chooser(st, sd, fp, sh, num_simulations=0, temperature=0.0,
                add_noise=False):
        d = ai.describe_pending(st, sd)
        acts = d['actions']
        a = next((x for x in acts if x.action_type in
                  ('combat_card_none', 'casualty_allocate')), acts[0])
        return a, None
    orig = ai.mcts_action_with_policy
    ai.mcts_action_with_policy = chooser
    srv.fp_net = srv.sh_net = 'SCRIPT'

    def step(line=None, n=1):
        if line is not None:
            _append(feed, line)
        out = []
        for _ in range(n):
            out.extend(srv.handle({'lines': ()}))
        return out
    try:
        step('$Wey moves Shadow Action Die (A) from SA Dice to '
             'Current Action.')
        step('<Wey> fords of isen')
        step('<Wey> nc', 2)
        step('<game> Wey rolls  6 1 1 1 1 ' if downgrade
             else '<game> Wey rolls  6 6 1 1 1 ', 2)
        step('<game> Wey rolls  6 ' if downgrade
             else '<game> Wey rolls  6 1 ', 3)
        if not downgrade:
            step('<game> Wey rolls  1 ', 3)      # leader reroll: miss
        step(None, 3)
        dp = ai.describe_pending(drv.es)
        if dp is None or dp['kind'] != 'combat:casualty_allocation'                 or dp['side'] is not Side.SHADOW:
            fails.append(('caw_wipe_control',
                          'NOT-MEASURED: expected the human casualty '
                          'pending, got %r' % (dp and dp['kind'],)))
            return
        if len(dp['actions']) < 2:
            fails.append(('caw_wipe_control', 'NOT-MEASURED: no '
                          'kill-vs-downgrade choice'))
            return
        defender_before = ai._armies_lookup(drv.es, 'fords of isen')
        expected_kinds = ['ELITE'] if downgrade else ['REGULAR', 'REGULAR']
        if sorted(u.kind.name for u in defender_before.units) != expected_kinds:
            fails.append(('caw_defender_changed_before_attacker',
                          [u.kind.name for u in defender_before.units]))
        if len(defender_before.leaders) != 1 or drv.echo_ledger:
            fails.append(('caw_premature_defender_echo', drv.echo_ledger))
        # Attacker allocation happens first. The defender's forced losses
        # are emitted only after this human choice commits.
        casualty_out = step('$Wey kills Isengard Regular in Orthanc.', 4)
        dp2 = ai.describe_pending(drv.es)
        if drv.pending_buf is not None and drv.pending_buf.records:
            fails.append(('caw_wipe_unmatched',
                          [r.get('norm', '')[:60]
                           for r in drv.pending_buf.records]))
        expected_pending = ('combat:attacker_continue' if downgrade
                            else 'combat:victor_advance')
        if dp2 is None or dp2['kind'] != expected_pending:
            fails.append(('caw_wipe_advance', dp2 and dp2['kind']))
        orth = drv.es.armies.get('Orthanc') or drv.es.armies.get(
            'orthanc')
        n_reg = sum(1 for u in orth.units if u.kind.name == 'REGULAR')
        if n_reg != 3:
            fails.append(('caw_wipe_kill_applied', n_reg))
        if downgrade:
            expected_lines = [
                '$WOTR_AI_FP<~Controls.296~>073 <~Controls.297~><~Game.1918~>',
                '$WOTR_AI_FP<~Scribe.141~>290 <~Scribe.142~><~Game.2038~>'
                '<~Scribe.143~><~Game.1918~><~Scribe.144~>']
            if sorted(casualty_out) != sorted(expected_lines):
                fails.append(('caw_downgrade_emission', casualty_out))
            defender_after = ai._armies_lookup(drv.es, 'fords of isen')
            if [u.kind.name for u in defender_after.units] != ['REGULAR']:
                fails.append(('caw_downgrade_not_applied', defender_after.units))
            commits = drv.n_commits_total
            echo_out = step('$WOTR_AI_FP kills Rohan Elite in Fords of Isen.', 2)
            echo_out += step('$WOTR_AI_FP moves Rohan Regular from FP '
                             'Reinforcements to Fords of Isen.', 2)
            echo_out += step(None, 3)
            if echo_out or drv.echo_ledger or drv.n_commits_total != commits:
                fails.append(('caw_downgrade_duplicate_after_echo',
                              (echo_out, drv.echo_ledger, drv.n_commits_total - commits)))
            if drv.alarms:
                fails.append(('caw_downgrade_echo_alarm', drv.alarms))
        else:
            # Fixed piece IDs from the board roster: Fords' Rohan Leader
            # 074 and Regulars 069/070, region Game.1918.
            expected_lines = [
                '$WOTR_AI_FP<~Controls.296~>074 <~Controls.297~><~Game.1918~>',
                '$WOTR_AI_FP<~Controls.296~>069 070 <~Controls.297~><~Game.1918~>']
            if sorted(casualty_out) != sorted(expected_lines):
                fails.append(('caw_forced_emission', casualty_out))
            commits = drv.n_commits_total
            echo_out = step('$WOTR_AI_FP kills group ([2]Rohan Regular) '
                            'in Fords of Isen.', 2)
            echo_out += step('$WOTR_AI_FP kills Rohan Leader in Fords '
                             'of Isen.', 2)
            echo_out += step(None, 3)
            if echo_out or drv.echo_ledger or drv.n_commits_total != commits:
                fails.append(('caw_duplicate_after_echo',
                              (echo_out, drv.echo_ledger, drv.n_commits_total - commits)))
            if ai._armies_lookup(drv.es, 'fords of isen') is not None:
                fails.append(('caw_defender_not_wiped', True))
            if drv.alarms:
                fails.append(('caw_echo_alarm', drv.alarms))
    finally:
        ai.mcts_action_with_policy = orig


def test_human_casualty_after_ai_wipe_matches(fails):
    _exercise_human_casualty_followthrough(fails)


def test_human_casualty_before_forced_ai_downgrade(fails):
    _exercise_human_casualty_followthrough(fails, downgrade=True)


def test_forced_casualty_delta_is_exact_and_opponent_only(fails):
    """Literal delta controls reject unvoiced losses beyond a forced stage."""
    import ai
    import game_state as gs
    import test_live_emission as TE
    from wotr_bridge.live import gesture_match as gm
    state, _ = TE._drive_to_battle()
    state.pending_combat = {
        'phase': 'casualty_allocation', 'alloc_stage': 'attacker',
        'attacker_side': Side.SHADOW, 'defender_side': Side.FREE_PEOPLES,
        'attacker_region': 'Orthanc', 'defender_region': 'Fords of Isen',
        'deciding_side': Side.SHADOW, 'atk_hits': 1, 'def_hits': 2}
    forced = gm._forced_defender_allocation_rows(state, Side.SHADOW)
    expected = {('fords of isen', 'rohan:regular'): -2}
    if forced != expected:
        fails.append(('forced_wipe_witness', forced))
    own = ('orthanc', 'isengard:regular')
    win = gm.WindowFacts()
    win.armies[own] = -1
    eng = gm.EngFacts()
    eng.armies.update({own: -1, **expected})
    if not gm._covers(win, eng, combat_flux=True, forced_army_rows=forced):
        fails.append(('forced_wipe_not_covered', True))
    if gm._covers(win, eng, combat_flux=True):
        fails.append(('spectator_hidden_loss_accepted', True))
    bad_rows = [
        {own: -2, **expected},
        {own: -1, ('fords of isen', 'rohan:regular'): -3},
        {own: -1, ('fords of isen', 'rohan:elite'): -2},
        {own: -1, ('edoras', 'rohan:regular'): -2},
        {own: -1, **expected, ('dale', 'the north:regular'): -1}]
    for rows in bad_rows:
        bad = gm.EngFacts()
        bad.armies.update(rows)
        if gm._covers(win, bad, combat_flux=True, forced_army_rows=forced):
            fails.append(('unvoiced_loss_accepted', rows))
    # A forced one-hit downgrade is a signed -Elite/+Regular delta.
    army = ai._armies_lookup(state, 'fords of isen')
    army.units = [gs.Unit(gs.Nation.ROHAN, gs.UnitKind.ELITE)]
    state.pending_combat['def_hits'] = 1
    forced = gm._forced_defender_allocation_rows(state, Side.SHADOW)
    expected = {('fords of isen', 'rohan:elite'): -1,
                ('fords of isen', 'rohan:regular'): 1}
    if forced != expected:
        fails.append(('forced_downgrade_witness', forced))
    if army.units[0].kind is not gs.UnitKind.ELITE:
        fails.append(('witness_mutated_live_state', True))
    # A defender with a real Regular-versus-Elite choice is never hidden.
    army.units.append(gs.Unit(gs.Nation.ROHAN, gs.UnitKind.REGULAR))
    if gm._forced_defender_allocation_rows(state, Side.SHADOW):
        fails.append(('defender_choice_hidden', True))
    state.pending_combat['def_hits'] = 0
    if gm._forced_defender_allocation_rows(state, Side.SHADOW):
        fails.append(('unsolicited_defender_loss_allowed', True))


def test_combat_roll_through_chat_attack_window(fails):
    """MEASURES (SCENARIO_DEFECTS D1, second root cause, real-client
    reproductions `attack_human_basic_t1/u1/u2`): when the HUMAN is the
    attacker, staging the action die opens a segmenter window that the
    committed chat-attack keeps open until the terminating die kill (at
    battle END).  The round's own '<game> X rolls' d6 line arrives
    WHILE that window is open and the segmenter buffers it as an
    'update' - it never reaches the d6 funnel, so the round starves
    forever (the AI-attacker seat never hits this: the AI's staging is
    a self-echo that opens no window).  ONE fix: a committed
    chat-attack window is transparent to its live combat traffic - an
    'update' record routes through the phase handler NOW; only the die
    kill (a 'closed' event, never an 'update') stays window content.
    Driven through the composed stack with a REAL segmenter window
    (the staged-die feed line the client renders), not a hand-set
    pending: the attacker's 5 d6 are harvested, the AI then asks for
    its own defender dice, and 30 idle ticks raise NO wedge."""
    import re
    import ai
    import test_live_emission as TE
    from emitter import render
    cur, at = TE._drive_to_battle()
    if at is None:
        fails.append(('caw_control', 'NOT-MEASURED: no attack'))
        return
    # the staged die's pool pid, so the client stage line binds it
    from wotr_bridge.live import client_view
    td = tempfile.mkdtemp()
    feed = os.path.join(td, 'ai_feed.txt')
    open(feed, 'w').close()
    srv = LiveServer(td, Side.FREE_PEOPLES, load_nets=False, seed=31)
    drv = srv.driver
    drv.es = cur
    drv.hands.es = cur
    drv.phase = 'actions'
    drv.last_actor = None
    # the flip's poisoned one-client nick binding (D1 first root cause)
    drv.nick_side = {'Wey': None}

    def chooser(st, sd, fp, sh, num_simulations=0, temperature=0.0,
                add_noise=False):
        d = ai.describe_pending(st, sd)
        acts = d['actions']
        a = next((x for x in acts
                  if x.action_type == 'combat_card_none'), acts[0])
        return a, None
    orig = ai.mcts_action_with_policy
    ai.mcts_action_with_policy = chooser
    srv.fp_net = srv.sh_net = 'SCRIPT'
    try:
        # the human stages a Shadow action die (opens a segmenter
        # window), then declares the attack by chat with it staged
        _append(feed, '$Wey moves Shadow Action Die (A) from SA Dice '
                      'to Current Action.')
        srv.handle({'lines': ()})
        if drv.seg.window is None:
            fails.append(('caw_control', 'NOT-MEASURED: the staged die '
                          'did not open a segmenter window'))
            return
        _append(feed, '<Wey> fords of isen')
        srv.handle({'lines': ()})
        if getattr(drv, '_chat_attack_window', None) is None:
            fails.append(('caw_control', 'NOT-MEASURED: the chat attack '
                          'did not commit (no _chat_attack_window)'))
            return
        _append(feed, '<Wey> nc')          # human attacker: no card
        out = srv.handle({'lines': ()})
        out += srv.handle({'lines': ()})
        # the AI FP defender commits nc and asks the human to roll the
        # attacker's 5 dice
        ask = next((ln for ln in out
                    if re.search(r'roll \d+ (die|dice)', ln)), None)
        if ask is None:
            fails.append(('caw_control', 'NOT-MEASURED: no attacker roll '
                          'ask: %r' % (out,)))
            return
        need = int(re.search(r'roll (\d+) (die|dice)', ask).group(1))
        p = srv.rand.pending()
        if p is None or str(p.kw.get('side')) != str(Side.SHADOW):
            fails.append(('caw_control', 'NOT-MEASURED: open ask is not '
                          'the attacker d6: %r' % (p and p.kw,)))
            return
        # the roll renders WHILE the die window is still open
        if drv.seg.window is None:
            fails.append(('caw_control', 'NOT-MEASURED: the die window '
                          'closed before the roll'))
            return
        faces = ' '.join('4' for _ in range(need))
        _append(feed, '<game> Wey rolls  %s ' % faces)
        n_al0 = len(drv.alarms)
        for _ in range(30):
            srv.handle({'lines': ()})
        wedge = [a for a in drv.alarms[n_al0:] if 'WEDGE' in a]
        if wedge:
            fails.append(('caw_roll_not_harvested', wedge[0]))
        # the attacker roll was consumed -> the AI now needs its OWN
        # defender dice (the pending advanced off the SHADOW ask)
        p2 = srv.rand.pending()
        if p2 is not None and p2.kind == 'd6' \
                and str(p2.kw.get('side')) == str(Side.SHADOW) \
                and p2.kw.get('ctx') == 'combat roll':
            fails.append(('caw_attacker_ask_still_open',
                          'the attacker 5 d6 were buffered in the die '
                          'window, never funneled: %r' % (p2.kw,)))
    finally:
        ai.mcts_action_with_policy = orig


def test_ai_defender_casualty_absorption_exact(fails):
    """MEASURES (fpgame9 F14/19/29/30, BL-88): the AI SHADOW defender
    holding elites absorbs EXACTLY the announced hits.  The live
    tester measured 1 hit -> 2 figures, 3 -> 6, and a downgrade on 0
    announced hits, in EVERY elite-holding round (all-regular rounds
    were exact).  Metric: the side's absorption capacity
    (2*elites + regulars) must drop by exactly the attacker's hits
    (here 1: one 6 in the city round, defender all-miss)."""
    import re
    import ai
    from test_live_engine_fixwave import (fresh_state as _fs88, place,
                                          set_dice, Face)
    from game_state import Nation
    st = _fs88()
    st.political.boxes[Nation.SAURON] = 3
    st.political.boxes[Nation.GONDOR] = 3
    place(st, 'west harondor', Nation.GONDOR, regulars=4, clear=True)
    place(st, 'pelargir', Nation.SAURON, regulars=2, elites=2,
          clear=True)
    st.hands[Side.SHADOW].hand = []
    st.hands[Side.FREE_PEOPLES].hand = []
    set_dice(st, Side.FREE_PEOPLES, [Face.ARMY])
    set_dice(st, Side.SHADOW, [Face.CHARACTER])

    def _absorb(st9):
        a = next((ar for k, ar in st9.armies.items()
                  if str(k).strip().lower() == 'pelargir'), None)
        if a is None:
            return 0
        e = sum(1 for u in a.units
                if u.nation == Nation.SAURON
                and u.kind.name == 'ELITE')
        r = sum(1 for u in a.units
                if u.nation == Nation.SAURON
                and u.kind.name == 'REGULAR')
        return 2 * e + r
    td = tempfile.mkdtemp()
    feed = os.path.join(td, 'ai_feed.txt')
    open(feed, 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=45)
    drv = srv.driver
    drv.es = st
    drv.hands.es = st
    drv.phase = 'actions'
    drv.last_actor = None
    drv.nick_side = {'Wey': None}
    # a minimal CLIENT board for the armies (real pid namespace) so
    # the AI's own casualty rows can render and the census helpers
    # have client facts
    import log_replay as lr
    _sau_e, _sau_r, _gon_r = [], [], []
    for _i in range(1, 500):
        _pid = '%03d' % _i
        try:
            _nat, _ut = lr.get_unit_type_from_piece_id(_pid, {})
        except Exception:
            continue
        _nn = str(getattr(_nat, 'value', _nat)).lower()
        _uu = str(getattr(_ut, 'value', _ut)).lower()
        if _nn == 'sauron' and _uu == 'elite':
            _sau_e.append(_pid)
        elif _nn == 'sauron' and _uu == 'regular':
            _sau_r.append(_pid)
        elif _nn == 'gondor' and _uu == 'regular':
            _gon_r.append(_pid)
    if len(_sau_e) < 2 or len(_sau_r) < 2 or len(_gon_r) < 4:
        fails.append(('bl88_control', 'pid map: %d/%d/%d'
                      % (len(_sau_e), len(_sau_r), len(_gon_r))))
        return
    for _pid in _sau_e[:2] + _sau_r[:2]:
        drv.view.piece_region[_pid] = 'Pelargir'
    for _pid in _gon_r[:4]:
        drv.view.piece_region[_pid] = 'West Harondor'
    pre88 = _absorb(drv.es)
    if pre88 != 6:
        fails.append(('bl88_control', 'setup absorption %d' % pre88))
        return

    def chooser(st9, sd, fp, sh, num_simulations=0, temperature=0.0,
                add_noise=False):
        d = ai.describe_pending(st9, sd)
        acts = d['actions']
        a = next((x for x in acts
                  if x.action_type == 'combat_card_none'), acts[0])
        return a, None
    orig = ai.mcts_action_with_policy
    ai.mcts_action_with_policy = chooser
    srv.fp_net = srv.sh_net = 'SCRIPT'
    out_all = []

    def _tick(n=1):
        for _ in range(n):
            out_all.extend(srv.handle({'lines': ()}))
    try:
        _append(feed, '$Wey moves Free Action Die (A) from FP Dice to '
                      'Current Action.')
        _tick()
        if drv.seg.window is None:
            fails.append(('bl88_control', 'no window from the staged '
                                          'FP die'))
            return
        _append(feed, '<Wey> pelargir from west harondor')
        _tick(2)
        if not isinstance(getattr(drv.es, 'pending_combat', None),
                          dict):
            fails.append(('bl88_control', 'chat attack did not open a '
                                          'battle'))
            return
        _append(feed, '<Wey> nc')
        _tick(2)
        for _ in range(12):
            p = srv.rand.pending()
            if p is not None and getattr(p, 'kind', '') == 'd6':
                m = next((mm for ln in reversed(out_all)
                          for mm in [re.search(
                              r'roll (\d+) (die|dice)', str(ln))]
                          if mm), None)
                need = int(m.group(1)) if m else 1
                is_atk = str(p.kw.get('side')) \
                    == str(Side.FREE_PEOPLES)
                faces = (['6'] + ['1'] * (need - 1)) if is_atk \
                    else ['1'] * need
                _append(feed, '<game> Wey rolls  %s '
                        % ' '.join(faces))
            _tick(2)
            if _absorb(drv.es) != pre88 \
                    and ai.describe_pending(drv.es) is None:
                break
        _tick(2)
        post88 = _absorb(drv.es)
        hits_line = next((str(ln) for ln in out_all
                          if re.search(r'\d+ hits? for', str(ln))),
                         None)
        if pre88 - post88 != 1:
            fails.append(('bl88_over_removal',
                          'absorption %d -> %d (want -1); announce=%r'
                          % (pre88, post88, hits_line)))
    finally:
        ai.mcts_action_with_policy = orig


def test_cc_pick_commit_reveal_prompt(fails):
    """MEASURES (shgame10 E1 blocker): the human ATTACKER commits a
    combat card through the IDENTITY lane ('pick N' against the
    published say_pick).  The engine applies the card and the round
    resolves, but the physical card is still in the client hand: the
    demand chain must hold end-to-end - (1) the owed-reveal marker is
    set by the pick commit (BL-82 one-writer), (2) cc_reveal_drag is
    prompted so the seat performs the drag (live: neither happened
    visibly and the hand census split, 'engine=0 client=1')."""
    import re
    import ai
    import game_logic as gl
    import test_live_emission as TE
    cur, at = TE._drive_to_battle()
    if at is None:
        fails.append(('e1_control', 'NOT-MEASURED: no attack'))
        return
    for _ in range(5):
        try:
            gl.draw_event_card(cur, Side.SHADOW, 'character')
            gl.draw_event_card(cur, Side.SHADOW, 'strategy')
        except Exception:
            break
    td = tempfile.mkdtemp()
    feed = os.path.join(td, 'ai_feed.txt')
    open(feed, 'w').close()
    srv = LiveServer(td, Side.FREE_PEOPLES, load_nets=False, seed=41)
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
        a = next((x for x in acts
                  if x.action_type == 'combat_card_none'), acts[0])
        return a, None
    orig = ai.mcts_action_with_policy
    ai.mcts_action_with_policy = chooser
    srv.fp_net = srv.sh_net = 'SCRIPT'
    out_all = []

    def _tick(n=1):
        for _ in range(n):
            out_all.extend(srv.handle({'lines': ()}))
    try:
        _append(feed, '$Wey moves Shadow Action Die (A) from SA Dice '
                      'to Current Action.')
        _tick()
        _append(feed, '<Wey> fords of isen')
        _tick(2)
        menu = (srv.status_menu() or {}).get('human_menu') or {}
        row = next((e for e in (menu.get('actions') or ())
                    if e.get('action_type') == 'combat_card_play'
                    and (e.get('params') or {}).get('say_pick')), None)
        if row is None:
            fails.append(('e1_control', 'NOT-MEASURED: no pickable '
                          'combat_card_play row published: %r'
                          % (menu.get('pending'),)))
            return
        _append(feed, '<Wey> pick %s' % row['params']['say_pick'])
        _tick(2)
        if not (getattr(drv, '_cc_reveal_owed', None) or {}):
            fails.append(('e1_marker_never_set',
                          (getattr(drv, 'chat_deferred', None),
                           menu.get('pending'))))
        # satisfy every d6 ask until the round resolves (idle ticks
        # in between - the sweep must get its chances)
        for _ in range(10):
            p = srv.rand.pending()
            if p is not None and getattr(p, 'kind', '') == 'd6':
                ask = next((m for ln in reversed(out_all)
                            for m in [re.search(
                                r'roll (\d+) (die|dice)', str(ln))]
                            if m), None)
                need = int(ask.group(1)) if ask else 1
                _append(feed, '<game> Wey rolls  %s '
                        % ' '.join('4' for _ in range(need)))
            _tick(2)
        _tick(3)
        if not any('reveal your committed combat card' in str(ln)
                   for ln in out_all):
            fails.append(('e1_no_reveal_prompt',
                          {'owed': getattr(drv, '_cc_reveal_owed',
                                           None),
                           'pc_open': getattr(drv.es, 'pending_combat',
                                              None) is not None,
                           'tail': [str(l)[:70]
                                    for l in out_all[-6:]]}))
    finally:
        ai.mcts_action_with_policy = orig


def test_ai_defender_combat_card_reveals(fails):
    """MEASURES (SCENARIO_DEFECTS D3, real-client reproduction
    `attack_human_basic_u3`): when the AI is the DEFENDER and plays its
    own combat card, `_ai_decide` emits '!CALL choose_card <pid>' but
    the engine apply_action(combat_card_play) STARVES on the round's
    rolls (NeedClientCall rolls back the post), so the engine state
    never carries `def_combat_card`.  The reveal sweep gated its 'both
    committed' test on the engine card key alone, so '!CALL
    reveal_chosen' never went out: the client hand stayed one card over
    the engine's (WITNESS ESCALATION) and the card never reached the
    table.  ONE fix: a side counts as committed when its commit is
    DEFERRED too (the AI's own choose_card, symmetric with the human's
    chat_deferred), scoped to the side that owns it.
    (1) AI defender + human attacker done -> exactly one reveal_chosen;
    (2) AI ATTACKER with the defender NOT yet committed -> NO reveal
        (the _ai_cc_chosen-alone regression this fix must avoid)."""
    import ai
    import game_logic as gl
    import test_live_emission as TE
    cur, at = TE._drive_to_battle()
    if at is None:
        fails.append(('rev_control', 'NOT-MEASURED: no attack'))
        return
    for _ in range(4):
        gl.draw_event_card(cur, Side.FREE_PEOPLES, 'character')
        gl.draw_event_card(cur, Side.FREE_PEOPLES, 'strategy')
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
    for cid in cur.hands[Side.FREE_PEOPLES].hand:
        drv.view.card_pid_map[cid] = '1%02d' % (
            len(drv.view.card_pid_map) + 1)
    picked = {'play': False}

    def chooser(st, sd, fp, sh, num_simulations=0, temperature=0.0,
                add_noise=False):
        d = ai.describe_pending(st, sd)
        acts = d['actions']
        ccp = next((x for x in acts
                    if x.action_type == 'combat_card_play'), None)
        if ccp is not None:
            picked['play'] = True
            return ccp, None
        return acts[0], None
    orig = ai.mcts_action_with_policy
    ai.mcts_action_with_policy = chooser
    srv.fp_net = srv.sh_net = 'SCRIPT'
    try:
        _append(feed, '$Wey moves Shadow Action Die (A) from SA Dice '
                      'to Current Action.')
        srv.handle({'lines': ()})
        _append(feed, '<Wey> fords of isen')
        srv.handle({'lines': ()})
        _append(feed, '<Wey> nc')          # human attacker: no card
        out = []
        for _ in range(4):
            out += srv.handle({'lines': ()})
        if not picked['play']:
            fails.append(('rev_control', 'NOT-MEASURED: the AI defender '
                          'had no combat_card_play to pick'))
            return
        if not any('<~Controls.840~>' in ln for ln in out):
            fails.append(('rev_control', 'NOT-MEASURED: the AI defender '
                          'did not commit a card: %r' % (out,)))
            return
        reveals = [ln for ln in out if 'reveal_chosen' in ln]
        if len(reveals) != 1:
            fails.append(('rev_ai_defender_reveal',
                          'the AI defender chose a card but reveal_'
                          'chosen went out %d time(s) (D3: client hand '
                          'diverges, card never reaches the table): %r'
                          % (len(reveals), out)))
    finally:
        ai.mcts_action_with_policy = orig


def test_ai_attacker_no_premature_reveal(fails):
    """MEASURES (the regression the D3 fix must not cause): the AI as
    ATTACKER commits its combat card FIRST; the reveal must wait for
    the human defender.  _ai_cc_chosen alone would reveal immediately -
    the deferred-commit term is scoped to the owning side, so the
    attacker's choice does not satisfy the defender's slot."""
    import ai
    import test_live_emission as TE
    cur, at = TE._drive_to_battle()
    if at is None:
        fails.append(('prem_control', 'NOT-MEASURED: no attack'))
        return
    c = ai.apply_action(cur, at)
    got = TE._walk_to(c, 'combat_card_play')
    if got is None:
        fails.append(('prem_control',
                      'NOT-MEASURED: no combat_card_play reachable'))
        return
    state, acts, side, kind = got
    if side is not Side.SHADOW:
        fails.append(('prem_control', 'NOT-MEASURED: first combat card '
                      'decision is not the AI side (%r)' % (side,)))
        return
    ccp = next(x for x in acts if x.action_type == 'combat_card_play')
    cid = ccp.params.get('card_id')
    td = tempfile.mkdtemp()
    feed = os.path.join(td, 'ai_feed.txt')
    open(feed, 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=31)
    drv = srv.driver
    drv.es = state
    drv.hands.es = state
    drv.phase = 'actions'
    drv.view.card_pid_map[cid] = '183'

    def chooser(st, sd, fp, sh, num_simulations=0, temperature=0.0,
                add_noise=False):
        return ccp, None
    orig = ai.mcts_action_with_policy
    ai.mcts_action_with_policy = chooser
    srv.fp_net = srv.sh_net = 'SCRIPT'
    try:
        out = srv.handle({'lines': ()})     # AI attacker commits
        out += srv.handle({'lines': ()})    # idle: human not committed
        if any('reveal_chosen' in ln for ln in out):
            fails.append(('prem_reveal',
                          'the AI attacker revealed before the human '
                          'defender committed: %r' % (out,)))
    finally:
        ai.mcts_action_with_policy = orig


def test_ai_fp_turn_start(fails):
    """MEASURES (WAVE-K R2 / PARITY-AUDIT-2: the AI-as-FREE_PEOPLES
    seat of the turn-start pipeline, offline through handle()): the
    human SHADOW draws -> the AI orders ITS FP draws -> the AI's own
    DECLARATION decision (declined here via the scripted chooser: a
    decline renders nothing) -> the AI says 'fp ok' (outbound kind) ->
    the human's '1e' chat allocates ONE eye -> the human's SA roll is
    adopted and the AI orders '!CALL roll_action_dice Game.2042' (G11:
    Shadow first) -> the FP roll lands -> actions, with the FREE PEOPLES
    (= the AI) acting FIRST.  No alarms.  Every step is AI=FP-gated, so
    this leg is the seat's only coverage."""
    import ai
    td = tempfile.mkdtemp()
    feed = os.path.join(td, 'ai_feed.txt')
    open(feed, 'w').close()
    srv = LiveServer(td, Side.FREE_PEOPLES, load_nets=False, seed=37)
    drv = srv.driver

    def chooser(state, side, fp, sh, num_simulations=0,
                temperature=0.0, add_noise=False):
        with srv.rand.pin():
            dp = ai.describe_pending(state, side)
            acts = (dp['actions'] if dp is not None
                    else ai.get_legal_actions(state, side))
        pa = next((a for a in acts if a.action_type == 'pass'), None)
        return (pa or (acts[0] if acts else None)), None
    orig = ai.mcts_action_with_policy
    ai.mcts_action_with_policy = chooser
    srv.fp_net = srv.sh_net = 'SCRIPT'
    try:
        _append(feed, '$Wey draws Shadow Character Card.',
                '$Wey draws Shadow Strategy Card.')
        out1 = srv.handle({'lines': ()})
        if '!CALL draw_fp_char' not in out1:
            fails.append(('aifp_draws', repr(out1)))
            return
        # the AI's draws are ordered ONE per starve (char, then strat)
        srv.rand.feed('card', ('FREE_PEOPLES', 'character',
                               drv.es.fp_char_deck[-1]))
        out1b = srv.handle({'lines': ()})
        if '!CALL draw_fp_strat' not in out1b:
            fails.append(('aifp_draws_strat', repr(out1b)))
            return
        srv.rand.feed('card', ('FREE_PEOPLES', 'strategy',
                               drv.es.fp_strat_deck[-1]))
        out2 = srv.handle({'lines': ()})
        out2 += srv.handle({'lines': ()})
        if not any(str(x).strip().endswith('fp ok') for x in out2):
            fails.append(('aifp_fp_ok', 'no fp ok: %r phase=%r open=%r'
                          % (out2, drv.phase, srv._open_step)))
            return
        if drv.phase != 'eyes':
            fails.append(('aifp_phase_eyes', drv.phase))
        if getattr(drv.es, '_declare_phase', False):
            fails.append(('aifp_declare_flag_cleared', True))
        # USER doctrine (chat only for what pieces cannot show): the
        # human Shadow's '1e' chat is REFUSED with the drag it stands
        # for and allocates NOTHING (the real session
        # build_isengard_war_sa_rec2: '1e' allocated engine-side, the
        # client still rolled 7 dice - 'adds 0 eyes' - roster desync)
        _append(feed, '<Wey> 1e')
        out3 = srv.handle({'lines': ()})

        def _eyes():
            return sum(1 for d in drv.es.dice[Side.SHADOW]
                       if d.current is not None
                       and getattr(d.current, 'name', '') == 'EYE')
        if _eyes() != 0:
            fails.append(('aifp_eye_chat_allocated', _eyes()))
        if not any('eyes go on the board' in str(x) for x in out3):
            fails.append(('aifp_eye_chat_refusal', repr(out3)))
        # the physical allocation (corpus form: the SA Dice -> Hunt box
        # drag under the local nick) is the ONE allocation path
        _append(feed, '$Wey moves Shadow Action Die (E) from SA Dice '
                      'to The Hunt For The Ring.')
        srv.handle({'lines': ()})
        if _eyes() != 1:
            fails.append(('aifp_human_eye_drag', _eyes()))
        _append(feed, '$Wey rolls A C H M P A .')
        out4 = srv.handle({'lines': ()})
        if '!CALL roll_action_dice Game.2042' not in out4:
            fails.append(('aifp_fp_roll_ordered',
                          'out=%r phase=%r' % (out4, drv.phase)))
            return
        _append(feed, '$Wey rolls C A P M .')
        out5 = srv.handle({'lines': ()})
        if drv.phase != 'actions':
            fails.append(('aifp_phase_actions', drv.phase))
        # FP acts FIRST = the AI's own action is the action phase's
        # first commit (it ran inside this very cycle: last commit label
        # 'ai:action', an emission in out5) - and the turn then passes
        # to the human Shadow
        last_lbl = drv.commits[-1][2] if drv.commits else None
        if not (out5 and str(last_lbl).startswith('ai:'))                 and srv._open_step is None:
            fails.append(('aifp_fp_acts_first',
                          'no AI first action: out=%r last_commit=%r '
                          'open=%r' % (out5, last_lbl, srv._open_step)))
        with srv.rand.pin():
            nxt = ai._next_acting_side(drv.es, drv.last_actor)
        if str(last_lbl).startswith('ai:') and nxt is not Side.SHADOW:
            fails.append(('aifp_alternation', repr(nxt)))
        if drv.alarms:
            fails.append(('aifp_alarms', repr(drv.alarms)))
    finally:
        ai.mcts_action_with_policy = orig


def test_ai_fp_random_casualty(fails):
    """MEASURES (WAVE-K K7, wkf2 LIVE WEDGE - AI-as-FP turn 1): the AI's
    OWN hunt decision 'hunt_sacrifice_random' must (1) order the
    client's random companion draw (the randomcomp funnel: chat
    'random companion draw?' + '!CALL random_companion') with NO
    emission hold, and (2) once the draw is fed, COMMIT and render the
    victim's physical fate (the guide-sacrifice grammar: a drag of the
    companion piece to FP Casualties or onto the map).  Before the fix:
    'no L2 planner for hunt_sacrifice_random' halted the step forever."""
    import ai
    td = tempfile.mkdtemp()
    feed = os.path.join(td, 'ai_feed.txt')
    open(feed, 'w').close()
    srv = LiveServer(td, Side.FREE_PEOPLES, load_nets=False, seed=41)
    drv = srv.driver
    drv.phase = 'actions'
    drv.es.pending_hunt = {'phase': 'hunt_sacrifice', 'damage': 1}
    names0 = set(str(getattr(c, 'name', c))
                 for c in drv.es.fellowship.companions)
    if 'Boromir' not in names0:
        fails.append(('rc_control', 'NOT-MEASURED: Boromir not in the '
                      'fellowship'))
        return

    def chooser(state, side, fp, sh, num_simulations=0,
                temperature=0.0, add_noise=False):
        with srv.rand.pin():
            dp = ai.describe_pending(state, side)
        acts = dp['actions'] if dp is not None else []
        a = next((x for x in acts
                  if x.action_type == 'hunt_sacrifice_random'), None)
        return (a or (acts[0] if acts else None)), None
    orig = ai.mcts_action_with_policy
    ai.mcts_action_with_policy = chooser
    srv.fp_net = srv.sh_net = 'SCRIPT'
    try:
        from wotr_bridge.live.driver import DriverHalt
        try:
            out1 = srv.handle({'lines': ()})
        except DriverHalt as e:
            fails.append(('rc_halt', 'the random-casualty step HALTED: '
                          '%s' % str(e)[:160]))
            return
        if '!CALL random_companion' not in out1:
            fails.append(('rc_ask', 'no random companion ask: %r notes=%r'
                          % (out1, srv.notes[-2:])))
            return
        if any('HOLD' in str(n) for n in srv.notes[-3:]):
            fails.append(('rc_hold', repr(srv.notes[-3:])))
        srv.rand.feed('companion', ['Boromir'])
        try:
            out2 = srv.handle({'lines': ()})
        except DriverHalt as e:
            fails.append(('rc_halt_post', 'the post-draw render HALTED: '
                          '%s' % str(e)[:160]))
            return
        names1 = set(str(getattr(c, 'name', c))
                     for c in drv.es.fellowship.companions)
        if 'Boromir' in names1:
            fails.append(('rc_commit', 'Boromir still in the fellowship: '
                          'out=%r open=%r notes=%r'
                          % (out2, srv._open_step, srv.notes[-3:])))
            return
        from wotr_bridge.live import undo_model
        pid = undo_model.resolve_named('Boromir')
        if pid is not None and not any(str(pid) in ln for ln in out2):
            fails.append(('rc_physical', 'no physical line for Boromir '
                          '(pid %s): %r' % (pid, out2)))
        if srv._open_step is not None:
            fails.append(('rc_step_open', repr(srv._open_step[0])))
        if drv.alarms:
            fails.append(('rc_alarms', repr(drv.alarms)))
    finally:
        ai.mcts_action_with_policy = orig


def test_side_header_guard(fails):
    """MEASURES: a LIVE_<side> request header that CONTRADICTS the
    server's configured AI side is REFUSED (no RESP) and the refusal
    reaches the panel status - with a sensitivity control proving the
    MATCHING header is still served.

    REPRODUCTION (manual-play delivery, 2026-08-20, scratchpad
    repro_sidehdr.py against the unchanged module): the client's
    'Play vs AI...' dialog asks the HUMAN which side the AI plays and
    that click is independent of the launcher's --side /
    -Dai.feed.aiside.  A server built with --side SA answered a
    LIVE_FREE_PEOPLES request with '!CALL draw_sa_char' and no
    complaint.  The rule-8 fail-closed #HAND check does NOT catch it
    (every #HAND line IS blinded - the WRONG side's), so the human's
    own hand would have been fully readable to the AI.  This leg goes
    RED if the guard is removed."""
    import json
    from wotr_bridge.live.live_server import build

    def _cycle(header):
        td = tempfile.mkdtemp()
        with open(os.path.join(td, 'ai_feed.txt'), 'w',
                  encoding='utf-8') as f:
            f.write('$Wey draws Free Character Card.\n'
                    '$Wey draws Free Strategy Card.\n')
        srv, bridge = build(td, 'SA', load_nets=False, seed=7)
        with open(os.path.join(td, 'ai_bridge_request.log'), 'w',
                  encoding='utf-8') as f:
            f.write(header + '\n$Wey<~Controls.1~>setup noise line\n')
        try:
            bridge.serve(srv.handle, max_cycles=3)
        finally:
            bridge.release_lock()
        rp = os.path.join(td, 'ai_bridge_move.txt')
        txt = (open(rp, encoding='utf-8').read()
               if os.path.exists(rp) else '')
        sp = os.path.join(td, 'ai_bridge_status.json')
        st = (json.load(open(sp, encoding='utf-8'))
              if os.path.exists(sp) else {})
        return txt, st

    txt, st = _cycle('LIVE_FREE_PEOPLES')
    if txt.strip():
        fails.append(('sidehdr_served_mismatch', repr(txt[:160])))
    if st.get('state') != 'error':
        fails.append(('sidehdr_panel_state', repr(st.get('state'))))
    if 'SIDE MISMATCH' not in str(st.get('waiting_on', '')):
        fails.append(('sidehdr_panel_text',
                      repr(str(st.get('waiting_on'))[:160])))

    # SENSITIVITY CONTROL: the matching header still gets served, and
    # the panel is NOT painted with an error (a guard that refuses
    # everything would pass the assertions above).
    txt2, st2 = _cycle('LIVE_SHADOW')
    if '!CALL draw_sa_char' not in txt2:
        fails.append(('sidehdr_control_starved', repr(txt2[:160])))
    if st2.get('state') == 'error':
        fails.append(('sidehdr_control_errored',
                      repr(str(st2.get('waiting_on'))[:160])))


def test_human_table_card_killed_in_its_slot(fails):
    """MEASURES: a play-on-table card the HUMAN seats is watched in the
    feed, and when the ENGINE discards it the live stack kills that
    piece IN THE SLOT it was seated in -- driven through the real feed
    + sidecar so driver._meta_track does the booking.

    REPRODUCTION (USER 2026-08-20): live_server._table_book was written
    ONLY by the AI's own seat line, so a card the human played was
    never booked and NOTHING removed it from their board when the
    engine dropped it.  Measured across all 14 play-on-table cards:
    14/14 removed when the AI seated them, 0/14 when the human did.
    USER's fix: "you can see where the human player places the table
    card, so just keep track of that location and kill it there."

    Also pins the SLOT: emitter.render normalises every kill to
    DICE_USED, which made the mop-up say 'kills X in Current Action'
    for a card sitting in Board3.  The corpus's own grammar for this
    act is 'kills <card> in Board<N>' (6,686 lines)."""
    import json
    td = tempfile.mkdtemp()
    feed = os.path.join(td, 'ai_feed.txt')
    meta = feed + '.meta.jsonl'
    open(feed, 'w').close()
    open(meta, 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=11)
    drv = srv.driver

    import game_state as _gs
    drv.es.persistent_cards.add_card(_gs.PersistentCard(
        card_id='FP_HORN', card_name='Horn of Gondor',
        side=Side.FREE_PEOPLES,
        card_type=_gs.PersistentCardType.HORN_OF_GONDOR))

    _append(feed, '$Wey moves Horn of Gondor/Mighty Attack '
                  'from Current FP Card to Board3.')
    with open(meta, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'seq': 1, 'gen': 1, 'nick': 'Wey',
                            'tags': [], 'pids': ['811'],
                            'ts': 0}) + '\n')
    srv.handle({'lines': ()})
    if not drv.sidecar.present:
        fails.append(('humanseat_no_sidecar',
                      'the meta channel did not open - this leg would '
                      'pass vacuously'))
        return
    # seats are keyed by CARD NAME (the one thing always visible - a
    # card played from a hidden hand has its pid redacted by the feed)
    seat = drv.table_seats.get('horn of gondor')
    if not seat or seat.get('slot') != 'Game.2031':
        fails.append(('humanseat_not_booked', repr(drv.table_seats)))
        return
    if drv.table_seat_pid('horn of gondor') != '811':
        fails.append(('humanseat_pid_unresolved', repr(seat)))
    if seat.get('armed') is not True:
        fails.append(('humanseat_not_armed',
                      'the engine holds this card, so the seat should '
                      'be armed: %r' % (seat,)))

    drv.es.persistent_cards.remove_card('FP_HORN')
    out = srv.handle({'lines': ()})
    # the kill AT a Board slot is the attested 296/297 form (W1#75)
    import emitter
    kills = [ln for ln in out if 'Controls.297' in ln
             and emitter.OPS['hand_exit_b']['re'].match(ln)]
    if not kills:
        fails.append(('humanseat_not_killed', repr(out)))
        return
    if not any('Game.2031' in ln for ln in kills):
        fails.append(('humanseat_wrong_slot', repr(kills)))
    if drv.table_seats:
        fails.append(('humanseat_book_not_cleared',
                      repr(drv.table_seats)))

    # CONTROL: a card still ON the table is NOT killed (a mop-up that
    # fired unconditionally would pass every assertion above).
    td2 = tempfile.mkdtemp()
    f2 = os.path.join(td2, 'ai_feed.txt')
    m2 = f2 + '.meta.jsonl'
    open(f2, 'w').close()
    open(m2, 'w').close()
    srv2 = LiveServer(td2, Side.SHADOW, load_nets=False, seed=11)
    srv2.driver.es.persistent_cards.add_card(_gs.PersistentCard(
        card_id='FP_HORN', card_name='Horn of Gondor',
        side=Side.FREE_PEOPLES,
        card_type=_gs.PersistentCardType.HORN_OF_GONDOR))
    _append(f2, '$Wey moves Horn of Gondor/Mighty Attack '
                'from Current FP Card to Board3.')
    with open(m2, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'seq': 1, 'gen': 1, 'nick': 'Wey',
                            'tags': [], 'pids': ['811'],
                            'ts': 0}) + '\n')
    srv2.handle({'lines': ()})
    out2 = srv2.handle({'lines': ()})
    if any('Controls.644' in ln for ln in out2):
        fails.append(('humanseat_killed_live_card', repr(out2)))

    # CONTROL 2 (USER 2026-08-20, the Tom Bombadil incident): a card
    # seated on the board that the ENGINE has NEVER held must NEVER be
    # killed -- an unarmed seat is not evidence of a discard, it is
    # evidence we have not seen the engine register the card yet.
    td3 = tempfile.mkdtemp()
    f3 = os.path.join(td3, 'ai_feed.txt')
    m3 = f3 + '.meta.jsonl'
    open(f3, 'w').close()
    open(m3, 'w').close()
    srv3 = LiveServer(td3, Side.SHADOW, load_nets=False, seed=11)
    _append(f3, '$Wey moves The Power of Tom Bombadil/Advantageous '
                'Position from Current FP Card to Board1.')
    with open(m3, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'seq': 1, 'gen': 1, 'nick': 'Wey',
                            'tags': [], 'pids': ['124'],
                            'ts': 0}) + '\n')
    # NOTE the first handle() BOOKS the seat and runs the mop-up in the
    # SAME call, so its output must be inspected too -- an earlier
    # version of this control only looked at later ticks and was
    # vacuous (it passed against the very defect it was written for).
    outs = [srv3.handle({'lines': ()})]   # engine holds NO such card
    for _ in range(4):
        outs.append(srv3.handle({'lines': ()}))
    for out3 in outs:
        if any('Controls.644' in ln for ln in out3):
            fails.append(('humanseat_killed_unconfirmed_card',
                          'a card the engine never held was killed: %r'
                          % (out3,)))
            break


def test_panel_turn_override_and_clear_error(fails):
    """MEASURES the two panel levers added 2026-08-20:

      * MY TURN / YOUR TURN -- set the driver's last_actor so the
        ENGINE's one answerer (ai._next_acting_side) names the side the
        user asked for.  The turn RULE is untouched; only the driver's
        'who last acted' bookkeeping is corrected, which is the single
        input that answerer takes from us.
      * CLEAR ERROR -- dismiss the STICKY server error state.  The
        stickiness is deliberate (a halt used to be repainted 'idle' by
        the next heartbeat and vanish), so it needs a dismiss or a
        one-off error follows the session for ever.

    Both are epoch-stamped: one press acts once."""
    import json
    import ai
    from wotr_bridge.live.live_server import build

    td = tempfile.mkdtemp()
    with open(os.path.join(td, 'ai_feed.txt'), 'w',
              encoding='utf-8') as f:
        f.write('$Wey draws Free Character Card.\n')
    srv, bridge = build(td, 'SA', load_nets=False, seed=7)
    drv = srv.driver
    drv.phase = 'actions'
    # both pools must carry ROLLED faces or _next_acting_side answers
    # None (a side with no rolled die is skipped) and the assertion
    # below would measure the empty pools, not the override
    import game_state as _gs
    for _side in (_gs.Side.FREE_PEOPLES, _gs.Side.SHADOW):
        for _d in drv.es.dice.get(_side, []):
            _d.used = False
            _d.current = _gs.ActionDieFace.ARMY

    # ---- turn override, both directions
    srv.on_control({'whose_turn': 'human', 'whose_turn_epoch': 100.0})
    if drv.last_actor is not srv.ai_side:
        fails.append(('turnovr_human_last_actor', repr(drv.last_actor)))
    srv.on_control({'whose_turn': 'ai', 'whose_turn_epoch': 101.0})
    if drv.last_actor is not srv.human_side:
        fails.append(('turnovr_ai_last_actor', repr(drv.last_actor)))

    # NOT VACUOUS: the engine answerer must actually name that side
    nxt = ai._next_acting_side(drv.es, drv.last_actor)
    if nxt is not srv.ai_side:
        fails.append(('turnovr_answerer',
                      'engine says %r acts next, expected the AI'
                      % (nxt,)))

    # an OLD epoch must not re-fire (one press acts once)
    srv.on_control({'whose_turn': 'human', 'whose_turn_epoch': 50.0})
    if drv.last_actor is not srv.human_side:
        fails.append(('turnovr_stale_epoch_applied',
                      repr(drv.last_actor)))

    # ---- the override must REFUSE while a decision is owed, and SAY
    # so (USER 2026-08-20: "clear error -> your turn does not work, it
    # still waits for my decision").  These buttons move last_actor,
    # which governs only the plain action alternation; an open pending
    # is consulted first and skipping it would answer a decision on the
    # player's behalf.
    drv.es.pending_card_option = {
        'card_id': 'X', 'card_name': 'Balrog of Moria',
        'side': 'sh', 'decider': 'sh', 'direct': True,
        'options': [{'balrog_declare_draw': True},
                    {'balrog_declare_draw': False}]}
    import ai as _ai_chk
    if _ai_chk.describe_pending(drv.es) is None:
        fails.append(('turnovr_pending_setup',
                      'no pending was opened - the guard leg below '
                      'would be vacuous'))
    else:
        before = drv.last_actor
        n_notes = len(srv.notes)
        srv.on_control({'whose_turn': 'human',
                        'whose_turn_epoch': 300.0})
        if drv.last_actor is not before:
            fails.append(('turnovr_overrode_open_pending',
                          'last_actor moved while a decision was owed'))
        if not any('IGNORED' in str(n) for n in srv.notes[n_notes:]):
            fails.append(('turnovr_silent_ignore',
                          'the override did nothing and said nothing: '
                          '%r' % (srv.notes[n_notes:],)))
    drv.es.pending_card_option = None

    # ---- sims: the panel's setting must be TAKEN and PUBLISHED
    # (USER 2026-08-20: "the mcts sim count doesn't update / acknowledge
    # my setting").  Two defects: exactly 1000 was discarded by an
    # `s != 1000` guard meant for AiOpp's spawn default (which cannot
    # reach the ADOPTED new stack), and nothing ever published `sims`
    # back, so the panel's 'thinking (N sims)' line could never show it.
    srv.on_control({'sims': 250})
    if srv.sims != 250:
        fails.append(('sims_not_taken', srv.sims))
    srv.on_control({'sims': 1000})
    if srv.sims != 1000:
        fails.append(('sims_1000_discarded',
                      'exactly 1000 was ignored: %r' % (srv.sims,)))
    published = (srv.status_menu() or {}).get('sims')
    if published != 1000:
        fails.append(('sims_not_published', repr(published)))
    # a 0/absent value is NOT a request and must not zero the setting
    srv.on_control({})
    if srv.sims != 1000:
        fails.append(('sims_cleared_by_empty_control', srv.sims))

    # ---- clear error
    ctl = os.path.join(td, 'ai_panel_control.json')
    bridge._err_note = 'BOOM: something failed'
    bridge.write_status(state='error', waiting_on='BOOM')
    with open(ctl, 'w', encoding='utf-8') as f:
        json.dump({'clear_error': 200.0}, f)
    bridge.read_control()
    if bridge._err_note:
        fails.append(('clearerr_not_cleared', bridge._err_note))
    st = json.load(open(os.path.join(td, 'ai_bridge_status.json'),
                        encoding='utf-8'))
    if st.get('state') == 'error':
        fails.append(('clearerr_status_still_error', repr(st)))

    # CONTROL: a NEW failure after a clear must re-latch (a clear that
    # disabled the error state would pass the assertions above)
    bridge._err_note = 'SECOND BOOM'
    bridge.read_control()          # same (already-consumed) epoch
    if not bridge._err_note:
        fails.append(('clearerr_swallowed_new_error',
                      'a stale clear_error epoch cleared a NEW error'))
    bridge.release_lock()


def test_panel_skip_decision(fails):
    """MEASURES the SKIP DECISION lever (USER 2026-08-20: "I want it to
    be able to skip the decision").  A stuck pending must not end the
    game, but a skip must never SPEND anything on the player's behalf:

      * a pending whose decline is an explicit action type
        (combat_card_none) is answered with it;
      * a card_option pending is answered with the option whose
        choices are ALL falsy (its decline);
      * a pending with NO decline is REFUSED, with a reason - never
        guessed at;
      * with nothing open, it says so.
    """
    import ai
    import game_state as _gs
    td = tempfile.mkdtemp()
    feed = os.path.join(td, 'ai_feed.txt')
    open(feed, 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=11)
    drv = srv.driver

    # nothing open
    items, msg = drv.skip_open_decision()
    if 'nothing to skip' not in msg:
        fails.append(('skip_idle_msg', msg))

    # a card_option pending: the decline is the all-falsy option
    drv.es.pending_card_option = {
        'card_id': 'X', 'card_name': 'Balrog of Moria',
        'side': 'sh', 'decider': 'sh', 'direct': True,
        'options': [{'balrog_declare_draw': True},
                    {'balrog_declare_draw': False}]}
    dp = ai.describe_pending(drv.es)
    if dp is None or dp['kind'] != 'card_option':
        fails.append(('skip_setup_cardopt',
                      'no card_option pending - leg would be vacuous'))
    else:
        items, msg = drv.skip_open_decision()
        if 'skipped' not in msg:
            fails.append(('skip_cardopt_refused', msg))
        if ai.describe_pending(drv.es) is not None:
            fails.append(('skip_cardopt_still_open',
                          repr(ai.describe_pending(drv.es))))
        # the DECLINE was taken, so the one-shot card is NOT spent
        pc = getattr(drv.es, 'persistent_cards', None)
        if pc is not None and pc.get_by_type(
                _gs.PersistentCardType.BALROG_OF_MORIA) is not None:
            fails.append(('skip_cardopt_spent_card',
                          'the skip chose the COMMITTING option'))

    # a pending with no decline at all must be REFUSED with a reason:
    # a REAL casualty allocation (the engine's own battle, 1 hit on a
    # 1R+1E Gondor garrison) offers only casualty_allocate options
    from test_live_engine_fixwave import field_battle_state, _to_allocation
    from game_state import Nation
    seed, post = _to_allocation(field_battle_state(
        atk_reg=5, def_nation=Nation.GONDOR, def_reg=1, def_elite=1), 1)
    dp = post is not None and ai.describe_pending(post)
    if not dp or dp['side'] is not Side.FREE_PEOPLES \
            or {a.action_type for a in dp['actions']} \
            != {'casualty_allocate'}:
        fails.append(('skip_control', 'NOT-MEASURED: no decline-free '
                      'casualty pending reached (%r)'
                      % (dp and sorted({a.action_type
                                        for a in dp['actions']}),)))
        return
    drv.es = post
    drv.hands.es = post
    items, msg = drv.skip_open_decision()
    if ai.describe_pending(drv.es) is None:
        fails.append(('skip_committing_pending_answered',
                      'the skip answered a decision with no decline'))
    if items:
        fails.append(('skip_committing_pending_acted', repr(items)))
    if not msg:
        fails.append(('skip_committing_pending_silent', repr(msg)))


def test_panel_reset_and_resync(fails):
    """MEASURES the RESET + RESYNC lever (USER 2026-08-20: "a button to
    resync and reset and clear all pending decisions, so the AI
    continues from there").

    A stuck game must come back: EVERY pending the engine's own
    answerer recognises is gone afterwards, the driver's open state is
    dropped, and the turn is handed to the AI.  Includes a pending with
    NO decline (the case SKIP DECISION deliberately refuses) - reset is
    the bigger hammer and must clear it anyway."""
    import ai
    td = tempfile.mkdtemp()
    open(os.path.join(td, 'ai_feed.txt'), 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=11)
    drv = srv.driver

    # a declinable pending AND a committing one that skip refuses
    drv.es.pending_card_option = {
        'card_id': 'X', 'card_name': 'Balrog of Moria',
        'side': 'sh', 'decider': 'sh', 'direct': True,
        'options': [{'balrog_declare_draw': True},
                    {'balrog_declare_draw': False}]}
    drv.es.pending_discard = {'side': 'fp', 'n': 1}
    drv.es.pending_guide_change = {'side': 'fp'}
    drv.last_actor = None

    # NOTE pending_guide_change here is deliberately MALFORMED (no
    # 'candidates'), so describe_pending itself raises -- the realistic
    # shape of a game that needs this button.  Reset must survive it.
    try:
        ai.describe_pending(drv.es)
    except Exception:
        pass

    items, report = drv.reset_and_resync()

    # a describe_pending that RAISES is itself a surviving pending
    # (the malformed one) - report it as such, never as a leg crash
    try:
        _left = ai.describe_pending(drv.es)
    except Exception as _e:
        _left = 'raises: %r' % (_e,)
    if _left is not None:
        fails.append(('reset_pending_survived', repr(_left)[:120]))
    for f in ('pending_card_option', 'pending_discard',
              'pending_guide_change'):
        if getattr(drv.es, f, None):
            fails.append(('reset_field_survived', f))
    if drv.last_actor is not srv.human_side:
        fails.append(('reset_turn_not_handed', repr(drv.last_actor)))
    else:
        nxt = ai._next_acting_side(drv.es, drv.last_actor)
        if nxt is not None and nxt is not srv.ai_side:
            fails.append(('reset_ai_not_next', repr(nxt)))
    if not any('resynced' in r for r in report):
        fails.append(('reset_no_resync', report))
    # EVERY step must be reported - this can erase a decision
    # mid-battle, so a silent reset would be the worst version of it
    if len(report) < 3:
        fails.append(('reset_underreported', report))

    # CONTROL: with nothing stuck, reset must still be safe and must
    # NOT invent pendings or leave the engine mid-anything
    srv2 = LiveServer(tempfile.mkdtemp(), Side.SHADOW,
                      load_nets=False, seed=11)
    open(os.path.join(srv2.client_dir, 'ai_feed.txt'), 'w').close()
    _i2, rep2 = srv2.driver.reset_and_resync()
    if ai.describe_pending(srv2.driver.es) is not None:
        fails.append(('reset_idle_created_pending', rep2))


def test_table_kill_owes_die_and_uses_watched_pid(fails):
    """MEASURES the 1.149 table kill's two live defects (USER
    2026-08-20, from their own feed):

      * the AI moved pid 145 out of the FP STRATEGY DECK because the
        planner read a STATIC pid table that claimed 145 was 'The Power
        of Tom Bombadil'; this client renders 145 as 'Kindred of
        Glorfindel'.  The target must come from the seat we WATCHED,
        resolved against the client's own dump (their dump: Board1
        holds pid 143).
      * "it should also kill its active die, which it neglected to do"
        - discard_table_card was missing from the owed-DIE-kill list,
        so only the card was banked and the spent die sat in Current
        Action for ever.  Corpus ends the act with
        `kills group (die, card) in Current Action`.
    """
    import emitter
    td = tempfile.mkdtemp()
    open(os.path.join(td, 'ai_feed.txt'), 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=11)
    drv = srv.driver

    # the human seats Tom Bombadil in Board1; the feed REDACTS its pid
    drv.view.piece_region['143'] = 'Board1'
    drv._track_table_seat(
        {'op': 'move', 'to': 'Board1',
         'obj': 'The Power of Tom Bombadil/Advantageous Position'},
        None)
    got = drv.table_seat_pid('the power of tom bombadil')
    if got != '143':
        fails.append(('tablekill_pid_from_dump',
                      'resolved %r, the client dump says 143'
                      % (got,)))
    # the static table must be GONE - it is what shipped the wrong card
    from wotr_bridge.live import emission as _em
    if hasattr(_em, '_TABLE_KILL_PID_BY_NAME'):
        fails.append(('tablekill_static_table_back',
                      'a hardcoded card-pid table is authoritative '
                      'again'))

    # the DIE must be owed alongside the card
    class _Act(object):
        action_type = 'discard_table_card'
        die_idx = 2
        params = {'card_id': 'FP_106',
                  'card_name': 'The Power of Tom Bombadil'}

    drv.owed_die_kills = []
    drv.owed_card_kills = []
    srv._note_owed_die_kill({'side': Side.SHADOW}, _Act())
    srv._note_owed_card_kill({'side': Side.SHADOW}, _Act())
    if not drv.owed_die_kills:
        fails.append(('tablekill_die_not_owed',
                      'the spent die was never banked - it strands in '
                      'Current Action'))
    else:
        ent = drv.owed_die_kills[0]
        if int(ent[1]) != 2:
            fails.append(('tablekill_wrong_die', repr(ent)))
        # ONE group kill: the card rides WITH the die, not separately
        if str(ent[2] or '') != '143':
            fails.append(('tablekill_card_not_grouped', repr(ent)))
    if drv.owed_card_kills:
        fails.append(('tablekill_card_double_banked',
                      'the card is banked twice - it would be killed '
                      'twice: %r' % (drv.owed_card_kills,)))


def test_mirror_snapshot_written(fails):
    """MEASURES: serve_handler publishes the ENGINE MIRROR feed --
    ai_live_state.pkl (unpicklable to a real GameState) and
    ai_live_status.json with the keys live_view.py reads.

    REPRODUCTION (USER 2026-08-20: "the engine mirror is not
    connected"): those two files were written ONLY by the retired
    monolith (ai_move_server._write_sided_snapshot), so against the
    new live stack the panel's ENGINE MIRROR button opened a viewer
    that polled forever on a file nobody wrote."""
    import json
    import pickle
    from wotr_bridge.live.live_server import build
    td = tempfile.mkdtemp()
    with open(os.path.join(td, 'ai_feed.txt'), 'w',
              encoding='utf-8') as f:
        f.write('$Wey draws Free Character Card.\n'
                '$Wey draws Free Strategy Card.\n')
    srv, bridge = build(td, 'SA', load_nets=False, seed=7)
    with open(os.path.join(td, 'ai_bridge_request.log'), 'w',
              encoding='utf-8') as f:
        f.write('LIVE_SHADOW\n$Wey<~Controls.1~>setup noise line\n')
    try:
        # serve_handler (NOT handle) is the production entry point -
        # the snapshot hangs off it, so a leg driving handle() alone
        # would pass while the mirror stayed dead.
        bridge.serve(srv.serve_handler, max_cycles=3)
    finally:
        bridge.release_lock()
    pkl = os.path.join(td, 'ai_live_state.pkl')
    stj = os.path.join(td, 'ai_live_status.json')
    if not os.path.exists(pkl):
        fails.append(('mirror_no_pkl', pkl))
        return
    if not os.path.exists(stj):
        fails.append(('mirror_no_status', stj))
        return
    try:
        st = pickle.load(open(pkl, 'rb'))
    except Exception as e:
        fails.append(('mirror_pkl_unloadable', repr(e)))
        return
    if getattr(st, 'turn', None) is None:
        fails.append(('mirror_pkl_not_a_state', repr(type(st))))
    d = json.load(open(stj, encoding='utf-8'))
    for k in ('mode', 'ai_side', 'turn', 'phase',
              'client_divergences', 'engine_vs_client', 'ts'):
        if k not in d:
            fails.append(('mirror_status_key', k))
    if d.get('ai_side') != 'SHADOW':
        fails.append(('mirror_ai_side', repr(d.get('ai_side'))))


def test_action_phase_waits_for_both_rolls(fails):
    """MEASURES: with the AI as SHADOW (the launcher's default), the
    action phase does NOT open while the human's FP pool is unrolled,
    the AI takes NO action in that state, and the human is ASKED to
    roll -- plus the control that the gate OPENS once they do, with
    FREE PEOPLES to act first.

    REPRODUCTION (USER 2026-08-20 live: "it's not respecting turn
    order at all"; scratchpad repro_turnorder.py).  Their feed shows
    the AI spending six Shadow dice back to back (H A C A A C) with no
    FP action between them and every FP face still None, and the
    server's own log saying 'waiting on: SHADOW to act' six times.
    MECHANISM: the engine's ONE alternation answerer counts a die as
    available only when it carries a rolled face
    (_next_acting_side -> _unused: `not d.used and d.current is not
    None`), so an UNROLLED pool is indistinguishable from an EXHAUSTED
    one; the driver opened 'actions' on whichever roll landed first
    and nothing ever asked the human to roll.  AUTO cannot show this
    (it rolls both pools together) and the one clean sided game on
    record was AI-as-FP, the branch that chains its own second roll.

    Goes RED if driver.pools_rolled() stops gating, and the roll-ask
    assertion goes RED if the chat ask is dropped again."""
    import ai
    td = tempfile.mkdtemp()
    feed = os.path.join(td, 'ai_feed.txt')
    open(feed, 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=99)
    drv = srv.driver
    _append(feed, '$Wey draws Free Character Card.',
            '$Wey draws Free Strategy Card.')
    srv.handle({'lines': ()})
    srv.rand.feed('card', ('SHADOW', 'character',
                           drv.es.sh_char_deck[-1]))
    srv.handle({'lines': ()})
    srv.rand.feed('card', ('SHADOW', 'strategy',
                           drv.es.sh_strat_deck[-1]))
    srv.handle({'lines': ()})
    _append(feed, '<Wey> fp ok')
    out = srv.handle({'lines': ()})

    # the human must be TOLD to roll their own pool
    n_fp = len([d for d in drv.es.dice[Side.FREE_PEOPLES]
                if not d.used and d.current is None])
    if not any(('roll %d dice' % n_fp) in s for s in out):
        fails.append(('roll_ask_to_human', repr(out)))

    # ONLY the AI's Shadow pool rolls
    unset = [d for d in drv.es.dice[Side.SHADOW]
             if not d.used and d.current is None]
    _append(feed, '$Wey rolls %s .' % ' '.join(['A'] * len(unset)))
    srv.handle({'lines': ()})
    if drv.phase != 'roll':
        fails.append(('opened_on_one_roll', drv.phase))
    if drv.pools_rolled():
        fails.append(('pools_rolled_wrong', 'FP pool is unrolled'))

    # NOT VACUOUS: the engine really would hand the AI the turn here.
    # If this stops being true the leg no longer guards anything.
    if ai._next_acting_side(drv.es, Side.SHADOW) is not Side.SHADOW:
        fails.append(('hazard_gone',
                      'the alternation answerer no longer skips an '
                      'unrolled pool - this leg is now vacuous'))

    used0 = sum(1 for d in drv.es.dice[Side.SHADOW] if d.used)
    for _ in range(6):
        srv.handle({'lines': ()})
    used1 = sum(1 for d in drv.es.dice[Side.SHADOW] if d.used)
    if used1 != used0:
        fails.append(('ai_acted_out_of_turn',
                      'SHADOW dice used %d -> %d with FP unrolled'
                      % (used0, used1)))

    # CONTROL: the gate must OPEN, not deadlock, and FP acts first
    fp_unset = [d for d in drv.es.dice[Side.FREE_PEOPLES]
                if not d.used and d.current is None]
    _append(feed, '$Wey rolls %s .' % ' '.join(['A'] * len(fp_unset)))
    srv.handle({'lines': ()})
    if drv.phase != 'actions':
        fails.append(('gate_deadlocked', drv.phase))
    elif ai._next_acting_side(drv.es, drv.last_actor) \
            is not Side.FREE_PEOPLES:
        fails.append(('fp_not_first',
                      repr(ai._next_acting_side(drv.es,
                                                drv.last_actor))))


def _client_json_field(txt, key):
    """AiOpp.jsonField() replicated EXACTLY, including the property
    that makes it dangerous: it returns the FIRST occurrence of the
    key anywhere in the text, with no notion of nesting.  The harness
    must ask the PRODUCT what it actually parses, not what a real JSON
    reader would."""
    i = txt.find('"%s"' % key)
    if i < 0:
        return None
    c = txt.find(':', i)
    if c < 0:
        return None
    j = c + 1
    while j < len(txt) and txt[j].isspace():
        j += 1
    if j >= len(txt):
        return None
    if txt[j] == '"':
        out = []
        k = j + 1
        while k < len(txt):
            ch = txt[k]
            if ch == '\\' and k + 1 < len(txt):
                out.append(txt[k + 1])
                k += 2
                continue
            if ch == '"':
                break
            out.append(ch)
            k += 1
        return ''.join(out)
    e = j
    while e < len(txt) and txt[e] not in ',}\r\n ':
        e += 1
    return txt[j:e]


def test_status_identity_first(fails):
    """MEASURES: the CLIENT's own parse of ai_bridge_status.json
    (AiOpp.jsonField, first-occurrence-wins) resolves 'pid' and
    'client_dir' to THIS server's identity, even though the Phase-3
    human_menu publication nests a 'pid' on every die piece.

    REPRODUCTION (manual-play smoke, 2026-08-20, real client): the
    first '"pid"' in the emitted file was a DIE id -
    '"pid": "210"' inside human_menu.dice[0] - so the client's
    adoption probe read pid 210, could not prove it owned the
    directory, and spawned a SECOND server: the RETIRED
    ai_move_server.py monolith, running alongside the live one and
    writing the same ai_bridge_move.txt for a human's game
    (ai_boot.log: 'NOT adopting pid 210 ... spawning our own',
    'launchServer: server pid=2356').  Goes RED if the identity keys
    stop being emitted first."""
    import json
    import os as _os
    from wotr_bridge.live.live_server import build
    td = tempfile.mkdtemp()
    with open(os.path.join(td, 'ai_feed.txt'), 'w',
              encoding='utf-8') as f:
        f.write('$Wey draws Free Character Card.\n'
                '$Wey draws Free Strategy Card.\n')
    srv, bridge = build(td, 'SA', load_nets=False, seed=7)
    with open(os.path.join(td, 'ai_bridge_request.log'), 'w',
              encoding='utf-8') as f:
        f.write('LIVE_SHADOW\n$Wey<~Controls.1~>setup noise line\n')
    try:
        bridge.serve(srv.handle, max_cycles=3)
    finally:
        bridge.release_lock()
    sp = os.path.join(td, 'ai_bridge_status.json')
    if not os.path.exists(sp):
        fails.append(('identity_first_nofile', sp))
        return
    txt = open(sp, encoding='utf-8').read()
    # NOT VACUOUS: this leg only measures anything while the menu
    # really does nest die pids.  If the publication ever stops
    # carrying them the leg says so instead of passing quietly.
    dice = ((json.loads(txt).get('human_menu') or {}).get('dice')
            or [])
    if not any('pid' in d for d in dice):
        fails.append(('identity_first_vacuous',
                      'no nested die "pid" in human_menu - this leg '
                      'no longer exercises the first-occurrence '
                      'hazard it was written for'))
    got = _client_json_field(txt, 'pid')
    if str(got) != str(_os.getpid()):
        fails.append(('identity_first_pid',
                      'the client parse yields %r, the server pid is '
                      '%r' % (got, _os.getpid())))
    gotdir = _client_json_field(txt, 'client_dir')
    if not gotdir or (_os.path.abspath(gotdir)
                      != _os.path.abspath(td)):
        fails.append(('identity_first_dir', repr(gotdir)))


def _script_chooser(srv, want):
    """A scripted MCTS stand-in: picks the first menu action whose
    action_type equals want['type'] (or the first non-draw action when
    want['type'] is 'other'), counting its own invocations in
    want['calls'].  The menu is the engine's own (get_legal_actions)."""
    import ai

    def chooser(state, side, fp, sh, num_simulations=0,
                temperature=0.0, add_noise=False):
        want['calls'] = want.get('calls', 0) + 1
        with srv.rand.pin():
            menu = ai.get_legal_actions(state, side)
        if want['type'] == 'other':
            act = next((a for a in menu
                        if a.action_type != 'draw_card'), None)
        else:
            act = next((a for a in menu
                        if a.action_type == want['type']), None)
        return act, None
    return chooser


def test_abandoned_ai_step(fails):
    """MEASURES (K3, W1#16/#65): an AI step abandoned mid-flight must
    leave NO residue - the randomness step closed (journal gone) and
    the server's step-scoped emission/directive state cleared - so the
    NEXT AI decision runs the chooser and emits/asks normally.  Two
    abandon paths, each on a real starved step (a draw_card action
    whose '!CALL draw_*' went out and whose card never arrived):
      (1) the panel RESET lever;
      (2) the due decision vanishing (ai_action_dp() -> None: the
          whose-turn override class).
    Defect before the fix: the stale journal replayed the OLD pick
    without running the chooser (same-keyed menu), so the AI re-ordered
    the draw it had abandoned."""
    import ai
    import game_logic
    from game_state import ActionDieFace
    td = tempfile.mkdtemp()
    feed = os.path.join(td, 'ai_feed.txt')
    open(feed, 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=23)
    drv = srv.driver
    game_logic.adopt_rolled_faces(
        drv.es, [(Side.SHADOW, 0, ActionDieFace.EVENT),
                 (Side.SHADOW, 1, ActionDieFace.ARMY),
                 (Side.SHADOW, 2, ActionDieFace.ARMY)])
    drv.phase = 'actions'
    drv.last_actor = Side.FREE_PEOPLES
    want = {'type': 'draw_card', 'calls': 0}
    orig = ai.mcts_action_with_policy
    ai.mcts_action_with_policy = _script_chooser(srv, want)
    srv.fp_net = srv.sh_net = 'SCRIPT'
    try:
        out1 = srv.handle({'lines': ()})
        if not any(s.startswith('!CALL draw_') for s in out1) \
                or srv._open_step is None or not srv._em_sent:
            fails.append(('k3_control', 'NOT-MEASURED: no starved '
                          'draw step: %r' % (out1,)))
            return
        # (1) RESET abandons the step; the AI then acts next and the
        # script now wants a plain army move (no randomness, no pending)
        want['type'] = 'army_move'
        n_calls = want['calls']
        srv._reset_requested = True
        out2 = srv.handle({'lines': ()})
        if want['calls'] != n_calls + 1:
            fails.append(('k3_reset_chooser_reran',
                          'chooser calls %d -> %d (stale journal '
                          'replayed the abandoned pick)'
                          % (n_calls, want['calls'])))
        if any(s.startswith('!CALL draw_') for s in out2):
            fails.append(('k3_reset_reordered_draw', repr(out2)))
        if any('JOURNAL MISMATCH' in str(n) for n in srv.notes):
            fails.append(('k3_reset_journal_mismatch',
                          repr(srv.notes[-3:])))
        if srv._open_step is not None:
            fails.append(('k3_reset_step_open',
                          repr(srv._open_step[0])))
        if not any(s.startswith('$') for s in out2):
            fails.append(('k3_reset_no_emission',
                          'the new decision emitted nothing: %r'
                          % (out2,)))
        # the abandoned step's staged die (engine: UNUSED) must not be
        # killed by its banked owed kill - the residue is an ALARM
        if any('<~Controls.644~>099' in s for s in out2):
            fails.append(('k3_reset_killed_unspent_die', repr(out2)))
        if not any('ABANDONED STEP RESIDUE' in str(a)
                   for a in drv.alarms):
            fails.append(('k3_reset_residue_silent', repr(drv.alarms)))
        # close the army move's continuation (its own 'pass'), then
        # (2) a fresh starved step whose due decision then VANISHES
        want['type'] = 'pass'
        srv.handle({'lines': ()})
        if ai.describe_pending(drv.es) is not None                 or drv.es.pending_army_move_die is not None:
            fails.append(('k3_control_cont', 'NOT-MEASURED: the army '
                          'move continuation did not close'))
            return
        want['type'] = 'draw_card'
        drv.phase = 'actions'
        out3 = srv.handle({'lines': ()})
        if not any(s.startswith('!CALL draw_') for s in out3) \
                or srv._open_step is None:
            fails.append(('k3_control_2', 'NOT-MEASURED: no second '
                          'starved draw step: %r notes=%r'
                          % (out3, srv.notes[-3:])))
            return
        drv.phase = 'eyes'                  # the due decision is gone
        srv.handle({'lines': ()})
        if srv._open_step is not None:
            fails.append(('k3_dpnone_step_kept', repr(srv._open_step)))
        if srv.rand.step_open or srv._em_sent or srv._dirs_sent \
                or srv._step_calls:
            fails.append(('k3_dpnone_residue',
                          'rand.step_open=%r em_sent=%r dirs=%r '
                          'calls=%r' % (srv.rand.step_open,
                                        srv._em_sent, srv._dirs_sent,
                                        srv._step_calls)))
        want['type'] = 'army_move'
        n_calls = want['calls']
        drv.phase = 'actions'
        drv.last_actor = Side.FREE_PEOPLES
        out4 = srv.handle({'lines': ()})
        if want['calls'] != n_calls + 1:
            fails.append(('k3_dpnone_chooser_reran',
                          'chooser calls %d -> %d'
                          % (n_calls, want['calls'])))
        if any(s.startswith('!CALL draw_') for s in out4):
            fails.append(('k3_dpnone_reordered_draw', repr(out4)))
        if not any(s.startswith('$') for s in out4):
            fails.append(('k3_dpnone_no_emission', repr(out4)))
    finally:
        ai.mcts_action_with_policy = orig


def test_adopt_refuses_partial_dump(fails):
    """MEASURES (W1#17): adopt_client_board must REFUSE a dump that is
    not a full, stable client read - an empty/partial dump would
    rewrite every engine region to the client's ABSENT roster (all
    armies wiped) while the census gate, diffing against the same
    dump, waved it through.  Pinned end-to-end through the RESET lever
    with NO dump file (the torn/missing-read case): the engine armies
    must survive."""
    from wotr_bridge.live import adopt, client_board
    import ai
    td = tempfile.mkdtemp()
    open(os.path.join(td, 'ai_feed.txt'), 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=11)
    drv = srv.driver
    n_units0 = sum(len(a.units) for a in drv.es.armies.values())
    n_lead0 = sum(len(a.leaders) for a in drv.es.armies.values())
    if n_units0 < 50:
        fails.append(('adopt_control', 'NOT-MEASURED: %d units at setup'
                      % n_units0))
        return
    # direct: an EMPTY dump and a PARTIAL dump are refused loudly
    for label, dump in (('empty', {}),
                        ('partial', {'001': 'Mordor', '002': 'Gondor'})):
        refused = False
        try:
            rep, nb, na = adopt.adopt_client_board(
                drv.es, dump, dynamic_types={}, ai_module=ai)
            refused = any('REFUSED' in r for r in rep)
        except adopt.AdoptionRefused:
            refused = True
        except Exception as e:
            fails.append(('adopt_%s_raised' % label, repr(e)[:120]))
        n_units = sum(len(a.units) for a in drv.es.armies.values())
        if n_units != n_units0:
            fails.append(('adopt_%s_wiped' % label,
                          'units %d -> %d' % (n_units0, n_units)))
        if not refused:
            fails.append(('adopt_%s_accepted' % label,
                          'no refusal on a %s dump' % label))
    # the census must diff against an INDEPENDENT read: a full dump
    # that disagrees with a second read of the file is refused
    full = {}
    for i in range(client_board.MIN_DUMP_PIECES + 5):
        full['%03d' % i] = 'Gorgoroth'
    dump_path = os.path.join(td, 'ai_client_board.txt')
    with open(dump_path, 'w', encoding='utf-8') as f:
        # the dump grammar: <name>\t<zone code>\t<pids>; Game.1874 =
        # Gondor - a DIFFERENT board from `full`
        for i in range(client_board.MIN_DUMP_PIECES + 5):
            f.write('p\tGame.1874\t%03d\n' % i)
    refused = False
    try:
        rep, nb, na = adopt.adopt_client_board(
            drv.es, full, dynamic_types={}, ai_module=ai,
            dump_path=dump_path)
        refused = any('REFUSED' in r for r in rep)
    except adopt.AdoptionRefused:
        refused = True
    except TypeError as e:
        fails.append(('adopt_dump_path_api', repr(e)[:120]))
    if not refused:
        fails.append(('adopt_unstable_accepted',
                      'dump disagreeing with the file was adopted'))
    if sum(len(a.units) for a in drv.es.armies.values()) != n_units0:
        fails.append(('adopt_unstable_wiped', 'armies changed'))
    # end-to-end: RESET with no stable dump leaves the armies alone
    os.remove(dump_path)
    srv._reset_requested = True
    srv.handle({'lines': ()})
    n_units = sum(len(a.units) for a in drv.es.armies.values())
    n_lead = sum(len(a.leaders) for a in drv.es.armies.values())
    if (n_units, n_lead) != (n_units0, n_lead0):
        fails.append(('adopt_reset_wiped',
                      'units %d -> %d leaders %d -> %d'
                      % (n_units0, n_units, n_lead0, n_lead)))
    if not any('REFUSED' in n or 'engine left untouched' in n
               for n in srv.notes):
        fails.append(('adopt_reset_silent', repr(srv.notes[-6:])))


def test_wwsat_card_pick_step(fails):
    """MEASURES (W1#18/#64): an AI step starving on 'card_pick' (Worn
    with Sorrow and Toil on the Shadow table, the AI-FP's random hunt
    casualty -> the engine's named funnel game_logic.pick_random_card)
    must (1) order the client's own random discard ('!CALL wwsat')
    instead of crashing every tick, and (2) COMPLETE once the client's
    'randomly discards a Free Character Card.' line renders in the
    feed - the AI's own card identity decoded from the sidecar pid via
    the commitment map (the feed producer), the card gone from the
    engine hand, no alarms, step closed."""
    import ai
    import game_logic as gl
    import json
    from game_state import PersistentCard, PersistentCardType
    td = tempfile.mkdtemp()
    feed = os.path.join(td, 'ai_feed.txt')
    open(feed, 'w').close()
    meta = feed + '.meta.jsonl'
    open(meta, 'w').close()
    srv = LiveServer(td, Side.FREE_PEOPLES, load_nets=False, seed=43)
    drv = srv.driver
    drv.phase = 'actions'
    # the FP hand holds a Character card (engine draw, depth 0 = the
    # engine's own RNG, outside the live funnels)
    gl.draw_event_card(drv.es, Side.FREE_PEOPLES, 'character')
    hand = drv.es.hands[Side.FREE_PEOPLES].hand
    if hasattr(gl, 'card_deck_of'):
        chars = [c for c in hand if gl.card_deck_of(c) == 'character']
    else:
        chars = [c for c in hand if 'Character' in c]
    if len(chars) != 1:
        fails.append(('wwsat_control', 'NOT-MEASURED: FP hand %r'
                      % (hand,)))
        return
    cid = chars[0]
    drv.view.card_pid_map[cid] = '150'
    drv.es.pending_hunt = {'phase': 'hunt_sacrifice', 'damage': 1}
    drv.es.persistent_cards.add_card(PersistentCard(
        card_id='SHADOW_Character_Worn_with_Sorrow_and_Toil',
        card_name='Worn with Sorrow and Toil', side=Side.SHADOW,
        card_type=PersistentCardType.WORN_WITH_SORROW))
    names0 = set(str(getattr(c, 'name', c))
                 for c in drv.es.fellowship.companions)
    if 'Boromir' not in names0:
        fails.append(('wwsat_control', 'NOT-MEASURED: no Boromir'))
        return

    def chooser(state, side, fp, sh, num_simulations=0,
                temperature=0.0, add_noise=False):
        with srv.rand.pin():
            dp = ai.describe_pending(state, side)
        acts = dp['actions'] if dp is not None else []
        a = next((x for x in acts
                  if x.action_type == 'hunt_sacrifice_random'), None)
        return (a or (acts[0] if acts else None)), None
    orig = ai.mcts_action_with_policy
    ai.mcts_action_with_policy = chooser
    srv.fp_net = srv.sh_net = 'SCRIPT'
    from wotr_bridge.live.driver import DriverHalt
    try:
        out1 = srv.handle({'lines': ()})
        if '!CALL random_companion' not in out1:
            fails.append(('wwsat_control', 'NOT-MEASURED: no companion '
                          'ask: %r' % (out1,)))
            return
        srv.rand.feed('companion', ['Boromir'])
        try:
            out2 = srv.handle({'lines': ()})
        except DriverHalt as e:
            fails.append(('wwsat_halt', str(e)[:160]))
            return
        except RuntimeError as e:
            fails.append(('wwsat_runtime_error',
                          'the card_pick starve escaped as %r' % (e,)))
            return
        p = srv.rand.pending()
        if getattr(p, 'kind', None) != 'card_pick':
            fails.append(('wwsat_control', 'NOT-MEASURED: pending %r '
                          'out=%r notes=%r'
                          % (p and (p.kind, p.kw), out2,
                             srv.notes[-3:])))
            return
        if '!CALL wwsat' not in out2:
            fails.append(('wwsat_directive', repr(out2)))
        # idle retry: no re-order
        out2b = srv.handle({'lines': ()})
        if '!CALL wwsat' in out2b:
            fails.append(('wwsat_reorder', repr(out2b)))
        # the client performs the discard; its line renders in the
        # feed, the sidecar names the AI's own card piece
        _append(feed, '$Wey randomly discards a Free Character Card.')
        with open(meta, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'seq': 1, 'gen': 1, 'raw_seq': 5,
                                'raw_new': True, 'nick': 'Wey',
                                'tags': ['Controls.452'],
                                'pids': ['150'], 'side': 'FP',
                                'ts': 0}) + '\n')
        try:
            out3 = srv.handle({'lines': ()})
        except DriverHalt as e:
            fails.append(('wwsat_halt_post', str(e)[:160]))
            return
        if cid in drv.es.hands[Side.FREE_PEOPLES].hand:
            fails.append(('wwsat_not_discarded',
                          'hand %r open=%r pending=%r notes=%r'
                          % (drv.es.hands[Side.FREE_PEOPLES].hand,
                             srv._open_step, srv.rand.pending(),
                             srv.notes[-3:])))
            return
        if srv._open_step is not None:
            fails.append(('wwsat_step_open', repr(srv._open_step[0])))
        if drv.alarms:
            fails.append(('wwsat_alarms', repr(drv.alarms)))
        if 'Boromir' in set(str(getattr(c, 'name', c))
                            for c in drv.es.fellowship.companions):
            fails.append(('wwsat_casualty', 'Boromir still in'))
    finally:
        ai.mcts_action_with_policy = orig


def test_directive_failure_holds_before_emitting(fails):
    """MEASURES (W1#18 ordering): when the directive for a starved AI
    step cannot be built, the step HOLDS LOUDLY (DriverHalt) BEFORE its
    pre-apply emission goes out - nothing emitted, _em_sent false, the
    emission's calls not booked as 'already ordered' - and the retry,
    once the builder works, emits and asks as a first attempt."""
    import ai
    import game_logic
    from game_state import ActionDieFace
    from wotr_bridge.live.driver import DriverHalt
    td = tempfile.mkdtemp()
    open(os.path.join(td, 'ai_feed.txt'), 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=23)
    drv = srv.driver
    game_logic.adopt_rolled_faces(
        drv.es, [(Side.SHADOW, 0, ActionDieFace.EVENT)])
    drv.phase = 'actions'
    drv.last_actor = Side.FREE_PEOPLES
    want = {'type': 'draw_card', 'calls': 0}
    orig = ai.mcts_action_with_policy
    ai.mcts_action_with_policy = _script_chooser(srv, want)
    srv.fp_net = srv.sh_net = 'SCRIPT'
    real_dir = srv.rand.directives_for

    def broken(call):
        raise RuntimeError('unknown NeedClientCall %r' % (call.kind,))
    srv.rand.directives_for = broken
    try:
        try:
            out1 = srv.handle({'lines': ()})
            fails.append(('dirfail_not_loud',
                          'no DriverHalt; out=%r' % (out1,)))
        except DriverHalt as e:
            if 'DIRECTIVE' not in str(e).upper():
                fails.append(('dirfail_message', str(e)[:120]))
        except RuntimeError as e:
            fails.append(('dirfail_runtime_error', repr(e)))
        if srv._em_sent or srv._step_calls:
            fails.append(('dirfail_emitted',
                          'em_sent=%r step_calls=%r'
                          % (srv._em_sent, srv._step_calls)))
        srv.rand.directives_for = real_dir
        out2 = srv.handle({'lines': ()})
        if not any(s.startswith('$') for s in out2) \
                or not any(s.startswith('!CALL draw_') for s in out2):
            fails.append(('dirfail_retry', repr(out2)))
    finally:
        ai.mcts_action_with_policy = orig
        srv.rand.directives_for = real_dir


def test_transport_error_cycle_replies(fails):
    """MEASURES (W1#21 transport / #23): a handler exception must still
    complete the client's handshake - an EMPTY reply file (a no-op to
    AiOpp.requestMove, which otherwise waits its full 30 s on every
    failed cycle) - with the error STICKY on the panel status
    (state='error', the message in waiting_on) until CLEAR ERROR; a
    request without a side header likewise gets the empty reply.  And
    a randomness LIVE HALT is sticky on the server: the next cycle
    refuses to play on (DriverHalt naming it) until the panel RESET
    clears it."""
    import json
    import tempfile as _tf
    from wotr_bridge.live.transport import FileBridge
    from wotr_bridge.live.driver import DriverHalt
    td = _tf.mkdtemp()
    resp = os.path.join(td, 'ai_bridge_move.txt')
    status = os.path.join(td, 'ai_bridge_status.json')

    def boom(request):
        raise RuntimeError('LIVE HALT: synthetic handler failure')
    bridge = FileBridge(td, owner_needle='test_live_server',
                        poll=0.01)
    with open(os.path.join(td, 'ai_bridge_request.log'), 'w',
              encoding='utf-8') as f:
        f.write('LIVE_SHADOW\n$Wey noise\n')
    try:
        bridge.serve(boom, max_cycles=2)
    finally:
        bridge.release_lock()
    if not os.path.exists(resp):
        fails.append(('err_no_reply', 'handler exception wrote no '
                      'reply file - the client waits 30 s per cycle'))
    else:
        if open(resp, encoding='utf-8').read().strip():
            fails.append(('err_reply_not_empty',
                          repr(open(resp).read()[:80])))
    try:
        st = json.load(open(status, encoding='utf-8'))
    except Exception as e:
        st = {'_err': repr(e)}
    if st.get('state') != 'error' \
            or 'synthetic handler failure' not in str(
                st.get('waiting_on', '')):
        fails.append(('err_status_not_sticky', repr(st)[:200]))
    # no side header -> empty reply, not a silent 30 s stall
    try:
        os.remove(resp)
    except OSError:
        pass
    with open(os.path.join(td, 'ai_bridge_request.log'), 'w',
              encoding='utf-8') as f:
        f.write('$Wey noise without a header\n')
    bridge2 = FileBridge(td, owner_needle='test_live_server',
                         poll=0.01)
    try:
        bridge2.serve(boom, max_cycles=2)
    finally:
        bridge2.release_lock()
    if not os.path.exists(resp):
        fails.append(('nohdr_no_reply', 'headerless request got no '
                      'reply file'))
    # the server-side sticky halt
    td2 = _tf.mkdtemp()
    open(os.path.join(td2, 'ai_feed.txt'), 'w').close()
    srv = LiveServer(td2, Side.SHADOW, load_nets=False, seed=5)
    srv.handle({'lines': ()})
    try:
        srv.rand._halt('synthetic desync')
    except RuntimeError:
        pass
    halted = False
    try:
        srv.handle({'lines': ()})
    except DriverHalt as e:
        halted = 'synthetic desync' in str(e)
    if not halted:
        fails.append(('sticky_halt_ignored',
                      'the cycle after a LIVE HALT played on'))
    srv._reset_requested = True
    try:
        srv.handle({'lines': ()})
    except DriverHalt as e:
        fails.append(('sticky_halt_reset', 'reset did not clear it: %s'
                      % str(e)[:100]))
    if srv.rand.sticky_halt:
        fails.append(('sticky_halt_flag', repr(srv.rand.sticky_halt)))


def test_mcts_miss_is_loud(fails):
    """MEASURES (W1#22): the chooser never takes a degraded decision
    silently - a search returning None HALTS the step (DriverHalt), a
    search pick absent from the engine menu is ALARMED (driver.alarms)
    before the menu order decides."""
    import ai
    import game_logic
    from game_state import ActionDieFace
    from wotr_bridge.live.driver import DriverHalt
    td = tempfile.mkdtemp()
    open(os.path.join(td, 'ai_feed.txt'), 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=23)
    drv = srv.driver
    game_logic.adopt_rolled_faces(
        drv.es, [(Side.SHADOW, 0, ActionDieFace.ARMY)])
    drv.phase = 'actions'
    drv.last_actor = Side.FREE_PEOPLES
    orig = ai.mcts_action_with_policy
    srv.fp_net = srv.sh_net = 'SCRIPT'
    try:
        ai.mcts_action_with_policy = (
            lambda *a, **k: (None, None))
        try:
            out = srv.handle({'lines': ()})
            fails.append(('mcts_none_silent', repr(out)))
        except DriverHalt as e:
            if 'MCTS' not in str(e):
                fails.append(('mcts_none_message', str(e)[:100]))
        srv._abandon_step('test')

        class _Foreign(object):
            action_type = 'army_move'
            die_idx = 0
            params = {'from_region': 'nowhere', 'to_region': 'nowhere'}
        ai.mcts_action_with_policy = (
            lambda *a, **k: (_Foreign(), None))
        n_al = len(drv.alarms)
        srv.handle({'lines': ()})
        if not any('MCTS PICK NOT IN MENU' in str(a)
                   for a in drv.alarms[n_al:]):
            fails.append(('mcts_miss_silent', repr(drv.alarms[n_al:])))
    finally:
        ai.mcts_action_with_policy = orig


def test_feed_reattach_halts(fails):
    """MEASURES (W1#19, sidecar present): a chat-document re-attach
    (sidecar gen 1 -> 2: the client began a NEW game and the agent
    re-emitted its document onto the same feed) must HALT the server
    loudly (DriverHalt naming the re-attach) instead of routing the
    new game's lines into the old game."""
    import json
    from wotr_bridge.live.driver import DriverHalt
    td = tempfile.mkdtemp()
    feed = os.path.join(td, 'ai_feed.txt')
    meta = feed + '.meta.jsonl'
    open(feed, 'w').close()
    open(meta, 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=3)
    _append(feed, '$Wey views the Shadow event hand.')
    with open(meta, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'seq': 1, 'gen': 1, 'replay': True,
                            'nick': 'Wey', 'tags': [], 'pids': [],
                            'ts': 1}) + '\n')
    srv.handle({'lines': ()})          # gen 1: the normal attach
    _append(feed, '$Wey views the Shadow event hand.')
    with open(meta, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'seq': 2, 'gen': 2, 'replay': True,
                            'nick': 'Wey', 'tags': [], 'pids': [],
                            'ts': 2}) + '\n')
    try:
        srv.handle({'lines': ()})
        fails.append(('reattach_silent', 'gen 2 lines routed as a '
                      'continuation'))
    except DriverHalt as e:
        if 'RE-ATTACH' not in str(e).upper():
            fails.append(('reattach_message', str(e)[:120]))


def test_ai_declare_never_arms_flag(fails):
    """MEASURES (K2, server half): the AI-FP declare step DECIDES only
    when the engine says 'declare' for the AI side; it never arms or
    clears the engine's declare flag itself (the driver owns WHEN via
    game_logic.open/close_declare_phase).  With the driver's arming
    disabled the step must decline by absence - flag untouched, no
    declaration applied, its 'fp ok' said."""
    td = tempfile.mkdtemp()
    feed = os.path.join(td, 'ai_feed.txt')
    open(feed, 'w').close()
    srv = LiveServer(td, Side.FREE_PEOPLES, load_nets=False, seed=17)
    drv = srv.driver
    if not hasattr(drv, '_ensure_declare_armed'):
        fails.append(('k2_control', 'NOT-MEASURED: driver has no '
                      '_ensure_declare_armed (K2 driver half absent)'))
        return
    drv._ensure_declare_armed = lambda: None      # the driver never arms
    _append(feed, '$Wey draws Shadow Character Card.',
            '$Wey draws Shadow Strategy Card.')
    out = srv.handle({'lines': ()})
    # the AI's own draws first (ordered, then fed), then the declare
    for _ in range(3):
        for deck, lst in (('character', drv.es.fp_char_deck),
                          ('strategy', drv.es.fp_strat_deck)):
            if any(('draw_fp_%s' % deck[:4]) in ln for ln in out):
                srv.rand.feed('card', ('FREE_PEOPLES', deck, lst[-1]))
        if getattr(drv.es, '_declare_phase', False):
            fails.append(('k2_server_armed_flag',
                          'es._declare_phase became True with the '
                          'driver arming disabled'))
            return
        out = srv.handle({'lines': ()})
        if any('fp ok' in ln for ln in out):
            break
    else:
        fails.append(('k2_control', 'NOT-MEASURED: no fp ok within 3 '
                      'cycles: out=%r notes=%r open=%r'
                      % (out, srv.notes[-4:], srv._open_step)))
        return
    if getattr(drv.es, '_declare_phase', False):
        fails.append(('k2_flag_left_armed', 'flag True after fp ok'))
    if any('fellowship_declare' in str(c) for c in drv.commits):
        fails.append(('k2_declared_without_menu', repr(drv.commits)))


def test_resync_is_reconcile_only(fails):
    """MEASURES (USER 2026-08-24 'resync should not automatically
    trigger the AI's turn'): a resync tick reconciles and NOTHING
    else - an open AI step does not run that tick (it runs on the
    NEXT tick), and the human is marked as owed the next action
    (last_actor = ai_side)."""
    td = tempfile.mkdtemp()
    feed = os.path.join(td, 'ai_feed.txt')
    open(feed, 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=42)
    drv = srv.driver
    drv.last_actor = Side.SHADOW          # AI just acted (pre-state)
    srv._open_step = ('ai_draws', None)   # an AI step is open
    srv._resync_requested = True
    out1 = srv.handle({'lines': ()})
    if any('draw_sa' in str(s) for s in out1):
        fails.append(('resync_ran_ai_step', out1))
    if drv.last_actor is not Side.SHADOW:
        fails.append(('resync_last_actor', drv.last_actor))
    if not any('reconcile only' in n for n in srv.notes):
        fails.append(('resync_note', srv.notes[-3:]))
    # the NEXT tick resumes the open step normally
    out2 = srv.handle({'lines': ()})
    if not any('draw_sa' in str(s) for s in out2):
        fails.append(('resync_next_tick_resumes', out2))


def test_table_seat_emits_under_open_pending(fails):
    """MEASURES (tester 2026-08-24 'A Power too Great never reached
    the board'): the Board-slot SEAT of a just-played persistent card
    emits at COMMIT even while an unrelated engine pending is open -
    only the DIE KILL waits for chain closure.  Before the fix the
    whole flush was gated on describe_pending()=None, so a long-lived
    pending (the discard-at-6 phantom) stranded the card in the
    Current-Card slot silently.  SENSITIVITY: the die kill really is
    still banked while the pending is open, and flushes after."""
    import ai
    import test_live_emission as TE
    cid = 'FREE_PEOPLES_Strategy_A_Power_too_Great'
    st = TE.fresh_state()
    TE.adopt(st)
    if cid in st.fp_strat_deck:
        st.fp_strat_deck.remove(cid)
    st.hands[Side.FREE_PEOPLES].hand.append(cid)
    act = next((a for a in ai.get_legal_actions(st, Side.FREE_PEOPLES)
                if a.action_type == 'play_event'
                and a.params.get('card_id') == cid), None)
    if act is None:
        fails.append(('seat_precondition', 'play_event not offered'))
        return
    post = ai.apply_action(st, act)
    td = tempfile.mkdtemp()
    open(os.path.join(td, 'ai_feed.txt'), 'w').close()
    srv = LiveServer(td, Side.FREE_PEOPLES, load_nets=False, seed=3)
    drv = srv.driver
    drv.es = post
    drv.view.card_pid_map[cid] = '106'
    srv._note_owed_die_kill({'side': Side.FREE_PEOPLES}, act)
    # an unrelated open pending (the tester's phantom discard shape)
    sh = post.hands[Side.SHADOW]
    sh.hand = list(sh.hand) + [
        'HU_SH_CHAR_%d' % i for i in range(max(0, 7 - len(sh.hand)))]
    post.pending_discard = {'side': Side.SHADOW}
    flush = srv._flush_owed_die_kills()
    if not any('Game.2029' in ln and '106' in ln for ln in flush):
        fails.append(('seat_emitted_under_pending', flush))
    if not drv.owed_die_kills \
            or any(len(e) > 2 and e[2] for e in drv.owed_die_kills):
        fails.append(('die_kill_still_banked_pid_stripped',
                      drv.owed_die_kills))
    if any('Controls.644' in ln for ln in flush):
        fails.append(('die_kill_not_deferred', flush))
    # The unrelated discard closes, but Gandalf's optional guide draw
    # still owns the played die. The AI's fixed policy takes it before cleanup.
    sh.hand = sh.hand[:6]
    post.pending_discard = None
    pending = ai.describe_pending(drv.es)
    if (getattr(drv.es, 'pending_card_option', None) or {}).get('card_id') \
            != '__gandalf_draw__' or pending is None:
        fails.append(('seat_gandalf_choice_missing', pending))
        return
    if srv._flush_owed_die_kills():
        fails.append(('seat_die_flushed_before_gandalf_choice', True))
    hand_before = list(drv.es.hands[Side.FREE_PEOPLES].hand)
    deck_before = list(drv.es.fp_strat_deck)
    out = srv._ai_decide(pending)
    if [line for line in out if line.startswith('!CALL draw_')] != ['!CALL draw_fp_strat']:
        fails.append(('seat_bonus_draw_call', out))
    if srv._flush_owed_die_kills():
        fails.append(('seat_die_flushed_before_bonus_result', True))
    drawn = 'FREE_PEOPLES_Strategy_The_Red_Arrow'
    if drawn not in drv.es.fp_strat_deck:
        fails.append(('seat_bonus_draw_fixture_missing', drawn))
        return
    drv.view.card_pid_map[drawn] = '145'
    srv.rand.feed('card', ('FREE_PEOPLES', 'strategy', drawn))
    out = srv._ai_decide(pending)
    if any(line.startswith('!CALL draw_') for line in out) \
            or drv.es.hands[Side.FREE_PEOPLES].hand != hand_before + [drawn] \
            or len(drv.es.fp_strat_deck) != len(deck_before) - 1:
        fails.append(('seat_bonus_not_exactly_once', out))
    flush2 = srv._flush_owed_die_kills()
    if not any('Controls.644' in ln for ln in flush2):
        fails.append(('die_kill_flushes_after', flush2))
    if any('Game.2029' in ln for ln in flush2):
        fails.append(('seat_not_repeated', flush2))
    if drv.owed_die_kills:
        fails.append(('owed_cleared', drv.owed_die_kills))


def test_turn_over_protocol(fails):
    """MEASURES (BL-3, 2026-08-25 - fpgame1 sat wedged in turn-1
    actions with every die spent): with both pools exhausted and
    nothing pending, (a) waiting_on names the RECOVER press, never
    '<human> to act'; (b) the tick announces 'turn over' exactly ONCE
    per turn; (c) the pass refusal explains the recover instead of
    'not your action turn'."""
    td = tempfile.mkdtemp()
    open(os.path.join(td, 'ai_feed.txt'), 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=7)
    drv = srv.driver
    drv.phase = 'actions'
    drv.last_actor = Side.SHADOW
    for s in (Side.FREE_PEOPLES, Side.SHADOW):
        for d in drv.es.dice.get(s, []):
            d.used = True
    if not drv._turn_over_pending():
        fails.append(('turn_over_pending', 'False with pools spent'))
        return
    w = srv._waiting_on()
    if 'Recover Action Dice' not in str(w):
        fails.append(('turn_over_waiting_on', w))
    if 'to act' in str(w):
        fails.append(('turn_over_waiting_on_lie', w))
    out1 = srv.handle({'lines': ()})
    if not any('turn over' in str(s) for s in out1):
        fails.append(('turn_over_announced', out1))
    out2 = srv.handle({'lines': ()})
    if any('turn over' in str(s) for s in out2):
        fails.append(('turn_over_announced_twice', out2))
    ref = drv._human_plain_pass()
    txt = ' '.join(str(i) for i in ref)
    if 'Recover Action Dice' not in txt:
        fails.append(('turn_over_pass_refusal', ref))
    # SENSITIVITY: with a ROLLED die still free the protocol stays
    # silent (an unrolled pool is legitimately actionless - phase
    # 'actions' only opens after both pools rolled)
    from game_state import ActionDieFace
    td2 = tempfile.mkdtemp()
    open(os.path.join(td2, 'ai_feed.txt'), 'w').close()
    srv2 = LiveServer(td2, Side.SHADOW, load_nets=False, seed=8)
    d2 = srv2.driver
    d2.phase = 'actions'
    d2.last_actor = Side.SHADOW
    for s in (Side.FREE_PEOPLES, Side.SHADOW):
        for d in d2.es.dice.get(s, []):
            d.current = ActionDieFace.CHARACTER
            d.used = True
    free = d2.es.dice[Side.FREE_PEOPLES][0]
    free.used = False
    if d2._turn_over_pending():
        fails.append(('turn_over_false_positive', 'die still free'))


def test_corruption_sweep_and_owed(fails):
    """MEASURES (BL-2, 2026-08-25): a commit that changed corruption
    with NO gesture and NO expressed op emits the marker-252 drag as
    the AI's own line and arms the owed-track hold; a commit whose
    expressed ops already move 252 does NOT double-walk."""
    import ai
    td = tempfile.mkdtemp()
    open(os.path.join(td, 'ai_feed.txt'), 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=11)
    drv = srv.driver
    pre = drv.es
    post = ai.clone_state(pre)
    post.fellowship.corruption = int(
        getattr(pre.fellowship, 'corruption', 0) or 0) + 1
    items = drv.post_commit_sweep(pre, post, label='pin')
    c0 = int(getattr(pre.fellowship, 'corruption', 0) or 0)
    want = 'Game.%d' % (2014 + c0 + 1)
    if not any('252' in str(i[1]) and want in str(i[1])
               for i in items if i and i[0] == 'line'):
        fails.append(('corruption_sweep_emits', items))
    owed = getattr(drv, '_track_owed', {})
    if owed.get('252', (None,))[0] != c0 + 1:
        fails.append(('corruption_owed_armed', owed))
    # voiced by expressed ops: no second walk
    td2 = tempfile.mkdtemp()
    open(os.path.join(td2, 'ai_feed.txt'), 'w').close()
    drv2 = LiveServer(td2, Side.SHADOW,
                      load_nets=False, seed=12).driver
    items2 = drv2.post_commit_sweep(
        pre, post, expressed=[('move', {'nick': 'x',
                                        'pieces': ['252'],
                                        'frm': 'Game.2014',
                                        'to': 'Game.2015'})],
        label='pin2')
    if any('252' in str(i[1]) for i in items2
           if i and i[0] == 'line'):
        fails.append(('corruption_sweep_double_walk', items2))


def test_political_emission_registers_inflight(fails):
    """MEASURES (BL-6, 2026-08-25 - the Isengard advance rendered
    twice): EVERY outgoing line that walks a political marker
    registers on the recon in-flight ledger via _dispatch, and
    _political_recon holds while the registered target matches the
    engine box - the dump alone (stale) no longer triggers a
    double-walk."""
    from log_replay import POLITICAL_MARKERS
    td = tempfile.mkdtemp()
    open(os.path.join(td, 'ai_feed.txt'), 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=13)
    drv = srv.driver
    pid, nation = next(iter(POLITICAL_MARKERS.items()))
    eng_box = drv.es.political.boxes.get(nation)
    if eng_box is None:
        fails.append(('political_pin_precondition', nation))
        return
    tgt = 'Game.%d' % (2000 + int(eng_box))
    from emitter import render
    ln = render('move', {'nick': drv.nick, 'pieces': [str(pid)],
                         'frm': 'Game.%d' % (2000 + int(eng_box) + 1),
                         'to': tgt}, local=False)
    srv._dispatch(('line', ln))
    infl = getattr(drv, '_pol_recon_inflight', {})
    if infl.get(str(pid)) != tgt:
        fails.append(('political_inflight_registered', infl))
    # a stale dump still shows the OLD box: the recon must hold
    stale_client_box = int(eng_box) + 1 if int(eng_box) < 3 \
        else int(eng_box) - 1
    drv.view.piece_region = {
        str(pid): 'Political %d' % (3 - stale_client_box)}
    items = drv._political_recon()
    if any(str(pid) in str(i[1]) for i in items
           if i and i[0] == 'line'):
        fails.append(('political_recon_double_walk', items))


def test_human_hand_visible_in_menu(fails):
    """MEASURES (USER 2026-08-25 'card plays should be in the menu,
    the engine should see the human hand'): (a) a decoded human draw
    swaps its placeholder for the real cid via the banked-ids sweep;
    (b) a real cid in the human hand publishes a play_event row with
    card_id readable and the client pid riding params."""
    from game_state import ActionDieFace
    td = tempfile.mkdtemp()
    open(os.path.join(td, 'ai_feed.txt'), 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=21)
    drv = srv.driver
    hm = drv.hands
    cid = 'FREE_PEOPLES_Strategy_A_Power_too_Great'
    ph = hm.human_draw('strategy')     # the rendered-draw placeholder
    srv._human_draw_ids = [{'side': 'FREE_PEOPLES',
                            'deck': 'strategy', 'cid': cid,
                            'pid': '117', 'name': 'x'}]
    drv.view.card_pid_map[cid] = '117'
    srv.handle({'lines': ()})
    hand = drv.es.hands[Side.FREE_PEOPLES].hand
    if cid not in hand or ph in hand:
        fails.append(('hand_swap', (cid in hand, ph in hand)))
    if getattr(srv, '_human_draw_ids', None):
        fails.append(('hand_swap_unbanked', srv._human_draw_ids))
    # (b) the publication: action phase, an Event die free
    drv.phase = 'actions'
    drv.last_actor = Side.SHADOW
    for d in drv.es.dice[Side.FREE_PEOPLES]:
        d.current = ActionDieFace.EVENT
        d.used = False
    s1 = srv.status_menu()
    hmenu = s1.get('human_menu', {})
    pe = [a for a in hmenu.get('actions', ())
          if a['action_type'] == 'play_event'
          and a['params'].get('card_id') == cid]
    if not pe:
        fails.append(('play_event_published',
                      hmenu.get('by_type')))
    elif str(pe[0]['params'].get('pid')) != '117':
        fails.append(('play_event_pid', pe[0]['params']))


def test_early_recover_refused(fails):
    """MEASURES (BL-13, shgame2: recover pressed with 4 live dice
    silently ended turn 1): _on_recover with unspent dice on either
    side REFUSES (chat + corrective undos) and the turn does NOT
    roll; with both pools spent the same press still rolls over."""
    td = tempfile.mkdtemp()
    open(os.path.join(td, 'ai_feed.txt'), 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=31)
    drv = srv.driver
    drv.phase = 'actions'
    drv.last_actor = Side.SHADOW
    t0 = int(drv.es.turn)
    rec = {'op': 'move', 'norm': 'moves group (dice) from Current '
                                 'Action to FP Dice', 'raw': 'x'}
    out = drv._on_recover(rec)
    txt = ' '.join(str(i) for i in out)
    if 'cannot recover' not in txt:
        fails.append(('early_recover_refusal', out))
    if int(drv.es.turn) != t0 or drv.phase != 'actions':
        fails.append(('early_recover_rolled',
                      (drv.es.turn, drv.phase)))
    # both pools spent: the same press legitimately rolls the turn
    from game_state import ActionDieFace
    for s in (Side.FREE_PEOPLES, Side.SHADOW):
        for d in drv.es.dice.get(s, []):
            d.current = ActionDieFace.CHARACTER
            d.used = True
    out2 = drv._on_recover(rec)
    if int(drv.es.turn) != t0 + 1:
        fails.append(('legit_recover_blocked', (drv.es.turn, out2)))


def test_nc_held_until_human_commit(fails):
    """MEASURES (BL-22, fpgame2 F15 - K1 blinding): the AI's 'nc'
    is HELD while the human's combat-card decision is open, and
    released once it closes."""
    td = tempfile.mkdtemp()
    open(os.path.join(td, 'ai_feed.txt'), 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=32)
    drv = srv.driver
    # a live round with the HUMAN (FP defender) commitment still
    # outstanding - the hold must apply even BEFORE their decision
    # window opens (fpgame5 F9: nc printed pre-window)
    drv.es.pending_combat = {'attacker_side': Side.SHADOW,
                             'defender_side': Side.FREE_PEOPLES,
                             'attacker_region': 'x',
                             'defender_region': 'y',
                             'phase': 'combat_cards',
                             'rounds_fought': 0}
    out = srv._dispatch(('chat', 'nc', 'nc'))
    if out or getattr(srv, '_held_nc', None) != 'nc':
        fails.append(('nc_not_held', (out, getattr(srv, '_held_nc',
                                                   None))))
    # the human commits (engine carries their card) -> released
    drv.es.pending_combat['def_combat_card'] = None
    out2 = srv.handle({'lines': ()})
    if not any('nc' in str(s) for s in out2):
        fails.append(('nc_not_released', out2))
    if getattr(srv, '_held_nc', None) is not None:
        fails.append(('nc_still_held', srv._held_nc))


def test_menu_eye_binds_pid(fails):
    """MEASURES (BL-25, shgame3 turn-1 paralysis): the allocated eye
    die's pid rides the SIDECAR segs[1] channel; the bind is
    slot-unique WITH SWAP (the client picked pid 105 while the static
    seed held it at reg6 - without eviction+swap, two pids shared a
    reg, die_piece_for's last-wins inversion shifted every published
    index and idx6 fell off the end)."""
    td = tempfile.mkdtemp()
    open(os.path.join(td, 'ai_feed.txt'), 'w').close()
    srv = LiveServer(td, Side.FREE_PEOPLES, load_nets=False, seed=33)
    drv = srv.driver
    drv.phase = 'eyes'
    ok = drv._allocate_one_eye(
        rec={'op': 'alloc_eye', 'raw': 'x',
             'meta': {'segs': [[], ['105']]},
             'norm': 'allocates an eye.'})
    if not ok:
        fails.append(('eye_alloc_refused_in_pin', 'see alarms'))
        return
    b = drv.view.die_id_map.get('105')
    if b is None or b[0] is not Side.SHADOW or b[1] != 0:
        fails.append(('eye_pid_bound', b))
    # coverage invariant: 7 SA regs, one pid each, no shared slots
    regs = sorted(i for p, (s, i) in drv.view.die_id_map.items()
                  if s is Side.SHADOW)
    if regs != list(range(7)):
        fails.append(('eye_bind_coverage', regs))
    # the displaced seed pid took the binder's old slot (SWAP)
    if drv.view.die_id_map.get('099') != (Side.SHADOW, 6):
        fails.append(('eye_bind_swap',
                      drv.view.die_id_map.get('099')))


def test_die_filter_will_wildcard(fails):
    """MEASURES (BL-32, fpgame3): WILL is the wildcard face on BOTH
    sides of the one-answerer seam - a staged W matches every
    candidate, a W-funded candidate matches any staged face; a plain
    mismatch still refuses."""
    from wotr_bridge.live import gesture_match as gm
    from game_state import ActionDieFace

    class _A(object):
        def __init__(self, di):
            self.die_idx = di
    td = tempfile.mkdtemp()
    open(os.path.join(td, 'ai_feed.txt'), 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=34)
    st = srv.driver.es
    dice = st.dice[Side.FREE_PEOPLES]
    dice[0].current = ActionDieFace.CHARACTER
    dice[1].current = ActionDieFace.WILL_OF_WEST
    ok_w = gm._die_filter(st, Side.FREE_PEOPLES, 'W')
    if not ok_w(_A(0)):
        fails.append(('staged_w_matches_any', 'refused C candidate'))
    ok_c = gm._die_filter(st, Side.FREE_PEOPLES, 'C')
    if not ok_c(_A(1)):
        fails.append(('w_candidate_matches_any', 'refused W row'))
    ok_a = gm._die_filter(st, Side.FREE_PEOPLES, 'A')
    if ok_a(_A(0)):
        fails.append(('plain_mismatch_still_refuses', 'A vs C passed'))


def main():
    tests = [test_turnstart_smoke, test_transport_seam,
             test_ai_declare_never_arms_flag,
             test_feed_reattach_halts,
             test_transport_error_cycle_replies, test_mcts_miss_is_loud,
             test_wwsat_card_pick_step,
             test_directive_failure_holds_before_emitting,
             test_abandoned_ai_step, test_adopt_refuses_partial_dump,
             test_ai_draw_decode, test_die_piece_binding,
             test_ai_draw_decode_live_variants,
             test_step_queue_no_clobber, test_status_menu_publication,
             test_casualty_rows_publication,
             test_status_wave_j_blocks, test_slot_cleanup,
             test_game_end, test_j1_bonus_dice,
             test_mordor_entry_and_recovery, test_deferral_watchdog,
             test_roll_ask_nonce, test_gesture_reply_lane,
             test_randomcomp_ask, test_own_emission_tracking,
             test_waiting_on,
             test_waiting_on_is_explicit_when_buffered,
             test_bonus_die_prompt_reads_the_board,
             test_window_rng_deferral_ask,
             test_draw_action_single_call,
             test_draw_action_extra_draw,
             test_combat_card_kill,
             test_combat_d6_harvest_after_prefix,
             test_human_casualty_after_ai_wipe_matches,
             test_human_casualty_before_forced_ai_downgrade,
             test_forced_casualty_delta_is_exact_and_opponent_only,
             test_ai_fp_turn_start,
             test_ai_fp_random_casualty,
             test_side_header_guard,
             test_status_identity_first,
             test_action_phase_waits_for_both_rolls,
             test_mirror_snapshot_written,
             test_human_table_card_killed_in_its_slot,
             test_panel_turn_override_and_clear_error,
             test_panel_skip_decision,
             test_panel_reset_and_resync,
             test_table_kill_owes_die_and_uses_watched_pid,
             test_table_seat_emits_under_open_pending,
             test_resync_is_reconcile_only,
             test_turn_over_protocol,
             test_corruption_sweep_and_owed,
             test_political_emission_registers_inflight,
             test_human_hand_visible_in_menu,
             test_early_recover_refused,
             test_nc_held_until_human_commit,
             test_cc_pick_commit_reveal_prompt,
             test_ai_defender_casualty_absorption_exact,
             test_menu_eye_binds_pid,
             test_die_filter_will_wildcard]
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
