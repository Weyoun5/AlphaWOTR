"""wotr_bridge/live/transport.py + chat.py - the file-bridge serve loop
and the sided chat vocabulary (REBUILD_PLAN Phase 2 lift-list).

Each test states its MEASURED QUANTITY first (instrument protocol,
design rule 11).  Every gate has a sensitivity control (the defect
class it must catch goes RED) and a specificity control (the
legitimate twin stays GREEN); every zero ships with a proven non-zero.

House style: fails-list runner, exit 1 on any failure, ASCII only.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

_SRC = os.path.dirname(os.path.abspath(__file__))
_BRIDGE = os.path.abspath(os.path.join(_SRC, '..', 'wotr_bridge'))
for _p in (_SRC, _BRIDGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from live import chat
from live import transport
from live.chat import ChatOut, ChatPolicyError
from live.transport import FileBridge, BridgeLockError, parse_request


def _tmpdir():
    return tempfile.mkdtemp(prefix='wotr_lift_c_')


class _A(object):
    """Minimal engine-Action shape (action_type, params, die_idx)."""

    def __init__(self, action_type, params=None, die_idx=0):
        self.action_type = action_type
        self.params = params or {}
        self.die_idx = die_idx

    def __repr__(self):
        return '<A %s %r>' % (self.action_type, self.params)


# ================================================================ (a)
def test_parse_request_grammar(fails):
    """MEASURES: the request-file grammar - all 5 side headers
    accepted with the exact log tail; a foreign header and an empty
    body yield (None, ...) (specificity: the loop must ignore them,
    never guess a side)."""
    body = ['$Wey moves Free Action Die (P) from FP Dice to Current '
            'Action.', '<Wey> fp ok']
    n_ok = 0
    for side in transport.SIDE_HEADERS:
        s, lines = parse_request('\n'.join([side] + body))
        if s == side and lines == body:
            n_ok += 1
        else:
            fails.append(('parse_header_%s' % side,
                          'got (%r, %d lines)' % (s, len(lines))))
    if n_ok != 5:
        fails.append(('parse_header_count', '%d of 5 accepted' % n_ok))
    s, lines = parse_request('\n'.join(['GONDOR'] + body))
    if s is not None or len(lines) != 3:
        fails.append(('parse_foreign_header',
                      'foreign header not refused: %r' % (s,)))
    if parse_request('') != (None, []):
        fails.append(('parse_empty', 'empty text mis-parsed'))


def test_bridge_roundtrip_and_dedup(fails):
    """MEASURES: (1) handler sees EXACTLY the request payload;
    (2) the response file carries exactly the handler's lines;
    (3) DEDUP: one written request file = exactly ONE handler call
    across many poll cycles (the rename-first consume), while a
    REWRITTEN request (a fresh client click, identical bytes) is a NEW
    request and is handled again - the old loop's measured semantics
    (deletion is the handshake; dedup is per consumed FILE, never per
    content); (4) a request without a side header is ignored (handler
    not called, no response)."""
    d = _tmpdir()
    try:
        calls = []

        def handler(req):
            calls.append((req.side, tuple(req.log_lines)))
            return (['$WOTR_AI_Shadow moves Shadow Action Die (E) '
                     'from SA Dice to The Hunt For The Ring. null',
                     '!CALL roll_action_dice 2043'],
                    'test-how')
        fb = FileBridge(d, owner_needle='test_live_transport_chat',
                        poll=0.005)
        req_text = ('LIVE_FREE_PEOPLES\n'
                    '$Wey moves Free Action Die (P) from FP Dice to '
                    'Current Action.\n<Wey> fp ok\n')
        transport.atomic_write(fb.req, req_text)
        fb.serve(handler, max_cycles=8)
        if len(calls) != 1:
            fails.append(('roundtrip_one_call',
                          'handler called %d times for one request '
                          'file' % len(calls)))
            return
        side, lines = calls[0]
        if side != 'LIVE_FREE_PEOPLES':
            fails.append(('roundtrip_side', repr(side)))
        want = ('$Wey moves Free Action Die (P) from FP Dice to '
                'Current Action.', '<Wey> fp ok')
        if lines != want:
            fails.append(('roundtrip_payload', repr(lines)))
        with open(fb.resp, encoding='utf-8') as f:
            got = f.read()
        want_resp = ('$WOTR_AI_Shadow moves Shadow Action Die (E) '
                     'from SA Dice to The Hunt For The Ring. null\n'
                     '!CALL roll_action_dice 2043\n')
        if got != want_resp:
            fails.append(('roundtrip_resp', repr(got[:120])))
        if os.path.exists(fb.req + '.processing'):
            fails.append(('roundtrip_processing_leak',
                          '.processing file left behind'))
        # dedup specificity: a REWRITTEN identical request is a fresh
        # click and IS handled again (measured old-loop semantics)
        transport.atomic_write(fb.req, req_text)
        fb.serve(handler, max_cycles=4)
        if len(calls) != 2:
            fails.append(('reclick_rehandled',
                          '%d calls after an identical rewrite '
                          '(want 2)' % len(calls)))
        # headerless request: ignored, no new handling, no fresh resp
        os.remove(fb.resp)
        transport.atomic_write(fb.req, 'no side header here\njunk\n')
        fb.serve(handler, max_cycles=4)
        if len(calls) != 2:
            fails.append(('headerless_ignored',
                          'handler ran on a headerless request'))
        # ...but the handshake completes with an EMPTY reply (a no-op
        # to AiOpp; a missing reply makes requestMove wait its full
        # 30 s before re-requesting - W1#21)
        if not os.path.exists(fb.resp):
            fails.append(('headerless_no_resp',
                          'no (empty) reply for a headerless request - '
                          'the client waits 30 s'))
        elif open(fb.resp, encoding='utf-8').read().strip():
            fails.append(('headerless_resp_content',
                          repr(open(fb.resp).read()[:80])))
        fb.release_lock()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_response_atomicity(fails):
    """MEASURES: the number of PARTIAL observations of a file written
    via transport.atomic_write under a tight concurrent reader, over
    12 writes of a ~2.4MB sentinel-terminated payload.  Must be ZERO.
    Non-zero control: >= 3 COMPLETE observations (else the zero is
    unreadable and this reports NOT-MEASURED).  A partial = the file
    exists with content not ending in the sentinel (a truncate-then-
    write emitter is exactly what this catches)."""
    d = _tmpdir()
    path = os.path.join(d, 'resp.txt')
    payload = ('x' * 118 + '\n') * 20000 + 'END-SENTINEL'
    obs = {'complete': 0, 'partial': 0, 'absent': 0, 'unreadable': 0}
    stop = []

    def watch():
        while not stop:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    t = f.read()
            except FileNotFoundError:
                obs['absent'] += 1
                continue
            except OSError:
                obs['unreadable'] += 1
                continue
            if t.endswith('END-SENTINEL\n'):
                obs['complete'] += 1
            else:
                obs['partial'] += 1
            time.sleep(0.0005)
    th = threading.Thread(target=watch, daemon=True)
    th.start()
    try:
        for _ in range(12):
            for attempt in range(200):
                try:
                    transport.atomic_write(path, payload)
                    break
                except PermissionError:
                    # the reader momentarily held the target during
                    # os.replace - retry; the product stays atomic
                    time.sleep(0.001)
            else:
                fails.append(('atomic_write_starved',
                              'os.replace never won against the '
                              'reader'))
                return
            time.sleep(0.002)
    finally:
        stop.append(1)
        th.join(timeout=5)
        shutil.rmtree(d, ignore_errors=True)
    # a measured partial is a failure regardless of the control - the
    # control guards only the ZERO case (a zero without a proven
    # non-zero is unreadable)
    if obs['partial']:
        fails.append(('atomicity_partials',
                      '%d partial reads observed (%r) - response '
                      'writes are not atomic' % (obs['partial'], obs)))
        return
    if obs['complete'] < 3:
        fails.append(('atomicity_control',
                      'NOT-MEASURED: only %d complete observations '
                      '(%r)' % (obs['complete'], obs)))
    print('  [atomicity] obs=%r' % (obs,))


# ================================================================ (b)
def test_single_writer_lock(fails):
    """MEASURES: the single-writer lock's four verdicts against REAL
    processes (client fact 18 - liveness by COMMAND LINE, never bare
    pid-exists):
      refuse   - lock names a live pid whose command line CONTAINS the
                 needle (sensitivity: the guard exists);
      takeover - lock names a live pid WITHOUT the needle (a reused
                 pid; the stale lock must be breakable);
      own      - lock names our own pid;
      takeover - lock names a dead pid."""
    marker = 'WOTR_LOCKMARKER_73X'
    child = subprocess.Popen(
        [sys.executable, '-c', 'import time; time.sleep(45)', marker],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    d = _tmpdir()
    lockp = os.path.join(d, transport.LOCK_NAME)
    try:
        # sanity: the probe can actually see the child's command line
        # (otherwise the refusal below would only prove the
        # conservative fallback, a different mechanism)
        from pid_owner import pid_command_line
        cl = pid_command_line(child.pid)
        if not cl or marker not in cl:
            fails.append(('lock_probe_control',
                          'NOT-MEASURED: cannot read the child '
                          'command line (%r)' % (cl and cl[:60],)))
            return
        # 1. refusal: live pid + matching command line
        with open(lockp, 'w') as f:
            f.write(str(child.pid))
        fb = FileBridge(d, owner_needle=marker)
        raised = False
        try:
            fb.acquire_lock()
        except BridgeLockError:
            raised = True
        if not raised:
            fails.append(('lock_refuses_live_owner',
                          'second writer NOT refused while a proven '
                          'live owner holds the lock'))
        # 2. stale sensitivity: SAME live pid, needle not in its
        # command line -> the lock is stale (reused pid) and breakable
        with open(lockp, 'w') as f:
            f.write(str(child.pid))
        fb2 = FileBridge(d, owner_needle='NEEDLE_NOT_PRESENT_88Q')
        how = fb2.acquire_lock()
        if how != 'takeover':
            fails.append(('lock_breaks_reused_pid',
                          'live-but-different-commandline pid was '
                          'not broken (got %r)' % (how,)))
        if open(lockp).read().strip() != str(os.getpid()):
            fails.append(('lock_takeover_writes_us',
                          'lock not rewritten to our pid'))
        # 3. our own pid -> proceed
        how = fb2.acquire_lock()
        if how != 'own':
            fails.append(('lock_own_pid', 'got %r' % (how,)))
        # 4. dead pid -> takeover
        child.terminate()
        child.wait(timeout=10)
        time.sleep(0.2)
        # End this server's ownership before simulating a different server.
        # A second instance in the same process is still a second writer.
        fb2.release_lock()
        with open(lockp, 'w') as f:
            f.write(str(child.pid))
        fb3 = FileBridge(d, owner_needle=marker)
        how = fb3.acquire_lock()
        if how != 'takeover':
            fails.append(('lock_breaks_dead_pid', 'got %r' % (how,)))
        fb3.release_lock()
    finally:
        if child.poll() is None:
            child.kill()
        shutil.rmtree(d, ignore_errors=True)


def test_panel_status_and_pause(fails):
    """MEASURES: (1) the status file's BOOT-1 ownership discipline - a
    foreign-pid file's SESSION keys are popped, identity keys stamped
    unconditionally (sensitivity), while a same-pid file keeps them
    (specificity); (2) PAUSED control: the handler is NOT called, an
    empty response keeps the handshake alive; un-pausing hands the
    request to the handler again."""
    import json
    d = _tmpdir()
    try:
        fb = FileBridge(d, owner_needle='test_live_transport_chat',
                        poll=0.005)
        # foreign-pid session keys popped
        transport.atomic_write(fb.status_path, json.dumps(
            {'pid': 999999999, 'turn': 7, 'note': 'stale'}))
        fb.write_status(state='idle')
        cur = json.load(open(fb.status_path, encoding='utf-8'))
        if 'turn' in cur or cur.get('note'):
            fails.append(('status_foreign_popped',
                          'dead session keys survived: %r' % (cur,)))
        if cur.get('pid') != os.getpid() or 'client_dir' not in cur \
                or 'ts' not in cur:
            fails.append(('status_identity_stamped', repr(cur)))
        # same-pid session keys KEPT (specificity)
        cur['turn'] = 9
        transport.atomic_write(fb.status_path, json.dumps(cur))
        fb.write_status(state='idle')
        cur2 = json.load(open(fb.status_path, encoding='utf-8'))
        if cur2.get('turn') != 9:
            fails.append(('status_own_kept',
                          'own-session turn popped: %r' % (cur2,)))
        # pause gating
        calls = []

        def handler(req):
            calls.append(req.side)
            return ['ok-line']
        transport.atomic_write(fb.control_path,
                               json.dumps({'paused': True}))
        transport.atomic_write(fb.req, 'LIVE_SHADOW\nline1\n')
        # max_cycles=1: the request cycle only, so the paused status
        # is the LAST write (idle heartbeats legitimately follow it in
        # a longer run - old-loop behavior kept)
        fb.serve(handler, max_cycles=1)
        if calls:
            fails.append(('pause_blocks_handler',
                          'handler ran while paused'))
        if not os.path.exists(fb.resp):
            fails.append(('pause_empty_reply',
                          'no keepalive response while paused'))
        elif open(fb.resp, encoding='utf-8').read().strip():
            fails.append(('pause_reply_nonempty',
                          'paused reply carried content'))
        st = json.load(open(fb.status_path, encoding='utf-8'))
        if st.get('state') != 'paused':
            fails.append(('pause_status', repr(st.get('state'))))
        # un-pause: same request rewritten is handled (control)
        transport.atomic_write(fb.control_path,
                               json.dumps({'paused': False}))
        transport.atomic_write(fb.req, 'LIVE_SHADOW\nline1\n')
        fb.serve(handler, max_cycles=4)
        if len(calls) != 1:
            fails.append(('unpause_handles',
                          '%d handler calls after unpause (want 1)'
                          % len(calls)))
        fb.release_lock()
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ================================================================ (c)
def test_chat_whitelist(fails):
    """MEASURES: the sided chat vocabulary at the ONE exit -
    SPECIFICITY: every allowed kind passes with a canonical specimen
    (count stated); SENSITIVITY: every named forbidden interface
    raises, an unknown kind raises, a menu riding an allowed kind
    raises (deny beats keep), an unregistered free-prose 'prompt'
    raises, and a two-card enumeration raises while the sanctioned
    'playable:' twin passes."""
    out = ChatOut()
    allowed = [
        ('field_battle', 'field battle'),
        ('nc', 'nc'),
        ('zero_eyes', '0e'),
        ('attack_declare', 'fords of isen from orthanc'),
        ('attack_declare', 'helms deep'),
        ('playable', 'playable: book of mazarbul, help unlooked for'),
        ('hit_count', '1 hit for you, 0 hits for me'),
        ('magnitude', 'relentless assault: i choose 0'),
        ('excess_dice', 'using the first 3: 3 1 4 (ignoring 5)'),
        ('roll_request', chat.roll_request(3)),
        ('roll_request', chat.roll_request(1)),
        ('leg_decline',
         chat.declined_leg_phrase('The Shadow Lengthens')),
        ('card_announce', 'plays grond hammer of the underworld'),
        ('echo', 'you chose: army move to gap of rohan'),
        ('pass', 'pass'),
        ('concede', 'gg'),
        ('hunt_roll', 'rolling for hunt'),
        ('discarding', 'discarding card'),
        ('reveal', 'reveal'),
        ('wwsat', 'wwsat'),
        ('balrog', 'balrog'),
        ('corsairs', 'corsairs to dol amroth, 3 left behind'),
        ('tile_face', '1r'),
        ('tile_face', 'eye'),
        ('mordor_entry', 'fsp enters mordor'),
        ('card_region', 'spirit of mordor: lorien'),
        ('hunt_tile', 'hunt tile: elven cloaks'),
        ('cc', 'cc: book of mazarbul'),
        ('refusal', chat.refusal_illegal('the fellowship cannot '
                                         'enter mordor mid-turn')),
        ('refusal', chat.refusal_closest('army move to gap of rohan',
                                         ['to'])),
    ]
    n_pass = 0
    for kind, text in allowed:
        try:
            out.say(kind, text)
            n_pass += 1
        except ChatPolicyError as e:
            fails.append(('allow_%s' % kind,
                          'refused %r: %s' % (text, e)))
    print('  [whitelist] %d/%d allowed specimens passed'
          % (n_pass, len(allowed)))
    denied = [
        # the named forbidden interfaces (kind alone refuses)
        ('casualty_menu', 'take 2 casualties'),
        ('discard_menu', 'discard a card'),
        ('retreat_prompt', "say 'retreat woodland realm' or 'stand'"),
        ('hand_list', 'strategy cards available'),
        ('pick_menu', 'choose one'),
        ('status', 'thinking...'),
        # unknown kind
        ('debug', 'harmless text'),
        # kind-laundering: menu content on an allowed kind (deny
        # beats keep - the buglist F4/E1 specimens verbatim)
        ('prompt', 'YOUR casualty allocation decision: 1) kill '
         'units only 2) downgrade 1 elite(s)'),
        ('prompt', "answer in chat: 'pick N'"),
        ('card_announce', 'plays a) book of mazarbul b) elven rope'),
        # unregistered free prose
        ('prompt', 'a prompt nobody built'),
        # the panel nag must never reach chat (no builder registers
        # prose under 'prompt' any more: the panel asks, not chat)
        ('prompt', "waiting: say 'nc' or name your combat card"),
        # wrong form for the kind
        ('hit_count', '1 / 0'),
        ('zero_eyes', '2 eyes allocated'),
    ]
    n_deny = 0
    for kind, text in denied:
        try:
            out.say(kind, text)
            fails.append(('deny_%s' % kind,
                          'ACCEPTED forbidden emission %r' % (text,)))
        except ChatPolicyError:
            n_deny += 1
    print('  [whitelist] %d/%d forbidden specimens refused'
          % (n_deny, len(denied)))
    # accepted texts reached the sink exactly once each
    if len(out.sent) != n_pass:
        fails.append(('sink_count', '%d sent vs %d passed'
                      % (len(out.sent), n_pass)))


def test_chat_card_disclosure_backstop(fails):
    """MEASURES: the content backstop - a message naming TWO distinct
    cards is refused however phrased (hand enumeration), ONE card
    passes (the mandated announce grammar), and the USER-allowed
    'playable:' form is exempt.  Non-zero control first: the card
    vocabulary must actually be loaded (>= 50 titles) or this reports
    NOT-MEASURED instead of a hollow green."""
    nv = len(chat._card_vocab())
    if nv < 50:
        fails.append(('vocab_control',
                      'NOT-MEASURED: card vocabulary has %d entries'
                      % nv))
        return
    print('  [backstop] vocabulary carries %d name forms' % nv)
    out = ChatOut()
    try:
        out.say('card_announce', 'plays book of mazarbul')
    except ChatPolicyError as e:
        fails.append(('one_card_ok', 'single-card announce refused: '
                      '%s' % e))
    raised = False
    try:
        out.say('card_announce',
                'plays book of mazarbul and elven rope')
    except ChatPolicyError:
        raised = True
    if not raised:
        fails.append(('two_cards_refused',
                      'a two-card enumeration walked through'))
    try:
        out.say('playable', 'playable: book of mazarbul, elven rope')
    except ChatPolicyError as e:
        fails.append(('playable_exempt',
                      'the sanctioned playable line refused: %s' % e))


# ================================================================ (d)
def test_match_choice_specimens(fails):
    """MEASURES: match_choice/classify_command on the REAL chat bodies
    from LIVE_0818_BUGLIST + the corpus grammar: 'fp ok', '0e', 'nc',
    'ok nc', 'retreat woodland realm', 'pick 3', 'ff 2', 'cc <name>',
    'field battle', 'random', magnitude statements (incl. the stated
    0 the old index path could not express), checksum stripping.
    Sensitivity AND specificity per family."""
    cid_bm = 'FREE_PEOPLES_Strategy_Book_of_Mazarbul'
    cid_hu = 'FREE_PEOPLES_Strategy_Help_Unlooked_For'
    menu_cc = [_A('combat_card_none'),
               _A('combat_card_play', {'card_id': cid_bm}),
               _A('combat_card_play', {'card_id': cid_hu})]
    # nc, first and last token (corpus A4)
    if chat.match_choice('nc', menu_cc) is not menu_cc[0]:
        fails.append(('nc_first', 'nc did not pick combat_card_none'))
    if chat.match_choice('ok nc', menu_cc) is not menu_cc[0]:
        fails.append(('nc_last', "'ok nc' missed (A4 last-token)"))
    # cc <name> and bare card name (+ trailing checksum token)
    if chat.match_choice('cc book of mazarbul', menu_cc) \
            is not menu_cc[1]:
        fails.append(('cc_name', 'cc book of mazarbul missed'))
    if chat.match_choice('book of mazarbul a1b2c3', menu_cc) \
            is not menu_cc[1]:
        fails.append(('bare_name_checksum',
                      'bare name with trailing hash missed'))
    if chat.match_choice('book', menu_cc) is not None:
        pass  # substring both ways CAN match short forms - not pinned
    # fp ok is a protocol command, not a menu answer
    if chat.classify_command('fp ok') != ('fp_ok', None):
        fails.append(('fp_ok', repr(chat.classify_command('fp ok'))))
    if chat.match_choice('fp ok', menu_cc) is not None:
        fails.append(('fp_ok_not_menu', "'fp ok' matched a menu item"))
    # VICTOR's advance decision (real-client leg final1, 2026-08-22: the
    # human Shadow wiped the AI's army and had NO chat word to decline
    # the advance - 'stay' only knew the retreat menus): 'stay' /
    # 'stand' / 'hold' / 'no advance' -> victor_stay; the retreat
    # menus keep their own answer when THEY are open (specificity)
    menu_va = [_A('victor_advance', {'n_regular': 1, 'n_elite': 0}),
               _A('victor_stay')]
    for w in ('stay', 'stand', 'hold', 'no advance', 'stay put'):
        if chat.match_choice(w, menu_va) is not menu_va[1]:
            fails.append(('victor_stay_' + w.replace(' ', '_'),
                          repr(chat.match_choice(w, menu_va))))
    menu_rt = [_A('combat_retreat', {'to': 'x'}), _A('combat_no_retreat')]
    if chat.match_choice('stay', menu_rt) is not menu_rt[1]:
        fails.append(('stay_retreat_menu', 'no_retreat not picked'))
    if chat.match_choice('advance', menu_va) is not None:
        pass   # the advance itself is a board gesture (units move)
    # 0e -> eye_allocate count=0 (D3); sensitivity: no eye menu -> None
    menu_eye = [_A('eye_allocate', {'count': n}) for n in (0, 1, 2)]
    got = chat.match_choice('0e', menu_eye)
    if got is None or got.params['count'] != 0:
        fails.append(('zero_e', repr(got)))
    got = chat.match_choice('2e', menu_eye)
    if got is None or got.params['count'] != 2:
        fails.append(('two_e', repr(got)))
    if chat.match_choice('0e', menu_cc) is not None:
        fails.append(('zero_e_specificity',
                      "'0e' matched on a non-eye menu"))
    # retreat <region> - both params spellings; unknown region -> None
    menu_ret = [_A('combat_retreat', {'to_region': 'Woodland Realm'}),
                _A('combat_retreat', {'to': 'Mirkwood'}),
                _A('combat_no_retreat')]
    got = chat.match_choice('retreat woodland realm', menu_ret)
    if got is not menu_ret[0]:
        fails.append(('retreat_region', repr(got)))
    got = chat.match_choice('retreat mirkwood', menu_ret)
    if got is not menu_ret[1]:
        fails.append(('retreat_to_key', repr(got)))
    if chat.match_choice('retreat gap of rohan', menu_ret) is not None:
        fails.append(('retreat_unknown',
                      'unknown region matched a retreat'))
    if chat.match_choice('stand', menu_ret) is not menu_ret[2]:
        fails.append(('stand', "'stand' missed combat_no_retreat"))
    if chat.match_choice('field battle', menu_ret) is not menu_ret[2]:
        fails.append(('field_battle_inbound',
                      "'field battle' missed combat_no_retreat (F2)"))
    # pick N (LEGACY) = 1-based menu index
    if chat.match_choice('pick 3', menu_ret) is not menu_ret[2]:
        fails.append(('pick_n', "'pick 3' missed menu[2]"))
    if chat.match_choice('pick 9', menu_ret) is not None:
        fails.append(('pick_range', 'out-of-range pick matched'))
    # magnitude statements (G3): bare number = COUNT on an effect
    # menu, incl. the stated 0; legacy index elsewhere (specificity)
    menu_mag = [_A('combat_effect_choice',
                   {'card_id': 'SHADOW_Character_Relentless_Assault',
                    'count': n}) for n in (0, 1, 2)]
    got = chat.match_choice('0', menu_mag)
    if got is None or got.params['count'] != 0:
        fails.append(('magnitude_zero',
                      'stated 0 not matched to count=0: %r' % (got,)))
    got = chat.match_choice('2', menu_mag)
    if got is None or got.params['count'] != 2:
        fails.append(('magnitude_two', repr(got)))
    if chat.match_choice('1', menu_ret) is not menu_ret[0]:
        fails.append(('index_legacy',
                      'bare 1 on a plain menu lost index semantics'))
    # 'ff' means Nazgul leadership forfeiture for Dread and Despair;
    # it must not spend army casualties for Relentless Assault.
    menu_forfeit = [_A('combat_effect_choice',
                       {'card_id': 'SHADOW_Character_Candles_of_Corpses',
                        'count': n}) for n in (1, 2, 3)]
    got = chat.match_choice('ff 2', menu_forfeit)
    if got is None or got.params['count'] != 2:
        fails.append(('ff_spaced', repr(got)))
    got = chat.match_choice('ff1', menu_forfeit)
    if got is None or got.params['count'] != 1:
        fails.append(('ff_joined', repr(got)))
    for body in ('ff 2', 'ff1'):
        if chat.match_choice(body, menu_mag) is not None:
            fails.append(('ff_not_army_casualties', body))
    # hunt grammar
    menu_hunt = [_A('hunt_sacrifice_random'),
                 _A('hunt_take_corruption'),
                 _A('hunt_sacrifice_guide')]
    if chat.match_choice('random', menu_hunt) is not menu_hunt[0]:
        fails.append(('hunt_random', "'random' missed"))
    if chat.match_choice('take the corruption', menu_hunt) \
            is not menu_hunt[1]:
        fails.append(('hunt_corruption', 'corruption phrase missed'))
    # protocol commands
    for body, want in (('pass null', ('pass', None)),
                       ('RESYNC', ('resync', None)),
                       ('press', ('press', None)),
                       ('stop', ('stop', None)),
                       ('0e', ('eyes', 0)),
                       ('field battle', ('field_battle', None)),
                       ('cc book of mazarbul',
                        ('cc', 'book of mazarbul')),
                       ('retreat woodland realm',
                        ('retreat', 'woodland realm')),
                       ('pick 3', ('pick', 3)),
                       ('ff2', ('ff', 2))):
        got = chat.classify_command(body)
        if got != want:
            fails.append(('classify_%s' % body.split()[0],
                          '%r -> %r (want %r)' % (body, got, want)))
    # siege legacy tolerance (F2 specimen forms)
    menu_sg = [_A('retreat_into_siege'), _A('combat_no_retreat')]
    if chat.match_choice('retreat into siege', menu_sg) \
            is not menu_sg[0]:
        fails.append(('siege_legacy',
                      "'retreat into siege' missed retreat_into_siege"))


# ================================================================ (e)
def test_region_short_forms(fails):
    """MEASURES (USER 2026-08-23 'region list + accepted shortforms'):
    chat.resolve_region - the ONE chat region resolver - accepts the
    exact name (case/accents/punctuation folded), the possessive
    without its apostrophe, a unique prefix, a unique word-prefix
    sequence; an ambiguous short form returns NO key and the candidate
    list (never a guess); nonsense returns neither."""
    import board
    b = board.load_board()
    keys = list((getattr(b, '_regions', None) or {}).keys())
    want = [("Helm's Deep", 'helm s deep'), ('helms deep', 'helm s deep'),
            ('goblins gate', 'goblin s gate'), ('Barad-Dur', 'barad dur'),
            ('osg', 'osgiliath'), ('fords of i', 'fords of isen'),
            ('tirith', 'minas tirith'), ('n ithilien', 'north ithilien'),
            ('shire', 'the shire'), ('west emyn', 'western emyn muil'),
            ('fords', 'fords of isen'), ('MM', 'minas morgul')]
    for text, key in want:
        got, _c = chat.resolve_region(text, keys)
        if got != key:
            fails.append(('region_form_' + text.replace(' ', '_'),
                          (got, key)))
    for text, n in [('minas', 2), ('rhun', 3),
                    ('mirkwood', 4), ('dol', 2)]:
        got, c = chat.resolve_region(text, keys)
        if got is not None or len(c) != n:
            fails.append(('region_ambiguous_' + text, (got, c)))
    got, c = chat.resolve_region('not a place', keys)
    if got is not None or c:
        fails.append(('region_nonsense', (got, c)))
    # the refusal names the candidates and is a registered builder text
    txt = chat.region_ambiguous('fords', ['fords of bruinen',
                                          'fords of isen'])
    if 'fords of bruinen' not in txt or 'fords of isen' not in txt:
        fails.append(('region_ambiguous_text', txt))
    try:
        chat.ChatOut().say('refusal', txt)
    except Exception as e:
        fails.append(('region_ambiguous_policy', repr(e)[:100]))


def test_declined_leg(fails):
    """MEASURES: declined_leg fires ONLY when a real alternative
    existed (USER 2026-08-18: a chain that ran out of legs is a forced
    stop, stays silent) and renders the corpus '<card>: pass' frame."""
    term = _A('card_option',
              {'card_id': 'SHADOW_Strategy_The_Shadow_Lengthens',
               'choices': {'_tsl_done': True}})
    real = _A('card_option',
              {'card_id': 'SHADOW_Strategy_The_Shadow_Lengthens',
               'choices': {'to': 'North Rhun'}})
    got = chat.declined_leg(term, [term, real])
    if got != 'the shadow lengthens: pass':
        fails.append(('decline_phrase', repr(got)))
    if chat.declined_leg(term, [term]) is not None:
        fails.append(('decline_forced_stop',
                      'announced a forced stop as a decision'))
    if chat.declined_leg(real, [term, real]) is not None:
        fails.append(('decline_specificity',
                      'announced a TAKEN leg as a decline'))
    try:
        ChatOut().say('leg_decline', got)
    except ChatPolicyError as e:
        fails.append(('decline_exit', str(e)))


def main():
    tests = [test_region_short_forms, 
        test_parse_request_grammar,
        test_bridge_roundtrip_and_dedup,
        test_response_atomicity,
        test_single_writer_lock,
        test_panel_status_and_pause,
        test_chat_whitelist,
        test_chat_card_disclosure_backstop,
        test_match_choice_specimens,
        test_declined_leg,
    ]
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
