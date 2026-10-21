"""Scenario-mode pins (contract K-S1): the prefix lane, the LOUD
desync, the live flip, and the fixture cutter's alignment - driven
through the REAL server (LiveServer in --scenario mode, netless) over
a REAL recorded fixture (lab\\scenarios\\<name>\\: the client's own raw
log, rendered feed and sidecar), never a hand-written feed.

These are regression pins of the instrument.  The evidence for any
flow is the real-client run (lab\\scenarios\\run_scenario.py); nothing
here claims client agreement.
"""
import json
import os
import shutil
import sys
import tempfile

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
_BRIDGE = os.path.join(_ROOT, 'wotr_bridge')
_LAB = os.path.join(os.path.dirname(_ROOT), 'WOTR_AI_training_lab')
_SCEN = os.path.join(_LAB, 'scenarios')
for _p in (_SRC, _ROOT, _BRIDGE, _SCEN):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from game_state import Side                              # noqa: E402
from wotr_bridge.live import scenario as scen            # noqa: E402
from wotr_bridge.live.driver import DriverHalt           # noqa: E402
from wotr_bridge.live.live_server import LiveServer      # noqa: E402

# the pinned fixture: a real recorded position (human FP, AI Shadow)
FIXTURE = os.path.join(_SCEN, 'ai_attacks_human_defends')
# the session it was cut from (raw log + feed + sidecar)
SESSION = os.path.join(_LAB, 'audit', 'live_boot_run',
                       'fw1cbt2_artifacts')


def _read(path):
    with open(path, 'rb') as f:
        return f.read().decode('utf-8', 'replace')


def _lines(path):
    out = [ln.rstrip('\r') for ln in _read(path).split('\n')]
    if out and out[-1] == '':
        out.pop()
    return out


def _dump_with_ai_hand(src_dump, dst, pids):
    """The session's real board dump with the AI hand zone (SA Hand =
    Controls.46) rewritten to exactly `pids`."""
    out = []
    for ln in _lines(src_dump):
        parts = ln.split('\t')
        if len(parts) >= 3 and parts[1].strip() == '<~Controls.46~>':
            parts[2] = ' '.join(pids)
            if len(parts) > 3:
                parts[3] = ' '.join('U' for _ in pids)
            ln = '\t'.join(parts)
        out.append(ln)
    with open(dst, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out) + '\n')


class _Rig(object):
    """A temp client dir fed line by line from the fixture."""

    def __init__(self, fixture=None, seed=42):
        self.td = tempfile.mkdtemp(prefix='scen_')
        self.sc = scen.Scenario(fixture or FIXTURE)
        self.feed_lines = list(self.sc.prefix)
        # the raw stream as the client produces it in a run: ITS OWN
        # hand claim (fresh commitment tables; the fixture carries no
        # claim line) followed by the replayed prefix
        self.claim = self._session_claim()
        self.raw_lines = [self.claim] + self.sc.raw_prefix()
        self.feed = os.path.join(self.td, 'ai_feed.txt')
        self.meta = self.feed + '.meta.jsonl'
        open(self.feed, 'w').close()
        self.meta_lines = _lines(self.sc.meta_path)
        open(self.meta, 'w').close()
        self.dump = os.path.join(self.td, 'ai_client_board.txt')
        self.srv = LiveServer(self.td, Side.SHADOW if self.sc.ai_side
                              == 'SA' else Side.FREE_PEOPLES,
                              load_nets=False, seed=seed,
                              scenario=self.sc)
        self.n_feed = 0

    def _session_claim(self):
        tag = '<~Controls.479~>' if self.sc.ai_side == 'SA'             else '<~Controls.464~>'
        src = self.sc.spec['setup']['source']
        raw_p = os.path.join(src['session'], src['raw_log'])
        for ln in scen.load_raw_log(raw_p):
            if tag in ln:
                return scen.strip_raw_hash(ln)
        raise RuntimeError('no %s claim line in %s' % (tag, raw_p))

    def append_feed(self, k):
        with open(self.feed, 'a', encoding='utf-8') as f:
            for ln in self.feed_lines[self.n_feed:self.n_feed + k]:
                f.write(ln + '\n')
        with open(self.meta, 'a', encoding='utf-8') as f:
            for ln in self.meta_lines[self.n_feed:self.n_feed + k]:
                f.write(ln + '\n')
        self.n_feed += k

    def append_line(self, line, meta=None):
        with open(self.feed, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
        with open(self.meta, 'a', encoding='utf-8') as f:
            f.write((meta or json.dumps({'seq': self.n_feed + 1,
                                         'gen': 1, 'tags': [],
                                         'pids': []})) + '\n')
        self.n_feed += 1

    def cycle(self, raw=None):
        return self.srv.handle({'lines': list(raw if raw is not None
                                              else self.raw_lines),
                                'side': 'AUTO'})

    def close(self):
        shutil.rmtree(self.td, ignore_errors=True)


def _ai_hand_pids(rig):
    """The AI-side draw pids of the fixture's raw prefix (the
    commitment decode's own input), in draw order."""
    from wotr_bridge.live import card_identity as ci
    out = []
    for ln in rig.raw_lines:
        m = ci.RAW_DRAW_RE.search(ln)
        if m and ci.DECK_INFO[m.group(2)][0] == rig.sc.ai_side_name:
            out.append(m.group(1))
    return out


def test_fixture_shape(fails):
    """MEASURES: the pinned fixture exists with the K-S1 files and its
    prefix replays clean through the prefix lane (the cutter's proof,
    re-run here so a drifted lane fails the pin, not the next run)."""
    for n in ('scenario.json', 'setup_raw.log', 'setup_feed.txt',
              'setup_feed.meta.jsonl', 'README.md'):
        if not os.path.exists(os.path.join(FIXTURE, n)):
            fails.append(('fixture_file', n))
            return
    sc = scen.Scenario(FIXTURE)
    if sc.cut <= 0 or len(sc.prefix) != sc.cut:
        fails.append(('fixture_cut', (sc.cut, len(sc.prefix))))
        return
    drv = scen.replay_prefix(sc.feed_path, sc.cut, sc.ai_side_name,
                             seed=42, meta_path=sc.meta_path,
                             expected=sc.prefix)
    probs = drv.prefix_problems()
    if probs:
        fails.append(('prefix_lane_clean', probs[:4]))
    if drv.prefix_seen != sc.cut:
        fails.append(('prefix_seen', (drv.prefix_seen, sc.cut)))
    summ = scen.position_summary(drv)
    if summ['turn'] < 1 or summ['phase'] != 'actions':
        fails.append(('position', (summ['turn'], summ['phase'])))
    if summ != sc.spec.get('position'):
        fails.append(('position_vs_fixture',
                      scen.summary_diff(sc.spec.get('position') or {},
                                        summ)[:4]))
    print('  [fixture] %s cut=%d: %s' % (sc.name, sc.cut,
                                         scen.summary_lines(summ)[0]))


def test_prefix_then_flip(fails):
    """MEASURES (K-S1): the server refuses to act while the prefix
    streams in (no RESP lines, status live=false, prefix_seen climbs),
    the AI-side draws are decoded from the raw stream but NOT fed to
    the funnel, the flip waits for a client dump whose AI hand
    agrees, adopts the decoded identities, and the driver is live
    afterwards with the fixture's position."""
    rig = _Rig()
    try:
        srv, sc = rig.srv, rig.sc
        n = sc.cut
        half = n // 2
        rig.append_feed(half)
        out = rig.cycle()
        if out:
            fails.append(('no_resp_in_prefix', out[:3]))
        st = srv.status_menu().get('scenario') or {}
        if st.get('live') is not False or st.get('prefix_seen') != half \
                or st.get('prefix_expected') != n:
            fails.append(('status_mid_prefix', st))
        # the AI's funnel must be EMPTY in the prefix lane (the lane
        # draws placeholders; the decode is banked for the flip)
        if any(v for v in srv.rand._feed['cards'].values()):
            fails.append(('funnel_fed_in_prefix',
                          {k: v for k, v in srv.rand._feed['cards'].items()
                           if v}))
        if not srv._scenario_pid_cid:
            fails.append(('decode_banked', 'no AI draw decoded'))
        rig.append_feed(n - half)
        out = rig.cycle()
        if out:
            fails.append(('no_resp_at_prefix_end', out[:3]))
        if srv.driver.live:
            fails.append(('flip_without_dump', 'flipped before any dump'))
        st = srv.status_menu().get('scenario') or {}
        if not st.get('flip_due') or st.get('prefix_seen') != n:
            fails.append(('status_flip_due', st))
        # a dump whose AI hand DISAGREES with the engine count must not
        # flip (waits, loudly noted)
        pids = _ai_hand_pids(rig)
        _dump_with_ai_hand(os.path.join(SESSION, 'ai_client_board.txt'),
                           rig.dump, pids[:1])
        rig.cycle()
        if srv.driver.live:
            fails.append(('flip_on_count_mismatch', 'flipped'))
        if not any('AI hand' in str(x) for x in srv.notes):
            fails.append(('mismatch_noted', srv.notes[-3:]))
        # the agreeing dump flips
        _dump_with_ai_hand(os.path.join(SESSION, 'ai_client_board.txt'),
                           rig.dump, pids)
        rig.cycle()
        if not srv.driver.live:
            fails.append(('flipped', srv.notes[-3:]))
            return
        st = srv.status_menu().get('scenario') or {}
        if st.get('live') is not True:
            fails.append(('status_live', st))
        hand = srv.driver.es.hands[srv.ai_side].hand
        want = [srv._scenario_pid_cid[p] for p in pids]
        if sorted(hand) != sorted(want):
            fails.append(('ai_hand_adopted', (hand, want)))
        if any(str(c).startswith('HU_') for c in hand):
            fails.append(('placeholders_left', hand))
        for cid, pid in zip(want, pids):
            if srv.driver.view.card_pid(cid) != pid:
                fails.append(('card_pid_map', (cid, pid,
                                               srv.driver.view.card_pid(
                                                   cid))))
        # conservation: no adopted cid may still sit in a deck
        es = srv.driver.es
        for attr in ('sh_char_deck', 'sh_strat_deck', 'fp_char_deck',
                     'fp_strat_deck'):
            dl = getattr(es, attr, []) or []
            for c in want:
                if c in dl:
                    fails.append(('deck_still_holds', (attr, c)))
        summ = scen.position_summary(srv.driver)
        ref = sc.spec.get('position') or {}
        diffs = scen.summary_diff(ref, summ)
        if diffs:
            fails.append(('position_at_flip', diffs[:4]))
        if srv.driver.nick != srv.nick or srv.driver.spectator:
            fails.append(('live_identity', (srv.driver.nick,
                                            srv.driver.spectator)))
        # live behaviour after the flip: the human's chat pass hands
        # the action to the AI, whose netless step emits something
        rig.append_line('<Wey> pass')
        out = rig.cycle()
        out2 = rig.cycle()
        if not (out or out2):
            fails.append(('ai_acts_after_flip', srv.notes[-3:]))
        print('  [flip] %s' % '; '.join(srv.driver.flip_report))
        print('  [post-flip RESP] %s' % ((out or out2)[:3],))
    finally:
        rig.close()


def test_desync_is_loud(fails):
    """MEASURES: a prefix line the client renders differently halts
    the server with 'SCENARIO DESYNC' naming the line; a parser no-op
    extra (the client's own re-claim 'views the hand' paragraph) is
    skipped, never a desync."""
    rig = _Rig()
    try:
        srv = rig.srv
        # through the fixture's own claim render ('views the hand')
        k = next(i for i, ln in enumerate(rig.feed_lines)
                 if ' views the ' in ln) + 1
        rig.append_feed(k)
        rig.cycle()
        # the client's re-claim renders a SECOND no-op paragraph: skipped
        rig.append_line(rig.feed_lines[k - 1])
        rig.cycle()
        if srv.driver.prefix_seen != k or not srv.driver.prefix_skipped:
            fails.append(('extra_noop_skipped',
                          (srv.driver.prefix_seen,
                           srv.driver.prefix_skipped)))
        # a content line that differs: LOUD, naming the line
        rig.append_line('$Wey draws Shadow Strategy Card.')
        try:
            rig.cycle()
            fails.append(('desync_loud', 'no halt'))
        except DriverHalt as e:
            if not str(e).startswith('SCENARIO DESYNC at line %d' % (k + 1)):
                fails.append(('desync_text', str(e)[:160]))
            print('  [desync] %s' % str(e)[:120])
    finally:
        rig.close()


def test_claim_race_is_loud(fails):
    """MEASURES: two DIFFERENT AI-side commitment tables in the raw
    stream (the client claimed fresh tables before the fixture's claim
    replayed) refuse the flip loudly."""
    rig = _Rig()
    try:
        srv = rig.srv
        rig.append_feed(rig.sc.cut)
        claim = rig.claim
        tag = '<~Controls.479~>'
        head, payload = claim.split(tag, 1)
        toks = payload.split()
        toks[0] = 'F' * len(toks[0])
        other = head + tag + ' '.join(toks)
        try:
            rig.cycle(raw=rig.raw_lines + [other])
            fails.append(('claim_race_loud', 'no halt'))
        except DriverHalt as e:
            if 'SCENARIO CLAIM RACE' not in str(e):
                fails.append(('claim_race_text', str(e)[:160]))
            print('  [claim race] %s' % str(e)[:100])
    finally:
        rig.close()


def test_cutter_alignment(fails):
    """MEASURES: make_scenario.align ties the prefix feed lines to the
    raw log by content (nick + tags + pids), ends the raw prefix before
    the raw line that renders line N+1, and refuses a cut inside a
    gesture window.  Self-contained over the permanent fw1cbt2 session
    (human FP, AI Shadow), independent of any shipped fixture."""
    import make_scenario as mk
    feed = mk.read_lines(os.path.join(SESSION, 'ai_feed.txt'))
    meta = mk.read_lines(os.path.join(SESSION, 'ai_feed.txt.meta.jsonl'))
    raw_p = mk.find_raw_log(SESSION)
    raw_all = scen.load_raw_log(raw_p)
    ng = scen.new_game_index(raw_all)
    CUT = 17
    start, end_excl, rows, problems = mk.align(feed, meta, raw_all,
                                               CUT, ng)
    if problems:
        fails.append(('align_problems', problems[:3]))
    if start != ng + 1:
        fails.append(('align_start', (start, ng + 1)))
    if not (rows and end_excl > start):
        fails.append(('align_range', (start, end_excl, len(rows))))
    by_seq = {}
    for ln in meta:
        r = json.loads(ln)
        by_seq[r['seq']] = r
    # every matched raw line carries the record's first tag, in order
    prev = start - 1
    for seq, j in rows:
        if j <= prev:
            fails.append(('align_monotone', (seq, j, prev)))
        prev = j
        tags = by_seq[seq].get('tags') or []
        if tags and ('<~%s~>' % tags[0]) not in raw_all[j]:
            fails.append(('align_tag', (seq, j, tags[0])))
    # the first raw_new line AFTER the cut is excluded from the prefix
    nxt = by_seq.get(CUT + 1)
    if nxt and nxt.get('raw_new') and nxt.get('nick'):
        if any(mk.raw_matches(raw_all[j], nxt)
               for j in range(start, end_excl)):
            fails.append(('next_line_excluded', CUT + 1))
    # a mid-window cut is refused by the lane proof (open window)
    mid = None
    for i, ln in enumerate(feed):
        if ' from FP Dice to Current Action.' in ln:
            mid = i + 1
            break
    if mid is None:
        fails.append(('mid_window_specimen', 'none in the session'))
        return
    drv = scen.replay_prefix(os.path.join(SESSION, 'ai_feed.txt'), mid,
                             'SA', seed=42,
                             meta_path=os.path.join(
                                 SESSION, 'ai_feed.txt.meta.jsonl'),
                             expected=feed[:mid])
    probs = drv.prefix_problems()
    if not any('open at cut' in p for p in probs):
        fails.append(('mid_window_refused', probs[:3]))
    print('  [cutter] raw %d..%d, %d rows; mid-window cut at %d -> %s'
          % (start + 1, end_excl, len(rows), mid,
             [p for p in probs if 'open' in p][:1]))


def main():
    fails = []
    tests = [test_fixture_shape, test_prefix_then_flip,
             test_desync_is_loud, test_claim_race_is_loud,
             test_cutter_alignment]
    for t in tests:
        n0 = len(fails)
        print('-- %s' % t.__name__)
        try:
            t(fails)
        except Exception as e:
            import traceback
            traceback.print_exc()
            fails.append((t.__name__, 'EXC %r' % (e,)))
        print('   %s' % ('ok' if len(fails) == n0 else 'FAIL %r'
                         % (fails[n0:],)))
    print('%s: %d failure(s)' % (os.path.basename(__file__), len(fails)))
    for f in fails:
        print('  FAIL', f)
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
