"""Scenario mode (contract K-S1): a SAVED POSITION replayed onto the
real client, verified line by line against the fixture, then LIVE.

A scenario fixture (lab\\scenarios\\<name>\\) carries the client's raw
op log up to a cut (setup_raw.log - the client REPLAYS it through its
own line executor, game.z.a per line) and the rendered feed those raw
lines produced (setup_feed.txt, N lines).  The server runs the live
feed's first N lines through the PREFIX LANE - the spectator lane:
both seats external, every roll/draw/tile fed from the log, no AI
step - verifying every line against the fixture (normalized; a
mismatch is a LOUD DriverHalt 'SCENARIO DESYNC').  At line N the
driver FLIPS to live: the AI owns its seat from there (claims,
decides, emits), the human's gestures are matched as in any live
game.

INVARIANTS
  * the engine state after the prefix is produced by the same lane the
    cutter (lab\\scenarios\\make_scenario.py) proved the fixture with:
    replay_prefix() is that ONE implementation;
  * the prefix lane never emits to the client and never feeds the
    AI's card funnel: the AI's hand identities are adopted at the flip
    from the client's own statement (the board dump's AI-hand pids)
    through the commitment decode (sanctioned raw import #1) - the
    human's pids never decode (poison password, card_identity);
  * the flip refuses (loudly) while the client dump and the engine
    disagree on the AI hand, while the prefix lane holds anything
    open (window / deferral / pending buffer), or when the AI-side
    claim lines in the raw stream carry different commitment tables
    (the client claimed fresh tables before the fixture's claim line
    replayed: the run must be repeated, never played on);
  * a live feed line that is NOT in the prefix (a second 'views the
    hand' paragraph from the client's own re-claim, a <game> note) is
    skipped ONLY when it is a parser no-op/system line; anything that
    carries game content must match or the lane halts.
"""
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, '..', '..', 'src'))
_BRIDGE = os.path.abspath(os.path.join(_HERE, '..'))
for _p in (_SRC, _BRIDGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from . import chat as chat_mod
from . import live_parser
from .driver import DriverHalt, LiveDriver
from .segmenter import Segmenter
from .spectator import SpectatorDriver

# the client's auto-log terminates every line with a checksum token
# (6+ hex chars) or the literal 'null'; the injectable op line is the
# text before it (AiOpp.inject feeds game.z.a(line, null))
_RAW_TAIL_RE = re.compile(r'\s+(?:[0-9A-F]{6,}|null)\s*$')
# the client's own new-game record: everything before it is the boot
# preamble the scratch client produces itself (setup Scribe lines,
# turn marker); the injectable prefix starts AFTER it
_NEW_GAME_TAG = '<~Controls.408~>'
# the AI-side hand-claim tag per side (commitment token broadcast)
_CLAIM_TAG = {'FREE_PEOPLES': '<~Controls.464~>',
              'SHADOW': '<~Controls.479~>'}
_HAND_ZONE = {'FREE_PEOPLES': 'fp hand', 'SHADOW': 'sa hand'}


def strip_raw_hash(line):
    """One auto-log line -> the injectable op line (checksum dropped)."""
    return _RAW_TAIL_RE.sub('', line.rstrip('\r\n'))


def load_raw_log(path):
    with open(path, 'rb') as f:
        txt = f.read().decode('utf-8', 'replace')
    return [ln for ln in txt.split('\n') if ln.strip()]


def new_game_index(raw_lines):
    """Index of the client's new-game record in a raw log, or None."""
    for i, ln in enumerate(raw_lines):
        if _NEW_GAME_TAG in ln:
            return i
    return None


def prefix_key(line):
    """Comparison key of one feed line: the parser's own normalization
    (so '%Nick' replay prefixes and '$Nick' live prefixes agree), with
    the machine-specific system lines reduced to their tag."""
    s = line.rstrip()
    if s.startswith('<address>'):
        return ('system', '<address>')
    kind, text = live_parser.normalize(s)
    if kind == 'board' and text.startswith('<game> ') \
            and 'has started a new War of the Ring game' in text:
        return ('system', '<game> new game')
    return (kind, text.rstrip())


def skippable_extra(line):
    """May a live line that does NOT match the expected prefix line be
    consumed silently?  Only parser no-ops ('views the hand' from the
    client's own re-claim, 'deselects') and untagged system notes."""
    rec = live_parser.parse(line)
    return rec.get('op') in ('noop', 'system', 'blank')


_DICE_ZONES = frozenset(('fp dice', 'sa dice', 'fp casualties',
                        'sa casualties', 'the hunt for the ring',
                        'current action', 'sa reinforcements',
                        'fp reinforcements'))


def reorderable_line(line):
    """A bookkeeping paragraph whose ORDER shifts run to run and so is
    matched inside its contiguous run as a MULTISET, not in sequence:
      * a no-op / system / blank line (address lines, 'views the ...
        hand', the #BLIND attestation, the new-game note);
      * a '<replay> Turn N' marker (the client may render it more than
        once, and it swaps order with the roster restore);
      * an Action-Die move among the pool / casualties / hunt-box /
        current-action / reinforcement zones (the turn-rollover dice
        roster restore, the eye returns, the bonus-die additions) -
        NOT a real gesture and NOT eye ALLOCATION into the Hunt box
        (that leaves a pool for the hunt and is a decision; excluded
        because its destination is the hunt box while its source is a
        live pool AND the phase is not a rollover - but a plain
        'SA Dice -> The Hunt For The Ring' is idempotent bookkeeping
        the driver routes either way, so admitting it into a run only
        relaxes ordering, never correctness)."""
    if skippable_extra(line):
        return True
    rec = live_parser.parse(line)
    op = rec.get('op')
    if op == 'turn':
        return True
    if op == 'move':
        obj = str(rec.get('obj', '') or '').lower()
        if 'action die' not in obj:
            return False
        frm = str(rec.get('from', '') or '').strip().lower()
        to = str(rec.get('to', '') or '').strip().lower()
        return frm in _DICE_ZONES and to in _DICE_ZONES
    return False


def claim_payload(raw_line, side_name):
    """The commitment-table payload of an AI-side claim line (tokens
    after the tag, checksum dropped), or None for any other line."""
    tag = _CLAIM_TAG[side_name]
    if tag not in raw_line:
        return None
    return strip_raw_hash(raw_line).split(tag, 1)[1].strip()


class Scenario(object):
    """One fixture directory, loaded and shape-checked."""

    def __init__(self, d):
        self.dir = os.path.abspath(d)
        p = os.path.join(self.dir, 'scenario.json')
        if not os.path.exists(p):
            raise DriverHalt('SCENARIO: no scenario.json in %s' % self.dir)
        with open(p, encoding='utf-8') as f:
            self.spec = json.load(f)
        self.name = str(self.spec.get('name') or
                        os.path.basename(self.dir))
        hs = str(self.spec.get('human_side', '')).upper()
        if hs not in ('FP', 'SA'):
            raise DriverHalt('SCENARIO %s: human_side must be FP|SA'
                             % self.name)
        self.human_side = hs
        self.ai_side = 'SA' if hs == 'FP' else 'FP'
        setup = dict(self.spec.get('setup') or {})
        self.cut = int(setup.get('cut', 0) or 0)
        self.raw_path = os.path.join(self.dir, setup.get(
            'raw_log', 'setup_raw.log'))
        self.feed_path = os.path.join(self.dir, setup.get(
            'feed', 'setup_feed.txt'))
        self.meta_path = os.path.join(self.dir, setup.get(
            'meta', 'setup_feed.meta.jsonl'))
        self.trigger = list(self.spec.get('trigger') or ())
        self.expect = list(self.spec.get('expect') or ())
        self.timeouts = dict(self.spec.get('timeouts') or {})
        self.seed = self.spec.get('seed')
        self.sims = self.spec.get('sims')
        self.nick = self.spec.get('nick')
        self.prefix = []
        if self.cut > 0:
            if not os.path.exists(self.feed_path):
                raise DriverHalt('SCENARIO %s: cut=%d but no feed %s'
                                 % (self.name, self.cut, self.feed_path))
            with open(self.feed_path, 'rb') as f:
                lines = f.read().decode('utf-8', 'replace').split('\n')
            lines = [ln.rstrip('\r') for ln in lines]
            if lines and lines[-1] == '':
                lines.pop()
            if len(lines) < self.cut:
                raise DriverHalt('SCENARIO %s: feed has %d lines, cut=%d'
                                 % (self.name, len(lines), self.cut))
            self.prefix = lines[:self.cut]

    @property
    def ai_side_name(self):
        return 'SHADOW' if self.ai_side == 'SA' else 'FREE_PEOPLES'

    @property
    def human_side_name(self):
        return 'SHADOW' if self.human_side == 'SA' else 'FREE_PEOPLES'

    def raw_prefix(self):
        """The injectable raw lines (checksums stripped), or [] when the
        fixture carries no raw log (cut 0 = a fresh game)."""
        if self.cut <= 0 or not os.path.exists(self.raw_path):
            return []
        return [strip_raw_hash(ln) for ln in load_raw_log(self.raw_path)]


# ----------------------------------------------------------------------
def position_summary(drv):
    """Engine-position facts a fixture is proven and re-verified by:
    everything the two lanes must agree on at the flip.  Identities of
    hidden cards never ride (counts only)."""
    import game_state as gs
    from board import sanitize
    es = drv.es
    armies = {}
    for k, a in es.armies.items():
        row = {}
        for u in a.units:
            key = '%s:%s' % (str(u.nation.name).lower(),
                             str(u.kind.name).lower())
            row[key] = row.get(key, 0) + 1
        nl = len(getattr(a, 'leaders', []) or [])
        if row or nl:
            if nl:
                row['leaders'] = nl
            armies[sanitize(str(k))] = row
    dice = {}
    for s in (gs.Side.FREE_PEOPLES, gs.Side.SHADOW):
        dice[s.name] = [[(d.current.name if d.current is not None
                          else None), bool(d.used)]
                        for d in es.dice.get(s, [])]
    hands = {s.name: len(es.hands[s].hand)
             for s in (gs.Side.FREE_PEOPLES, gs.Side.SHADOW)}
    political = {str(k.name): int(v)
                 for k, v in es.political.boxes.items()}
    war = sorted(k for k, v in political.items() if v >= 3)
    import ai
    dp = ai.describe_pending(es)
    fs = getattr(es, 'fellowship', None)
    fel = None
    if fs is not None:
        fel = {'progress': int(getattr(fs, 'progress', 0) or 0),
               'corruption': int(getattr(fs, 'corruption', 0) or 0),
               'hidden': bool(getattr(fs, 'hidden', True)),
               'mordor': bool(getattr(fs, 'mordor_entered', False))}
    return {
        'turn': int(getattr(es, 'turn', 0) or 0),
        'phase': str(getattr(drv, 'phase', '')),
        'last_actor': (drv.last_actor.name
                       if getattr(drv, 'last_actor', None) is not None
                       else None),
        'pending': (str(dp['kind']) if dp is not None else None),
        'pending_side': (dp['side'].name if dp is not None else None),
        'at_war': war,
        'political': political,
        'armies': armies,
        'dice': dice,
        'hunt_box': len(getattr(es, 'hunt_box', []) or []),
        'hands': hands,
        'fellowship': fel,
        'besieged': sorted(str(b) for b in
                           (getattr(es, 'besieged_strongholds', set())
                            or set())),
    }


def summary_lines(summary):
    """Human-readable position summary (the cutter prints it; the
    README quotes it)."""
    out = ['turn %s  phase %s  last_actor %s  pending %s (%s)'
           % (summary.get('turn'), summary.get('phase'),
              summary.get('last_actor'), summary.get('pending'),
              summary.get('pending_side')),
           'at war: %s' % (', '.join(summary.get('at_war') or ())
                           or '-'),
           'political: %s' % json.dumps(summary.get('political'),
                                       sort_keys=True),
           'hands: %s  hunt box: %s  fellowship: %s'
           % (summary.get('hands'), summary.get('hunt_box'),
              summary.get('fellowship'))]
    for s, faces in sorted((summary.get('dice') or {}).items()):
        out.append('dice %s: %s' % (s, ' '.join(
            '%s%s' % ((f or '-'), '*' if used else '')
            for f, used in faces)))
    for reg, row in sorted((summary.get('armies') or {}).items()):
        out.append('  %-24s %s' % (reg, ' '.join(
            '%s=%d' % (k, v) for k, v in sorted(row.items()))))
    if summary.get('besieged'):
        out.append('besieged: %s' % ', '.join(summary['besieged']))
    return out


def summary_diff(a, b):
    """Keys whose values differ between two summaries (first-level,
    armies per region)."""
    diffs = []
    for k in sorted(set(a) | set(b)):
        if k == 'armies':
            ra, rb = a.get(k) or {}, b.get(k) or {}
            for reg in sorted(set(ra) | set(rb)):
                if ra.get(reg) != rb.get(reg):
                    diffs.append('armies[%s]: %r != %r'
                                 % (reg, ra.get(reg), rb.get(reg)))
        elif a.get(k) != b.get(k):
            diffs.append('%s: %r != %r' % (k, a.get(k), b.get(k)))
    return diffs


# ----------------------------------------------------------------------
class ScenarioDriver(SpectatorDriver):
    """The spectator lane over the prefix, the live driver after the
    flip - ONE object, so every piece of driver bookkeeping the prefix
    built (die pid bindings, nick-side bindings, table seats, the
    human hand model, turn draws, phase) is the live driver's own."""

    # every SpectatorDriver override is dispatched on the flip flag;
    # the helpers the spectator lane calls internally (mirror, shim
    # lanes) need no dispatch - only the spectator lane reaches them
    _DISPATCHED = ('_route', '_pending_buffer_applies', '_on_draw',
                   '_ai_turn', '_on_eye', '_on_action_roll',
                   '_on_recover', '_on_chat', '_on_window',
                   '_match_window_now', '_on_undo', 'census')

    def __init__(self, engine_state, ai_side, feed_path, randomness,
                 expected_prefix, chat_out=None, dump_path=None,
                 nick='WOTR_AI'):
        import game_state as gs
        SpectatorDriver.__init__(self, engine_state, feed_path,
                                 randomness)
        # the flip restores these; the spectator lane must see NEITHER
        # seat as its own (nick '<spectator>' never self-echoes)
        self._ai_side_live = ai_side
        self._human_side_live = (gs.Side.SHADOW
                                 if ai_side is gs.Side.FREE_PEOPLES
                                 else gs.Side.FREE_PEOPLES)
        self._nick_live = nick
        self._dump_path = dump_path
        if chat_out is not None:
            self.chat = chat_out
        self.view.dump_path = dump_path
        self.prefix_expected = list(expected_prefix)
        self.prefix_n = len(self.prefix_expected)
        self.prefix_seen = 0
        # the client's boot PREAMBLE (attestation, new-game note, Turn
        # marker, address lines, the claim's 'views the hand'): its
        # paragraphs land in a boot-timing-dependent order and count
        # (a second 'views' from the mask re-assert), so the leading
        # run of no-op/system lines is matched as a MULTISET; content
        # lines are matched strictly, in order
        # runs[k] = (run_end_exclusive) for every k that begins a
        # maximal run (len >= 2) of reorderable bookkeeping lines; a
        # 1-line run is matched strictly (no ordering freedom to grant)
        self._run_end = {}
        i = 0
        exp = self.prefix_expected
        while i < len(exp):
            if reorderable_line(exp[i]):
                j = i
                while j < len(exp) and reorderable_line(exp[j]):
                    j += 1
                if j - i >= 2:
                    self._run_end[i] = j
                i = j
            else:
                i += 1
        self._run_pool = None            # keys remaining in the open run
        self._run_end_cur = None
        self.prefix_skipped = []        # (feed seq, line) extras
        self.prefix_items = []          # lane outputs (never sent)
        self.live = self.prefix_n == 0
        self.flip_due = self.live
        self.flip_report = []
        self.desync = None              # the first LOUD mismatch text
        if self.live:
            self._become_live()

    # ---- lane dispatch -------------------------------------------------
    def _lane(self):
        return LiveDriver if self.live else SpectatorDriver

    def _route(self, ev, payload):
        return self._lane()._route(self, ev, payload)

    def _pending_buffer_applies(self, dp):
        return self._lane()._pending_buffer_applies(self, dp)

    def _on_draw(self, rec):
        return self._lane()._on_draw(self, rec)

    def _ai_turn(self):
        return self._lane()._ai_turn(self)

    def _on_eye(self, rec):
        return self._lane()._on_eye(self, rec)

    def _on_action_roll(self, rec):
        if self.live:
            return LiveDriver._on_action_roll(self, rec)
        import game_state as gs
        was = self.phase
        out = SpectatorDriver._on_action_roll(self, rec)
        if was != 'actions' and self.phase == 'actions':
            # the live driver's alternation seed (FP acts first): the
            # prefix lane must leave the flip with the same answer
            self.last_actor = gs.Side.SHADOW
        return out

    def _on_recover(self, rec):
        return self._lane()._on_recover(self, rec)

    def _on_window(self, w):
        return self._lane()._on_window(self, w)

    def _match_window_now(self, w, side):
        if self.live:
            return LiveDriver._match_window_now(self, w, side)
        n0 = self.window_stats['matched']
        out = SpectatorDriver._match_window_now(self, w, side)
        if self.window_stats['matched'] > n0:
            self.last_actor = side       # alternation bookkeeping
        return out

    def _on_undo(self, ev, rec):
        return self._lane()._on_undo(self, ev, rec)

    def census(self):
        return self._lane().census(self)

    # ---- the prefix lane's chat protocol (both seats external) ----------
    def _on_chat(self, rec):
        if self.live:
            return LiveDriver._on_chat(self, rec)
        import ai
        import game_state as gs
        n_commits = self.n_commits_total
        out = SpectatorDriver._on_chat(self, rec)
        if self.n_commits_total != n_commits:
            return out
        cmd = chat_mod.classify_command(rec.get('body', ''))
        kind = cmd[0] if cmd else None
        if kind == 'eyes' and self.phase in ('declare', 'eyes'):
            # the Shadow seat's 'Ne' chat (human Shadow, or the AI's
            # own '0e' announce): closes the declaration window, then
            # allocates through the ONE eye answerer
            self._settle_declare_decline("'Ne' eye chat")
            self.phase = 'eyes'
            try:
                n = int(cmd[1] or 0)
            except (TypeError, ValueError):
                n = 0
            for _ in range(max(0, n)):
                if not self._allocate_one_eye():
                    self.findings.append(('eye_alloc_failed',
                                          'prefix chat %r'
                                          % rec.get('body', '')))
                    break
            return out
        if kind is None or ai.describe_pending(self.es) is not None:
            return out
        if self.phase != 'actions':
            return out
        if kind == 'pass':
            side = (ai._next_acting_side(self.es, self.last_actor)
                    if self.last_actor is not None
                    else gs.Side.FREE_PEOPLES)
            if side is None:
                self.findings.append(('prefix_pass_nobody',
                                      'no side to act'))
                return out
            with self.rand.pin():
                menu = ai.get_legal_actions(self.es, side)
            pa = next((a for a in menu if a.action_type == 'pass'),
                      None)
            if pa is None:
                self.findings.append(('prefix_pass_illegal',
                                      '%s has no pass' % side.name))
                return out
            items = self._commit_human(pa, via='prefix-pass')
            if self.chat_deferred is None:
                self.last_actor = side
            self.prefix_items.extend(items)
            return out
        if kind in ('attack', 'attack_from', 'region'):
            w = self.seg.window
            if w is None or not getattr(w, 'die_face', None):
                if kind == 'region' and not self._is_region_name(cmd[1]):
                    return out
                self.findings.append(('prefix_attack_no_window',
                                      rec.get('body', '')[:60]))
                return out
            side = (gs.Side.SHADOW if w.die_side == 'SHADOW'
                    else gs.Side.FREE_PEOPLES)
            if kind == 'attack':
                spec = cmd[1]
            elif kind == 'attack_from':
                ask = getattr(self, '_attack_ask', None)
                if ask is None:
                    self.findings.append(('prefix_attack_from_noask',
                                          rec.get('body', '')[:60]))
                    return out
                spec = (ask, cmd[1])
            else:
                if not self._is_region_name(cmd[1]):
                    return out
                spec = (cmd[1], None)
            # the live declaration handler, with the ACTING seat as
            # its human (the chat's seat is the staged die's seat)
            saved = self.human_side
            self.human_side = side
            try:
                items = self._human_chat_attack(spec)
            finally:
                self.human_side = saved
            for it in items:
                if it and it[0] == 'chat' and it[1] == 'refusal':
                    self.findings.append(('prefix_attack_refused',
                                          str(it[2])[:120]))
            self.prefix_items.extend(items)
        return out

    # ---- the prefix verification ---------------------------------------
    def _ingest_line(self, raw, sc):
        if self.live:
            return LiveDriver._ingest_line(self, raw, sc)
        if self.desync is not None:
            raise DriverHalt(self.desync)
        k = self.prefix_seen
        # open a reorderable run when we reach its first line
        if self._run_pool is None and k in self._run_end:
            self._run_end_cur = self._run_end[k]
            self._run_pool = [prefix_key(ln) for ln in
                              self.prefix_expected[k:self._run_end_cur]]
        if self._run_pool is not None:
            key = prefix_key(raw)
            if key in self._run_pool:
                self._run_pool.remove(key)
                self.prefix_seen += 1
                LiveDriver._ingest_line(self, raw, sc)
                if not self._run_pool:
                    self._run_pool = None
                    self._run_end_cur = None
                if self.prefix_seen == self.prefix_n:
                    self.flip_due = True
                return []
            if reorderable_line(raw):
                # an extra bookkeeping paragraph the fixture did not
                # record (a duplicate turn marker, an extra no-op)
                self.prefix_skipped.append((self._feed_seq, raw))
                LiveDriver._ingest_line(self, raw, sc)
                return []
            # real content arrived: the rest of the run never rendered
            # this boot (noted, never a desync)
            if self._run_pool:
                self.prefix_skipped.append(
                    (self._feed_seq, 'run not fully rendered: %r'
                     % (self._run_pool,)))
            self.prefix_seen = self._run_end_cur
            self._run_pool = None
            self._run_end_cur = None
            k = self.prefix_seen
        if k >= self.prefix_n:
            if skippable_extra(raw):
                self.prefix_skipped.append((self._feed_seq, raw))
                return []
            self.desync = (
                'SCENARIO DESYNC: feed line %d arrived beyond the %d-line '
                'prefix before the live flip: %r'
                % (self._feed_seq, self.prefix_n, raw[:120]))
            raise DriverHalt(self.desync)
        exp = self.prefix_expected[k]
        if prefix_key(raw) != prefix_key(exp):
            if skippable_extra(raw):
                self.prefix_skipped.append((self._feed_seq, raw))
                return []
            self.desync = ('SCENARIO DESYNC at line %d: expected %r got %r'
                           % (k + 1, exp[:140], raw[:140]))
            raise DriverHalt(self.desync)
        self.prefix_seen = k + 1
        items = LiveDriver._ingest_line(self, raw, sc)
        # lane outputs never reach the client; a refusal or a hold in
        # the prefix means the client did not reproduce the fixture
        for it in items:
            if it and it[0] == 'chat' and it[1] == 'refusal':
                self.findings.append(('prefix_refusal', str(it[2])[:120]))
        self.prefix_items.extend(items)
        if self.prefix_seen == self.prefix_n:
            self.flip_due = True
        return []

    def prefix_open_state(self):
        """What the prefix lane still holds open (must be nothing at
        the cut): the cutter refuses such a cut, the flip halts on it."""
        open_ = []
        if self.seg.window is not None:
            open_.append('open gesture window')
        if self.deferred is not None:
            open_.append('deferred window')
        if self.pending_buf is not None and self.pending_buf.records:
            open_.append('pending buffer (%d records)'
                         % len(self.pending_buf.records))
        if self.chat_deferred is not None:
            open_.append('deferred chat decision')
        if self.holds:
            open_.append('%d matcher hold(s)' % len(self.holds))
        if self.uncommitted:
            open_.append('%d uncommitted line(s)' % len(self.uncommitted))
        return open_

    def prefix_problems(self):
        """Findings/alarms the prefix lane accumulated - the cut is
        proven only when this is empty."""
        out = []
        for k, d in self.findings:
            out.append('finding %s: %s' % (k, str(d)[:160]))
        for a in self.alarms:
            out.append('alarm: %s' % str(a)[:160])
        for o in self.prefix_open_state():
            out.append('open at cut: %s' % o)
        if self.unknown_lines:
            out.append('%d unknown line(s)' % len(self.unknown_lines))
        return out

    # ---- the flip --------------------------------------------------------
    def _become_live(self):
        import game_state as gs
        self.live = True
        self.spectator = False
        self.ai_side = self._ai_side_live
        self.human_side = self._human_side_live
        self.nick = self._nick_live
        self.seg = Segmenter(nick=self._nick_live)
        hm = self.hand_models.get(self.human_side)
        if hm is None:
            from .driver import HandModel
            hm = HandModel(self.es, self.human_side)
        hm.es = self.es
        self.hand_models = {self.human_side: hm}
        self.hands = hm
        self.view.dump_path = self._dump_path
        if self.rand is not None:
            self.rand.reset_feed()

    def _rebuild_ai_decks(self, ai_side, hand):
        """The AI side's two decks after the flip = the card table
        minus the adopted hand minus every cid the prefix already put
        face-up or on the table; no placeholder remains on the AI
        side.  Invariant: the prefix lane drew the AI's cards as
        placeholders, each pop taking a REAL cid out of the fictional
        deck - a live draw of that cid would halt 'deck desync'; the
        AI's identities ride the client's commitment decode, so its
        decks must hold every real cid the client can still draw.
        (The human side keeps its placeholder model: its draws stay
        opaque by design.)"""
        import game_state as gs
        from cards import get_cards_by_deck
        tag = 'fp' if ai_side is gs.Side.FREE_PEOPLES else 'sh'
        gone = set(str(c) for c in hand)
        for pile in (getattr(self.es, 'discard_face_up', None)
                     or {}).values():
            gone.update(str(c) for c in (pile or ()))
        pcb = getattr(self.es, 'persistent_cards', None)
        for cards in (getattr(pcb, 'fp_cards', None) or (),
                      getattr(pcb, 'sh_cards', None) or ()):
            gone.update(str(getattr(c, 'card_id', '')) for c in cards)
        for deck, attr in (('Character', tag + '_char_deck'),
                           ('Strategy', tag + '_strat_deck')):
            table = [c.card_id for c in get_cards_by_deck(ai_side, deck)]
            old = list(getattr(self.es, attr, None) or ())
            keep = [c for c in old if c in table and c not in gone]
            missing = [c for c in table if c not in gone
                       and c not in keep]
            setattr(self.es, attr, keep + missing)
            self.flip_report.append(
                '%s rebuilt: %d real (%d restored, %d placeholder(s) '
                'dropped)' % (attr, len(keep) + len(missing),
                              len(missing),
                              sum(1 for c in old
                                  if str(c).startswith('HU_'))))

    def flip(self, ai_hand_cids, card_pids):
        """Flip to live.  `ai_hand_cids`: the AI hand as the client
        states it (dump pids -> decoded cids, in zone order);
        `card_pids`: cid -> pid for those cards.  Raises DriverHalt on
        any disagreement with the prefix lane's engine state."""
        import game_logic as gl
        if self.live:
            return
        open_ = self.prefix_open_state()
        if open_:
            raise DriverHalt('SCENARIO FLIP refused: the prefix lane '
                             'still holds %s' % '; '.join(open_))
        ai_side = self._ai_side_live
        hand = self.es.hands[ai_side].hand
        if len(hand) != len(ai_hand_cids):
            raise DriverHalt(
                'SCENARIO FLIP refused: the engine holds %d card(s) in '
                'the AI hand after the prefix, the client dump shows %d'
                % (len(hand), len(ai_hand_cids)))
        new_hand = []
        for cid in ai_hand_cids:
            cid = str(cid)
            if cid in new_hand:
                raise DriverHalt('SCENARIO FLIP refused: duplicate AI '
                                 'card %s in the client hand' % cid)
            new_hand.append(cid)
        self.es.hands[ai_side].hand = new_hand
        self.es.hands[ai_side].needs_discard = len(new_hand) > 6
        self._rebuild_ai_decks(ai_side, new_hand)
        for cid, pid in card_pids.items():
            self.view.card_pid_map[str(cid)] = str(pid)
            self.view.note_drawn(str(pid))
        self._become_live()
        self.flip_report = [
            'prefix %d/%d verified (%d extra line(s) skipped)'
            % (self.prefix_seen, self.prefix_n, len(self.prefix_skipped)),
            'AI hand adopted from the client: %d card(s)' % len(new_hand),
            'phase %s turn %s last_actor %s'
            % (self.phase, getattr(self.es, 'turn', None),
               self.last_actor.name if self.last_actor is not None
               else None)] + list(self.flip_report)
        self.findings.append(('scenario_flip', '; '.join(self.flip_report)))


# ----------------------------------------------------------------------
def replay_prefix(feed_path, n, ai_side_name, seed=1234, meta_path=None,
                  expected=None):
    """THE prefix lane, offline: replay the first `n` lines of
    `feed_path` through a ScenarioDriver (spectator lane, no flip).
    Returns the driver (its prefix_problems() / position_summary()
    are the cutter's proof and the server's reference)."""
    import board as board_mod
    import game_logic
    import game_state as gs
    import random as _random
    from . import randomness as rnd_mod
    from .sidecar import SidecarReader
    _random.seed(seed)
    st = gs.setup_initial_game(board_mod.load_board())
    game_logic.initialize_decks(st)
    rand = rnd_mod.ClientRandomness(seed=seed)
    rand.install()
    ai_side = (gs.Side.SHADOW if ai_side_name in ('SA', 'SHADOW')
               else gs.Side.FREE_PEOPLES)
    if expected is None:
        with open(feed_path, 'rb') as f:
            lines = f.read().decode('utf-8', 'replace').split('\n')
        lines = [ln.rstrip('\r') for ln in lines]
        if lines and lines[-1] == '':
            lines.pop()
        expected = lines[:n]
    drv = ScenarioDriver(st, ai_side, feed_path, rand, expected,
                         nick='<scenario>')
    if meta_path and os.path.exists(meta_path):
        drv.sidecar = SidecarReader(meta_path)
    while not drv.flip_due:
        try:
            new = drv.feed.poll()
        except Exception as e:
            drv.findings.append(('feed_error', repr(e)))
            break
        if not new:
            break
        sc = drv.sidecar
        if sc is not None:
            sc.poll()
        base = len(drv.feed.lines) - len(new)
        for i, raw in enumerate(new):
            drv._feed_seq = base + i + 1
            if drv.flip_due:
                break
            try:
                drv._ingest_line(raw, sc)
            except DriverHalt as e:
                drv.findings.append(('halt', str(e)))
                return drv
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:
                drv.findings.append(('route_error', '%r' % (e,)))
                try:
                    drv.rand.reset_feed()
                except Exception:
                    pass
    if drv.prefix_seen < n:
        drv.findings.append(('prefix_short', 'replayed %d of %d lines'
                             % (drv.prefix_seen, n)))
    return drv


def ai_hand_from_dump(dump, ai_side_name, pid_to_cid):
    """(cids in zone order, {cid: pid}, undecoded pids) from the
    client's board dump: the AI-hand zone's pids through the
    commitment decode the raw stream already yielded."""
    want = _HAND_ZONE[ai_side_name]
    pids = sorted(p for p, z in (dump or {}).items()
                  if str(z).strip().lower() == want)
    cids, card_pids, missing = [], {}, []
    for p in pids:
        cid = pid_to_cid.get(str(p))
        if cid is None:
            missing.append(str(p))
        else:
            cids.append(cid)
            card_pids[cid] = str(p)
    return cids, card_pids, missing


def claim_tables_consistent(raw_lines, ai_side_name):
    """(ok, n_claims): every AI-side claim line in the raw stream
    carries the SAME commitment tables (the client adopted the
    fixture's tables); two different payloads = the client claimed
    fresh tables first and the replayed claim was rejected."""
    seen = set()
    for ln in raw_lines:
        p = claim_payload(ln, ai_side_name)
        if p is not None:
            seen.add(p)
    return (len(seen) <= 1, len(seen))
