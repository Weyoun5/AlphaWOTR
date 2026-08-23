"""Client-native undo stack model + refusal press computation (LIFT-B).

LIFTED from wotr_bridge/ai_move_server.py (read-only donor; nothing is
imported from it):
  CLIENT_SNAP_VERBS      <- _CLIENT_SNAP_VERBS          (:3065-3096)
  stack_after            <- _client_stack_after         (:3098-3124)
  presses_to_revert      <- the undo-eligibility / press-count logic
                            inside _sided_refuse        (:3683-3746)
  RestoreWatch           <- _verify_restore_watch       (:10867-10944)
                            + the watch construction in _sided_refuse
                            (:3744-3778, _touched_events semantics)

INPUT IS THE RENDERED FEED (the new driver's only wire; REBUILD_PLAN
architecture).  The donor worked on raw <~Controls.N~> op lines and
classified them through _OP_VERB - a hand-written table that is on the
plan's RETIRE list (defect 9: duplicate vocabulary).  Here every line
is classified by wotr_bridge/live/live_parser.parse, whose grammar is
src/rendered_replay.parse_line - the corpus-proven single answerer.
The rendered second token IS the client's own dispatch axis
(client/decompiled-src/ay_0.java interprets the rendered verb), so no
op-id table exists to go stale.  _VERB_OF below is not a vocabulary:
it annotates parse_line's OWN op names with the client verb each
rendering carries; test_live_witnesses asserts the annotation covers
every op live_parser can emit, so grammar growth breaks a test, never
silently drops a verb.

CLIENT FACTS THAT BIND THIS MODULE (REBUILD_PLAN registry):
  #1  Client undo (Controls.819) is LIFO, whole-board, NO author
      filter; snapshots are pushed on moves / kills / flips /
      allocates AND DRAWS (ay_0.k() from the Interpreter.48/.51/.52/
      .54 branches AND from h(), the .50 draws handler; driven 7/7 on
      the shipped jar by streak/client_undo_probe.py).  An <auto>
      line with a snapshot verb (the '(chosen) to Current .. Card'
      reveal) pushes ITS OWN snapshot (the interpreter accepts
      '<auto>' at ay_0.java:411 and runs k() for 'moves'); ONE press
      (ay_0.o()) pops the newest snapshot and then keeps popping
      while the next-older logged line starts with '<auto>' - so
      <auto> snapshots fold into the press of the entry ABOVE them
      (newer), and an <auto> on TOP of the stack costs a press of
      its own.  MEASURED on the real client (D5, probe_cc_refusal_
      u_green3): one press after '$Wey chooses a card' + the <auto>
      reveal retracted the reveal only - the older staged die stayed
      in Current Action.  'chooses' (l()) pushes nothing.  The donor's _snap9 (:3689)
      deliberately omits 'draws' - KNOWN WRONG, kept there only
      because two old pins trap it (test_adv_refusal_side_effects legs
      2 and 6).  THIS LIFT USES THE CLIENT-TRUE SET: a draw pushes a
      snapshot, so a draw in the window costs a press and a draw above
      the window is a line a press lands on.  (Contradiction on file:
      streak/measure_452_snapshot.py found zero same-nick undo presses
      within six lines of 386 card plays - absence-of-evidence from
      logs; the decompiled dispatch is the stronger source and fact #1
      adopts it.)
  #2  A press reaches a gesture only if nothing was pushed after it;
      the AI's OWN emissions eat presses first.  presses_to_revert
      therefore charges one extra press per own-emission snapshot
      above the window, and returns None the moment a HUMAN (or
      unmodelled) snapshot sits above it - the human's later gestures
      may never be re-forged by us (measured: census_refusal_stack_
      tail seed 8801 - one press with seven lines above landed on the
      AI's own drag and destroyed it).
  #3  '$Nick' is NOT hand/zone attribution.  The ONE legitimate nick
      use here is own-EMISSION detection: lines the server itself
      wrote echo with the server's nick; lines produced by the
      client's own handlers (directive draws/rolls) carry the LOCAL
      seat's nick and are deliberately treated as not-ours.

DEVIATIONS from the donor (each paid for or safety-preserving):
  * <auto> FOLD is modelled (fact #1): an <auto> snapshot line is an
    entry of its own; every later push absorbs the contiguous <auto>
    entries beneath it (Entry.folds), so they ride that entry's press;
    an <auto> run still on top costs one press.  The donor declared
    any <auto> 'outside the proven model'.
  * a 'chooses' line IMMEDIATELY followed by its '<auto> .. Hand
    (chosen) to Current .. Card.' reveal is a PUBLIC, deterministic
    pair (the event play through the client's own chooser): the
    choose costs nothing, the reveal is the snapshot the press lands
    on.  A 'chooses' with no reveal (the hidden combat commit) stays
    ineligible - the driver's own lane (_withdraw_human_cc) handles
    that form.
  * 'undoes' lines POP the modelled stack (LIFO, folds go with their
    entry); an underflow (nothing modelled to pop) is an unmodelled
    entry -> uncountable.  The donor skipped undo lines entirely,
    which over-counted the stack (safe direction, less true).
  * 'redoes' is unmodelled -> uncountable (the donor skipped it; a
    redo RE-pushes state we cannot see, so counting past one would be
    a guess).
  * rendered 'changes' (elven-ring/Mouth die face) and 'adds N eyes'
    are collateral-reverted pass-through lines, exactly the donor's
    'changes' class (:3708-3709).
  * rolls / random-discards (and a reveal-less 'chooses') never push
    a snapshot (not in the five) but ARE client randomness / hidden
    state: inside a span they make the window undo-INELIGIBLE (donor
    :3711), above the window they do not block a press (no snapshot
    was pushed - donor stack semantics unchanged).

Nothing here mutates driver or engine state; RestoreWatch verifies a
dump and reports - the corrective action belongs to the driver.
ASCII only.
"""
import os
import sys
from collections import namedtuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_BRIDGE = os.path.abspath(os.path.join(_HERE, '..'))
_SRC = os.path.abspath(os.path.join(_HERE, '..', '..', 'src'))
for _p in (_SRC, _BRIDGE, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import live_parser

# what the client pushes an undo snapshot for - READ OUT OF THE CLIENT
# (ay_0.java k()/h(); donor :3065-3096).  CLIENT-TRUE: includes 'draws'.
CLIENT_SNAP_VERBS = ('moves', 'kills', 'flips', 'allocates', 'draws')

# collateral-reverted lines: no snapshot of their own; their board
# effect is restored by whichever earlier snapshot a press lands on
_PASS_VERBS = ('changes', 'adds')

# client randomness / hidden-state productions: no snapshot, and a
# window containing one may not be undone by us (re-driving them would
# re-randomize what the client already produced)
_SOFT_VERBS = ('rolls', 'chooses', 'randomly')

# live_parser op -> the client verb its rendering carries.  Annotation
# of the parser's OWN vocabulary, not a second grammar; None = the line
# is stateless for the undo stack.  unclassified_ops() below lets the
# tests prove full coverage of live_parser's emission set.
_VERB_OF = {
    'move': 'moves',
    'kill': 'kills',
    'flip': 'flips',
    'draw': 'draws',
    'alloc_eye': 'allocates',
    'roll_dice': 'rolls',
    'roll_num': 'rolls',
    'choose': 'chooses',
    'random_discard': 'randomly',
    'die_change': 'changes',
    'add_eye': 'adds',
    'undo': 'undoes',
    'redo': 'redoes',
    'turn': None,
    'chat': None,
    'system': None,
    'blank': None,
    'noop': None,          # 'views the .. hand' / 'deselects a ..'
    'other': 'unknown',
}

# every op live_parser documents itself as emitting (live_parser.parse
# docstring + rendered_replay.parse_line returns)
_LIVE_PARSER_OPS = (
    'move', 'kill', 'flip', 'draw', 'roll_dice', 'roll_num', 'add_eye',
    'alloc_eye', 'choose', 'undo', 'redo', 'turn', 'other', 'chat',
    'blank', 'system', 'noop', 'die_change', 'random_discard')


def unclassified_ops():
    """live_parser ops with no _VERB_OF annotation (tests assert [])."""
    return [op for op in _LIVE_PARSER_OPS if op not in _VERB_OF]


Entry = namedtuple('Entry', 'index verb folds modelled')

# the chooser's public reveal: the ONE <auto> grammar a 'chooses' line
# may be paired with (live_parser renders it as a 'move' from the
# '.. Hand (chosen)' zone)
_REVEAL_MARK = 'Hand (chosen) to Current'


def _is_auto(line):
    s = str(line).lstrip()
    return s.startswith(('<auto>', '%auto>', 'auto>'))


def is_reveal_line(line):
    """Is this feed line the chooser's '<auto> moves <title> from
    <side> Hand (chosen) to Current <FP|SA> Card.' reveal?"""
    return _is_auto(line) and _REVEAL_MARK in str(line)


def _push(entries, k, verb):
    """Push one snapshot entry; the contiguous <auto> entries on top
    of the stack fold into it (ay_0.o(): their press is this entry's
    press)."""
    folds = ()
    while entries and entries[-1].modelled \
            and entries[-1].verb == '<auto>':
        e = entries.pop()
        folds = (e.index,) + e.folds + folds
    entries.append(Entry(k, verb, folds, True))


def is_own_line(line, nick):
    """Is this feed line the AI's own emission (or a directive)?
    Donor: _is_own_line (:3126-3133), tightened so a nick can never
    prefix-match a longer nick ('WOTR_AI' vs '<WOTR_AI_FP>')."""
    if not nick:
        return False
    s = str(line).lstrip()
    if s.startswith('!CALL'):
        return True
    if s.startswith('%'):
        s = s[1:]
    if s.startswith('$' + nick):
        rest = s[len('$' + nick):]
        return rest[:1] in ('', ' ')
    return s.startswith('<' + nick + '>')


def classify(line):
    """One feed line -> (verb, rec).  verb is the client dispatch verb
    ('moves', 'kills', ..., 'unknown') or None for stateless lines.
    The grammar is live_parser.parse - the single answerer."""
    rec = live_parser.parse(line)
    op = rec.get('op')
    verb = _VERB_OF.get(op, 'unknown')
    if rec.get('unknown'):
        verb = 'unknown'
    return verb, rec


def stack_after(lines, start=0):
    """[Entry] for every line at or after `start` that the CLIENT would
    have pushed onto its undo stack (donor: _client_stack_after,
    :3098-3124; whoever wrote it - the client's undo has no author
    filter).

    LIFO fold semantics (fact #1): an <auto> snapshot line is a
    modelled entry (verb '<auto>'); every later push absorbs the
    contiguous <auto> entries beneath it (Entry.folds) - the client's
    press pops the entry and then every <auto> directly older than it.
    An 'undoes' line POPS the most recent entry (its folds go with
    it); popping an empty modelled stack is unmodelled.  Any line the
    grammar cannot classify is an unmodelled entry - a stack we cannot
    count is one a press cannot be aimed at."""
    entries = []
    for k in range(max(0, start), len(lines)):
        ln = lines[k]
        verb, _rec = classify(ln)
        if _is_auto(ln):
            if verb in CLIENT_SNAP_VERBS:
                _push(entries, k, '<auto>')
            elif verb is not None and verb not in _PASS_VERBS:
                entries.append(Entry(k, 'unknown:<auto>', (), False))
            continue
        if verb is None:
            continue                       # stateless (chat/system/..)
        if verb == 'undoes':
            if entries and entries[-1].modelled:
                entries.pop()
            else:
                entries.append(Entry(k, 'undoes:underflow', (), False))
            continue
        if verb in CLIENT_SNAP_VERBS:
            _push(entries, k, verb)
        elif verb in _PASS_VERBS:
            # collateral: reverts with the preceding snapshot; folds
            # there when one exists, otherwise it is invisible to the
            # stack (donor parity: 'changes' was never appended)
            if entries and entries[-1].modelled:
                e = entries[-1]
                entries[-1] = e._replace(folds=e.folds + (k,))
        elif verb in _SOFT_VERBS:
            continue                       # no snapshot pushed
        else:
            entries.append(Entry(k, 'unknown:%s' % verb, (), False))
    return entries


def presses_to_revert(lines, window, own_nick=None):
    """How many client UNDO presses revert the human gesture in
    `window` = (w0, w1), or None if a press cannot reach it (the driver
    then takes the corrective path - exhaustion HOLDS, never guesses).

    Donor: the span walk in _sided_refuse (:3683-3746) with the
    CLIENT-TRUE verb set (draws snapshot; fact #1) and the stack-above
    gate of _client_stack_after, extended per fact #2: the AI's OWN
    emission snapshots above the window are chargeable (one press
    each - they pop first and the driver may re-forge its own lines);
    ANY human or unmodelled snapshot above the window returns None."""
    w0, w1 = window
    w0, w1 = max(0, w0), min(len(lines), w1)
    # the UNDO SPAN: trailing run of non-own lines in the window.
    # Presses pop from the tail, so older un-ingested lines below w0
    # are never reached (donor :3672-3688).
    span = []
    j = w1 - 1
    while j >= w0 and not is_own_line(lines[j], own_nick):
        span.append(j)
        j -= 1
    span.reverse()
    if not span:
        return None
    # ---- THE SPAN'S OWN SNAPSHOTS ----
    entries = []
    first_state_verb = None
    for n_i, i in enumerate(span):
        ln = lines[i]
        verb, _rec = classify(ln)
        if _is_auto(ln):
            if verb in CLIENT_SNAP_VERBS:
                _push(entries, i, '<auto>')
                if first_state_verb is None:
                    first_state_verb = '<auto>'
                continue
            if verb is None or verb in _PASS_VERBS:
                continue
            return None               # an <auto> the grammar cannot place
        if verb is None:
            continue                  # chat / system: stateless
        if verb == 'chooses':
            # the chooser's PUBLIC pair: the very next stateful line
            # must be its reveal (the snapshot the press lands on); a
            # reveal-less choose is a hidden commit - ineligible
            nxt = None
            for i2 in span[n_i + 1:]:
                v2, _r2 = classify(lines[i2])
                if v2 is None:
                    continue
                nxt = i2
                break
            if nxt is None or not is_reveal_line(lines[nxt]):
                return None
            continue                  # pushes nothing (ay_0.l())
        if first_state_verb is None:
            first_state_verb = verb
        if verb in CLIENT_SNAP_VERBS:
            _push(entries, i, verb)
        elif verb in _PASS_VERBS:
            pass                      # collateral-reverted (donor :3708)
        else:
            return None               # rolls/unknown/undoes/...
    if first_state_verb not in CLIENT_SNAP_VERBS + ('<auto>',):
        return None               # leading residue (donor :3713)
    if not entries:
        return None
    # ---- THE STACK IS NOT THE HUMAN'S (donor :3715-3743) ----
    # the lines above the window continue the SAME stack: an own
    # snapshot above eats a press first (and absorbs the window's
    # trailing <auto> run into its press); a human or unmodelled
    # entry above is unreachable.
    n_span = len(entries)
    for k in range(w1, len(lines)):
        ln = lines[k]
        verb, _rec = classify(ln)
        if _is_auto(ln):
            if verb in CLIENT_SNAP_VERBS:
                _push(entries, k, '<auto>')
            elif verb is not None and verb not in _PASS_VERBS:
                return None
            continue
        if verb is None:
            continue
        if verb == 'undoes':
            if entries and entries[-1].modelled \
                    and len(entries) > n_span:
                entries.pop()
                continue
            return None               # a press beneath the window
        if verb in CLIENT_SNAP_VERBS:
            _push(entries, k, verb)
        elif verb in _PASS_VERBS:
            pass
        elif verb in _SOFT_VERBS:
            continue                  # no snapshot pushed
        else:
            return None               # uncountable stack above the window
    # ---- PRESS COUNT: one press per entry still standing; a HUMAN
    # snapshot still standing above the window is never re-forged
    for e in entries:
        if not e.modelled:
            return None
        if e.index >= w1 and e.verb != '<auto>'                 and not is_own_line(lines[e.index], own_nick):
            return None               # his later gesture: never re-forged
    return len(entries)


class RestoreWatch(object):
    """REFUSAL-RESTORE VERIFICATION: after pressing undo, the CLIENT'S
    OWN dump must show the pieces back home; if it never does, alarm
    loudly and point at RESYNC instead of letting later gestures
    falsely refuse from a drifted region.

    Donor: _verify_restore_watch (:10867-10944) + the expect/notat
    construction in _sided_refuse (:3744-3778).  The donor's inline
    watch had NEVER EXECUTED OUTSIDE LIVE PLAY (62 records built, zero
    verifications, zero alarms across a 6-seed soak - finding C4 /
    NEXT-1); this class is a plain object whose alarm is PROVEN able
    to fire in src/test_live_witnesses.py.

    Two expectation kinds (donor :10905-10926):
      expect {pid: zone}  the piece must COME BACK to its pre-gesture
                          origin (frm known from the gesture line);
      notat  {pid: zone}  a destination-only gesture (die KILL, swap,
                          send-to-box) cannot say where the piece came
                          FROM - but a restore that landed means it is
                          no longer THERE (the die-105 wedge check).

    verify(dump) returns None (not armed / no dump), [] (verified -
    watch disarms), or the mismatch rows (ticks decrement; on
    exhaustion the STICKY .alarm is set and the watch disarms).  The
    caller supplies dumps (client_board.read_board); this class never
    reads files and never mutates game state - it notices."""

    def __init__(self, expect=None, notat=None, ticks=6):
        self._watch = None
        self.alarm = None
        if expect or notat:
            self.arm(expect, notat, ticks)

    @property
    def armed(self):
        return self._watch is not None

    def arm(self, expect=None, notat=None, ticks=6):
        """Arm (or re-arm) the watch.  Zones are decoded names, as
        read_board returns them."""
        self._watch = {'expect': {str(k): str(v)
                                  for k, v in (expect or {}).items()},
                       'notat': {str(k): str(v)
                                 for k, v in (notat or {}).items()},
                       'left': int(ticks)}

    def clear_alarm(self):
        """RESYNC acknowledged: drop the sticky alarm."""
        self.alarm = None

    def verify(self, dump):
        if self._watch is None:
            return None
        if not dump:
            return None
        from board import sanitize

        def _norm(v):
            x = sanitize(str(v))
            return x[:-11] if x.endswith(' stronghold') else x
        bad = []
        for pid, want in self._watch['expect'].items():
            got = dump.get(str(pid))
            if got is None:
                continue
            g, e = _norm(got), _norm(want)
            if g != e:
                bad.append('%s@%s (expected %s)' % (pid, g, e))
        for pid, notz in self._watch['notat'].items():
            got = dump.get(str(pid))
            if got is None:
                continue
            g, e = _norm(got), _norm(notz)
            if g == e:
                bad.append('%s STILL@%s (the refused gesture put it '
                           'there and the restore did not take it '
                           'back)' % (pid, g))
        if not bad:
            self._watch = None
            self.alarm = None
            print('    [restore-watch] refusal restore verified on '
                  'the client board')
            return []
        self._watch['left'] -= 1
        if self._watch['left'] <= 0:
            self._watch = None
            self.alarm = ('restore did not land (%s) - press RESYNC '
                          'CHECK' % '; '.join(bad[:2]))
            print('!!! [restore-watch] RESTORE DID NOT LAND: %s - the '
                  'client board drifted after a refusal; press RESYNC '
                  'CHECK' % '; '.join(bad[:3]))
        return bad


def resolve_named(name):
    """Rendered figure name -> client pid, for NAMED figures only
    (companions/minions; A17: PIECE_NAME_ALIASES covers the two
    replacement figures, so 'Aragorn' and 'Strider' both resolve to
    241).  Fungible unit names return None - resolving those is the
    fungible tracker's job (design rule 10), never a guess here."""
    import log_replay as lr
    key = str(name).strip().lower()
    for pid in list(lr.COMPANIONS) + list(lr.MINION_PIECES):
        for nm in lr.names_for_pid(pid) or ():
            if str(nm).lower() == key:
                return pid
    return None


def expectations_from_window(lines, resolve=None, pids=None):
    """Build (expect, notat, unverifiable) for a RestoreWatch from a
    rendered gesture window.

    Semantics lifted from the donor's _touched_events consumption
    (:3771-3778): a gesture whose line names an ORIGIN puts the piece
    back there (expect); a destination-only gesture (kill) can only
    say where the piece must NOT still be (notat).  <auto> trailers are
    reverted with their gesture, so their moves count too.

    Rendered lines name pieces, not pids.  `pids` (optional, aligned
    with `lines`: the sidecar's exact piece ids per record, or None)
    is the FIRST answerer - the line's own pieces in render order;
    otherwise `resolve` (default resolve_named) maps a name to a pid
    (named figures only).  Unresolvable pieces are returned in
    `unverifiable` - REPORTED, never silently dropped (the silent
    version of this gap is exactly finding C4)."""
    import rendered_replay as rr
    resolve = resolve or resolve_named
    expect, notat, unver = {}, {}, []
    for j, ln in enumerate(lines):
        verb, rec = classify(ln)
        op = rec.get('op')
        line_pids = None
        if pids is not None and j < len(pids) and pids[j]:
            line_pids = [str(x) for x in pids[j]]
        if op == 'move':
            obj = str(rec.get('obj', ''))
            if obj.startswith('group (') and obj.endswith(')'):
                pieces = rr.split_group(obj[len('group ('):-1])
            else:
                pieces = [obj]
            exact = (line_pids if line_pids is not None
                     and len(line_pids) == len(pieces) else None)
            for k, nm in enumerate(pieces):
                pid = exact[k] if exact is not None else resolve(nm)
                if pid is None:
                    unver.append((nm, rec.get('from')))
                else:
                    expect.setdefault(str(pid), str(rec.get('from')))
        elif op == 'kill':
            pieces = list(rec.get('pieces', []) or [])
            exact = (line_pids if line_pids is not None
                     and len(line_pids) == len(pieces) else None)
            for k, nm in enumerate(pieces):
                pid = exact[k] if exact is not None else resolve(nm)
                if pid is None:
                    unver.append((nm, rec.get('zone')))
                else:
                    notat.setdefault(str(pid), str(rec.get('zone')))
    if unver:
        print('    [restore-watch] %d window piece(s) unverifiable by '
              'name (fungible or unknown): %s'
              % (len(unver), '; '.join(str(u[0]) for u in unver[:4])))
    return expect, notat, unver
