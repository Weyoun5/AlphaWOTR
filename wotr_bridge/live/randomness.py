"""Client-randomness funnel protocol for the THIN sided driver.

LIFTED from wotr_bridge/ai_move_server.py (the monolith stays
untouched; nothing here imports from it), decoupled from AiBrain into
ONE class, ClientRandomness, that owns all of its state explicitly.

PROVENANCE MAP (old monolith line refs -> here):
  NeedClientCall                 :64-90      -> NeedClientCall (verbatim
                                                contract, incl. the
                                                BaseException rationale)
  _install_live_patches          :11539-11772-> ClientRandomness.install
                                                (one install per process,
                                                CURRENT-instance dispatch
                                                via module _ACTIVE - the
                                                synth-probe14 lesson)
  _live_advance                  :11774-11800-> ClientRandomness.advance
  _feed_d6                       :11803-11813-> _consume_d6
  _feed_tile                     :11815-11894-> _consume_tile (parser-
                                                authority heals B9/B9(c)
                                                DROPPED - design rule 1:
                                                the engine is the one
                                                state authority, so the
                                                pool either holds the fed
                                                face or it is a real
                                                desync and halts)
  _DWARVEN_REROLL_CAP+_feed_dwarven_face
                                 :11896-11956-> _consume_dwarven_face
  _is_companion_seq/_is_wwsat_seq:11958-11989-> _is_companion_seq /
                                                _is_wwsat_seq (tripwire
                                                CLASSIFIERS only, never
                                                answerers - the companion
                                                answer moved to the named
                                                funnel patch below)
  _feed_companion                :12009-12034-> _consume_companion (now
                                                reached through the NAMED
                                                engine funnel
                                                game_logic.pick_random_
                                                companion, landed Phase 1
                                                2026-08-19, instead of
                                                random.choice sniffing)
  _feed_card                     :12036-12079-> _consume_card
  _feed_missing_cards            :12082-12094-> _missing_cards
  _feed_commit                   :12096-12113-> step_commit
  pick (journal)                 :976-1002   -> journal_pick (+ act_key
                                                :970-974)
  _legal RNG pin                 :12180-12200-> pin
  _attempt_begin                 :12202-12251-> begin_attempt
  _directive_response (builders) :12378-12540-> directives_for (pure
                                                builder: returns
                                                Directive tuples; chat
                                                whitelisting/rendering,
                                                the dry-pool self-heal,
                                                the balrog announce and
                                                the scan floors/cursors
                                                are ROUTER/EMISSION
                                                concerns per the rebuild
                                                layout)
  unconsumed-randomness honesty  :6934-6942  -> unconsumed (the driver's
                                                step-end gate reads the
                                                product's own counters)
  quarantine feed reset          :6899-6903  -> reset_feed

NOT lifted here (different homes per REBUILD_PLAN):
  * _ingest_pending's LOG-LINE harvest (regexes over Controls.* lines,
    scan floors, consumed-cursors, roster-lag deferral): the router
    parses lines and calls feed() exactly once per client outcome; the
    action-roll roster verdict needs the engine dice roster and
    adopt_rolled_faces, both engine/router-side.
  * the WWSAT-ordering emission stage hook (:11652-11697): emission
    ordering, belongs to emission.py.
  * kind 'action_roll' feeding: faces are adopted engine-side through
    game_logic.adopt_rolled_faces (one answerer per quantity); this
    module only BUILDS the ask (directives_for supports the kind).

WWSAT STATUS (measured 2026-08-19): game_logic._wwsat_resolve still
makes its random FP-hand pick through raw random.choice
(game_logic.py:2570) - the Phase-1 named funnel for it has NOT landed.
Until it does, a live WWSAT hand discard on the real trajectory HALTS
LOUDLY at the random.choice tripwire with a message naming exactly
that (fail loud, never wait silently).  The old _feed_wwsat answerer
is deliberately NOT carried over.

MEASURED CLIENT FACTS preserved (verify against REBUILD_PLAN registry):
  fact 19  d6 excess drops from the RIGHT within one roll line;
           cross-line surplus feeds the step's next consumer  (feed)
  G1       each side rolls its own dice - a human-side d6 ask is a CHAT
           ask, an AI-side one is '!CALL combat_roll'  (directives_for)
  G11      the SHADOW pool rolls first  (directives_for 'action_roll')
  scoping  real-trajectory only: funnels divert at _adv_depth > 0
           (inside advance()); MCTS clones run at depth 0 and keep the
           engine's own RNG  (install/advance)

ASCII only (cp1252 console).
"""
import contextlib
import os
import random
import sys
from collections import namedtuple

# wotr_bridge/live/ -> src and wotr_bridge on the path (house style of
# live_parser.py; no package-relative imports, the parent dir has no
# __init__.py)
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, '..', '..', 'src'))
_BRIDGE = os.path.abspath(os.path.join(_HERE, '..'))
for _p in (_SRC, _BRIDGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class NeedClientCall(BaseException):
    """The engine reached a point that needs RANDOMNESS, and the user
    mandate is that all randomness be executed by the CLIENT's own
    built-in handlers (its Math.random, its commitment machinery) -
    never fabricated server-side.  Raised out of a real game advance
    (never out of MCTS simulation clones), caught by the driver, and
    answered with `!CALL <directive>` lines that AiOpp executes through
    the real client handlers (wotr_bridge/PROTOCOL.md, Directive
    protocol).

    Derives from BaseException ON PURPOSE: the rules engine is full of
    broad `except Exception` guards (event_effects.play_event_card
    swallowed this mid-card-effect and left the card half-applied AND
    still in hand - the t116 fake-client run then watched the engine
    play Candles of Corpses twice).  Control flow must never be eaten
    by an effect guard.  kinds:
      'd6'          kw: need=<int more d6 faces>, ctx, side
                    (side = WHOSE dice, G1: each side rolls its own)
      'tile'        one hunt-tile draw            (client hunt)
      'draws'       kw: items=[(side_name, deck), ...] (client fdrawc..)
      'action_roll' kw: sides={side_name: (zone_code, [die pids])}
                    (router-raised; faces land via adopt_rolled_faces)
      'randomcomp'  one random-companion draw     (client randomcomp,
                    A.A(), Controls.637; reached through the named
                    engine funnel game_logic.pick_random_companion)
      'card_pick'   one random card pick (WWSAT hand discard: the
                    client's own random discard, identity never
                    rendered; reached through the named engine funnel
                    game_logic.pick_random_card)

    (Provenance: ai_move_server.py:64-90, verbatim contract.)
    """

    def __init__(self, kind, **kw):
        self.kind = kind
        self.kw = kw
        super().__init__('%s %r' % (kind, kw))


# A directive answering a NeedClientCall.  kind:
#   'call'  text is the full '!CALL ...' line for AiOpp
#   'chat'  text is the nick-less message BODY; the router renders it
#           through the single whitelisted chat exit (design rule 7)
Directive = namedtuple('Directive', ('kind', 'text'))

_SIDE_NAMES = ('FREE_PEOPLES', 'SHADOW')
_DECK_KEYS = ('character', 'strategy')
_CARD_ATTR = {('FREE_PEOPLES', 'character'): 'fp_char_deck',
              ('FREE_PEOPLES', 'strategy'): 'fp_strat_deck',
              ('SHADOW', 'character'): 'sh_char_deck',
              ('SHADOW', 'strategy'): 'sh_strat_deck'}


def _side_name(side):
    """'FREE_PEOPLES'/'SHADOW' from a Side enum or a name string;
    None passes through."""
    if side is None:
        return None
    return getattr(side, 'name', str(side))


def tile_face(tile):
    """Normalized tile identity: special_name FIRST (HuntTile repr
    calls 'The Ring is Mine!' an Eye), accents/punctuation folded so
    engine names meet the corpus piece table.
    (Provenance: ai_move_server.py:13267-13281, byte-equal logic.)"""
    import re as _re
    import unicodedata as _ud
    s = (getattr(tile, 'special_name', None) or
         getattr(tile, 'label', None) or
         getattr(tile, 'name', None) or
         getattr(tile, 'code', None) or tile)
    s = _ud.normalize('NFKD', str(s))
    s = s.encode('ascii', 'ignore').decode('ascii')
    return _re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip()


def act_key(a):
    """Journal identity of an engine Action.
    (Provenance: ai_move_server.py:970-974.)"""
    return (a.action_type, a.die_idx,
            tuple(sorted((str(k), str(v)) for k, v in
                         (a.params or {}).items())))


def _is_companion_seq(seq):
    """True for a non-empty sequence whose every element is a
    Companion.  TRIPWIRE CLASSIFIER ONLY (never an answerer): names the
    bypassed funnel in the halt message.
    (Provenance: ai_move_server.py:11958-11972.)"""
    try:
        if not seq:
            return False
        from game_state import Companion
        for x in seq:
            if not isinstance(x, Companion):
                return False
        return True
    except Exception:
        return False


def _is_wwsat_seq(seq):
    """True for a non-empty sequence of FP CHARACTER card-id strings -
    the Worn-with-Sorrow-and-Toil random FP-hand discard population.
    TRIPWIRE CLASSIFIER ONLY.
    (Provenance: ai_move_server.py:11974-11989.)"""
    try:
        if not seq:
            return False
        for x in seq:
            if not (isinstance(x, str) and 'Character' in x
                    and x.startswith('FREE_PEOPLES')):
                return False
        return True
    except Exception:
        return False


# ---- one install per process, current-instance dispatch -------------
# The patch targets are MODULE globals; a per-instance guard with
# closures frozen on one instance stacked funnel layers whose inner
# layer consulted a STALE instance's depth (always 0) - the real
# trajectory fell through to raw random.randint and the tripwire
# halted the game (synth probe14, 2026-08-16).  The funnels install
# once per process and every closure consults the instance that is
# CURRENTLY deciding via _ACTIVE.
_INSTALLED = False
_ACTIVE = None


def _active_live():
    """The ClientRandomness instance currently on a REAL advance, or
    None (MCTS clones / no live instance -> engine RNG)."""
    a = _ACTIVE
    if a is not None and a._adv_depth > 0:
        return a
    return None


class ClientRandomness(object):
    """Owner of the client-randomness funnel protocol: the fed-outcome
    queues, the starving NeedClientCall, the deterministic-retry
    machinery (pick journal + per-step RNG streams + probe pins), and
    the !CALL directive builders.

    Protocol (one engine step, possibly retried across ticks):
      install()                    once per process, before play
      begin_attempt()              before EVERY attempt of a step
      with advance(): <engine step>   funnels consume the feed or
                                      raise NeedClientCall (pending())
      directives_for(pending())    -> !CALL / chat-ask Directives
      feed(kind, payload)          once per harvested client outcome
      <retry begin_attempt()/advance(): journal + per-step RNG replay
       the previous attempt exactly>
      unconsumed()                 step-end honesty gate reads it
      step_commit()                drop consumed prefixes, close step

    `human_side` (Side or name string, optional): whose d6 asks become
    chat asks instead of '!CALL combat_roll' (G1: each side rolls its
    own dice).
    """

    # cap on how many client d6 values one dwarven reroll may eat.  A
    # real die always has SOME face outside the original's class (the
    # engine checks `valid` is non-empty first), so the loop terminates
    # with probability 1 - the cap exists so a face-table bug becomes a
    # loud halt instead of an infinite drain on the client feed.
    # (Provenance: ai_move_server.py:11896-11901.)
    DWARVEN_REROLL_CAP = 24

    def __init__(self, human_side=None, seed=None):
        self.human_side = _side_name(human_side)
        self.installed = False
        # ENGINE-GAP SHIM (spectator expiry retry ONLY, 2026-08-19):
        # the p.40 hunt re-roll is the Shadow player's OPTIONAL choice,
        # but game_logic._hunt_roll (FROZEN) takes it unconditionally —
        # a client that declined (no re-roll line exists) starves the
        # funnel forever.  Armed, a STARVED 'hunt reroll' consume
        # returns natural 1s: a 1 never succeeds, so the outcome is
        # byte-equivalent to declining.  Every use lands in shim_uses
        # so the census keeps reporting the engine debt.
        self.reroll_decline_shim = False
        self.shim_uses = []
        # last tripwire/desync halt message (STICKY: survives resets,
        # so a quarantined halt stays visible; instrument rule - a
        # component that cannot proceed fails loudly, never silently)
        self.sticky_halt = None
        # every card id that ever entered a live hand through the
        # funnel - verify.py's unfunneled-arrival witness reads this
        self.cid_provenance = set()
        # client-outcome queues + per-attempt consume cursors.
        # d6 entries are (side_tag, value) pairs consumed by a SET of
        # indices (side-lane selection is not prefix-shaped); the
        # other queues keep prefix cursors.
        self._feed = {'d6': [], 'tiles': [], 'cards': {},
                      'companion': [], 'card_pick': []}
        self._feed_pos = {'d6': set(), 'tiles': 0, 'cards': {},
                          'companion': 0, 'card_pick': 0}
        # the NeedClientCall currently starving the engine
        self._pending = None
        self._pending_got = 0          # d6 faces accepted toward it
        # real-trajectory depth (advance()); MCTS clones stay at 0
        self._adv_depth = 0
        # deterministic-retry machinery
        self._rng = random.Random(seed)
        self._step_open = False
        self._step_seed = 0
        self._step_rng_base = None
        self._step_outer_rng = None
        self._probe_ix = 0
        self._journal = []
        self._journal_ix = 0
        # standalone-advance support: a defined residual stream even
        # before the first begin_attempt
        self._step_rng_cur = random.Random(
            self._rng.getrandbits(64)).getstate()
        self._rng_outer = None

    # ---- loud failure ------------------------------------------------
    def _halt(self, msg):
        full = 'LIVE HALT: %s' % msg
        self.sticky_halt = full
        raise RuntimeError(full)

    # ---- install (module patches, once per process) --------------------
    def install(self):
        """Patch the engine's randomness funnels ONCE per process; this
        instance becomes the CURRENT randomness owner (module _ACTIVE).
        The patches consult the active instance's advance depth so MCTS
        simulations (depth 0) keep the engine's own RNG - determinized
        search is allowed to imagine outcomes; only the REAL trajectory
        must come from the client.
        (Provenance: ai_move_server.py:11539-11772.)"""
        global _INSTALLED, _ACTIVE
        _ACTIVE = self
        self.installed = True
        import emitter
        # the emitter planners that can SOURCE randomness (plan_roll,
        # plan_draw_card) must halt rather than invent - same fact as
        # the funnels: from this point the client owns randomness
        emitter.LIVE_MODE = True
        if _INSTALLED:
            return
        _INSTALLED = True
        import game_logic as gl
        import combat as cb

        orig_gl_batch = gl.roll_d6_batch
        orig_cb_batch = cb.roll_d6_batch
        orig_tile = gl.pick_hunt_tile
        orig_pop = gl.pop_hunt_tile
        orig_draw = gl.draw_event_card
        orig_dwarf = gl.use_dwarven_ring
        orig_pick_comp = gl.pick_random_companion
        orig_pick_card = gl.pick_random_card

        def _gl_batch(n, ctx='', side=None):
            a = _active_live()
            if a is not None:
                return a._consume_d6(n, ctx, side=side)
            return orig_gl_batch(n, ctx)

        def _cb_batch(n, ctx='', side=None):
            a = _active_live()
            if a is not None:
                return a._consume_d6(n, ctx, side=side)
            return orig_cb_batch(n, ctx)

        def _tile(state):
            a = _active_live()
            if a is not None:
                return a._consume_tile(state)
            return orig_tile(state)

        def _pop(state):
            a = _active_live()
            if a is not None:
                # pop semantics: the drawn tile leaves the pool here
                t = a._consume_tile(state)
                state.hunt_pool.remove(t)
                return t
            return orig_pop(state)

        def _draw(state, side, deck):
            a = _active_live()
            if a is not None:
                return a._consume_card(state, side, deck)
            return orig_draw(state, side, deck)

        def _dwarf(state, side, *, die_idx, forced_face=None, **kw):
            # DWARVEN RING (USER randomness doctrine 2026-08-16): the
            # reroll face comes from the CLIENT's own d6 roller, never
            # from the engine's random.choice.  Only the real
            # trajectory is diverted; MCTS clones at depth 0 keep the
            # engine RNG, which is what search should use.
            a = _active_live()
            if a is not None and forced_face is None:
                forced_face = a._consume_dwarven_face(
                    state, side, die_idx)
            return orig_dwarf(state, side, die_idx=die_idx,
                              forced_face=forced_face, **kw)

        def _pick_comp(candidates, ctx=''):
            # the NAMED companion funnel (Phase 1, 2026-08-19):
            # replaces the old random.choice sequence-type sniffing -
            # ONE patched site, the engine names the draw itself
            a = _active_live()
            if a is not None:
                return a._consume_companion(candidates, ctx)
            return orig_pick_comp(candidates, ctx)

        def _pick_card(candidates, ctx=''):
            # the NAMED card funnel (Phase 1 completion, 2026-08-19):
            # WWSAT's random FP-hand Character discard and any future
            # random card pick - the client performs the discard and
            # its rendered line carries no identity (blinding holds)
            a = _active_live()
            if a is not None:
                return a._consume_card_pick(candidates, ctx)
            return orig_pick_card(candidates, ctx)

        gl.roll_d6_batch = _gl_batch
        cb.roll_d6_batch = _cb_batch
        gl.pick_hunt_tile = _tile
        gl.pop_hunt_tile = _pop
        gl.draw_event_card = _draw
        gl.use_dwarven_ring = _dwarf
        gl.pick_random_companion = _pick_comp
        gl.pick_random_card = _pick_card

        # ---- TRIPWIRES (kept, per the instrument rule: anything
        # unfunneled on the real trajectory must HALT LOUDLY with a
        # named sticky message - never fall through to engine RNG,
        # never wait silently).  The classifiers name the defect class
        # so the halt is actionable; they never ANSWER anything.
        orig_choice = random.choice
        orig_randint = random.randint

        def _choice(seq):
            a = _active_live()
            if a is not None:
                if _is_companion_seq(seq):
                    a._halt(
                        'unfunneled random.choice over %d Companion '
                        'object(s) on the REAL trajectory - a '
                        'companion pick BYPASSED the named funnel '
                        'game_logic.pick_random_companion (route the '
                        'site through it).  Sample: %r'
                        % (len(seq), [str(x)[:40]
                                      for x in list(seq)[:4]]))
                if _is_wwsat_seq(seq):
                    a._halt(
                        'unfunneled random.choice over %d FP-Character '
                        'card id(s) on the REAL trajectory - a random '
                        'card pick (WWSAT class) BYPASSED the named '
                        'funnel game_logic.pick_random_card (landed '
                        '2026-08-19; route the site through it).  '
                        'Sample: %r'
                        % (len(seq), [str(x)[:40]
                                      for x in list(seq)[:4]]))
                # USER 2026-08-16: "make sure all randomness comes
                # from the client ... every native invocation must
                # fail LOUDLY."  Real trajectory: a random.choice
                # arriving here is the engine inventing an outcome the
                # client never saw.  Falling through to orig_choice
                # would make an UNFUNNELED surface indistinguishable
                # from a funneled one.  MCTS is unaffected: clones run
                # at depth 0 and keep the engine's own RNG.
                a._halt(
                    'unfunneled random.choice on the REAL trajectory '
                    'over %d item(s) of type %r - engine-generated '
                    'randomness the client never produced.  Route '
                    'this pick through a named game_logic funnel or a '
                    '!CALL directive.  Sample: %r'
                    % (len(seq),
                       type(seq[0]).__name__ if seq else None,
                       [str(x)[:40] for x in list(seq)[:4]]))
            return orig_choice(seq)

        def _randint(x, y):
            # C4 hardening: every outcome-bearing d6 goes through
            # roll_d6_batch, which the funnel above REPLACES on the
            # real trajectory - so any randint arriving here IS a
            # leak from a path nobody has funneled.
            # random.shuffle is deliberately NOT covered: pool-order
            # shuffles are outcome-irrelevant (draws are fed by card
            # and tile IDENTITY, never by deck position).
            a = _active_live()
            if a is not None:
                a._halt(
                    'unfunneled random.randint(%r, %r) on the REAL '
                    'trajectory - engine-generated randomness the '
                    'client never produced.  Route it through a '
                    'client feed or a !CALL directive.' % (x, y))
            return orig_randint(x, y)

        random.choice = _choice
        random.randint = _randint
        print('    [live] client-randomness funnels patched '
              '(d6 batches / hunt tiles / card draws / dwarven '
              'faces / named companion picks + choice/randint '
              'tripwires)')

    # ---- real-advance scope + per-step residual RNG stream -----------
    def advance(self):
        """Context manager marking a REAL game advance.  Inside:
        patched funnels consume the client feed; residual engine
        randomness (non-funneled random.* survivors like
        random.random) runs on a dedicated per-step RNG stream that
        resets on every retry of the step, so retries are
        deterministic regardless of how much global RNG the
        interleaved MCTS consumed.
        (Provenance: ai_move_server.py:11774-11800.)"""
        @contextlib.contextmanager
        def _cm():
            self._adv_depth += 1
            if self._adv_depth == 1:
                self._rng_outer = random.getstate()
                random.setstate(self._step_rng_cur)
            try:
                yield
            finally:
                if self._adv_depth == 1:
                    self._step_rng_cur = random.getstate()
                    random.setstate(self._rng_outer)
                self._adv_depth -= 1
        return _cm()

    # ---- attempt / step lifecycle -------------------------------------
    def begin_attempt(self):
        """Start an attempt of the current engine step (opening the
        step if none is open).  Every retry of one step replays the
        previous attempt exactly: fixed residual-RNG base, pinned
        outer stream (legality probes run OUTSIDE advance() and
        _event_card_has_effect dry-runs are RNG-dependent - t121 g1 T8
        journal-mismatch), journal cursor and feed cursors reset.
        (Provenance: ai_move_server.py:12202-12251.)"""
        if not self._step_open:
            tmp = random.Random(self._rng.getrandbits(64))
            self._step_rng_base = tmp.getstate()
            self._step_outer_rng = random.getstate()
            self._step_seed = self._rng.getrandbits(64)
            self._step_open = True
            self._journal = []
        self._step_rng_cur = self._step_rng_base
        random.setstate(self._step_outer_rng)
        self._probe_ix = 0
        self._journal_ix = 0
        self._feed_pos = {'d6': set(), 'tiles': 0, 'cards': {},
                          'companion': 0, 'card_pick': 0}

    def step_commit(self):
        """Engine step completed: drop the consumed feed prefixes and
        close the step (journal cleared, pending cleared).  The driver
        runs its unconsumed() honesty gate BEFORE this.
        (Provenance: ai_move_server.py:12096-12113 + :6965-6967.  The
        old wwsat row is gone WITH its answerer; its lesson - commit
        every queue you feed - is why this method and feed() share one
        kind list.)"""
        f, p = self._feed, self._feed_pos
        f['d6'] = [e for i, e in enumerate(f['d6'])
                   if i not in p['d6']]
        f['tiles'] = f['tiles'][p['tiles']:]
        f['companion'] = f['companion'][p['companion']:]
        f['card_pick'] = f['card_pick'][p['card_pick']:]
        for key in list(f['cards']):
            f['cards'][key] = f['cards'][key][p['cards'].get(key, 0):]
        self._feed_pos = {'d6': set(), 'tiles': 0, 'cards': {},
                          'companion': 0, 'card_pick': 0}
        self._step_open = False
        self._journal = []
        self._journal_ix = 0
        self._pending = None
        self._pending_got = 0

    @property
    def step_open(self):
        """True while an engine step is open (begin_attempt without a
        step_commit/abort_step)."""
        return bool(self._step_open)

    def abort_step(self, reason=''):
        """Close an OPEN step WITHOUT committing (K3): the step was
        abandoned (panel skip/reset, a whose-turn override, the AI's
        due decision vanishing).  Invariants: the journal is discarded
        so the next step's chooser RUNS (a kept journal replays a
        stale pick against a different menu - silently when same-
        keyed, JOURNAL MISMATCH otherwise); the pending ask and every
        consume cursor reset; fed outcomes STAY banked - an outcome the
        client produced is reported by the next step's honesty gate,
        never dropped here.  Idempotent on a closed step."""
        self._feed_pos = {'d6': set(), 'tiles': 0, 'cards': {},
                          'companion': 0, 'card_pick': 0}
        self._step_open = False
        self._journal = []
        self._journal_ix = 0
        self._probe_ix = 0
        self._pending = None
        self._pending_got = 0
        self.last_abort = str(reason)

    def unconsumed(self):
        """Client-produced outcomes not consumed by the step so far -
        the step-end honesty gate's quantity (randomness still
        unconsumed when the step completes is excess the rules did not
        call for).  (Provenance: ai_move_server.py:6934-6942.)"""
        f, p = self._feed, self._feed_pos
        return {
            'd6': len(f['d6']) - len(p['d6']),
            'tiles': len(f['tiles']) - p['tiles'],
            'cards': sum(len(v) - p['cards'].get(k, 0)
                         for k, v in f['cards'].items()),
            'companion': len(f['companion']) - p['companion'],
            'card_pick': len(f['card_pick']) - p['card_pick'],
        }

    def reset_feed(self):
        """Quarantine reset: drop every fed outcome and the pending
        expectation.  sticky_halt is NOT cleared (sticky by design).
        (Provenance: ai_move_server.py:6899-6903.)"""
        self._feed = {'d6': [], 'tiles': [], 'cards': {},
                      'companion': [], 'card_pick': []}
        self._feed_pos = {'d6': set(), 'tiles': 0, 'cards': {},
                          'companion': 0, 'card_pick': 0}
        self._pending = None
        self._pending_got = 0

    # ---- pending + feeding ---------------------------------------------
    def pending(self):
        """The NeedClientCall currently starving the engine, or None."""
        return self._pending

    def _starve(self, exc):
        self._pending = exc
        self._pending_got = 0
        raise exc

    def feed(self, kind, payload, side=None):
        """Supply ONE harvested client outcome.  The router calls this
        exactly once per client log line/outcome (scan-cursor
        discipline - never re-ingesting a line - is the router's).

        `side` (d6 only, optional): WHO rolled the line, when the
        caller can attribute it compliantly (the spectator binds
        roller nicks to sides via the zone-attributed draw lines).
        A side-tagged value is consumed only by that side's own rolls
        (or side-less consumers); untagged values serve anyone.  This
        is what keeps interleaved combat segments straight: the client
        renders A-roll, A-reroll, D-roll, D-reroll while the engine
        consumes A-roll, D-roll, A-reroll, D-reroll (MT-siege
        specimen) - side lanes make both orders one stream each.

        kinds and payloads:
          'd6'         list/tuple of int faces from ONE roll line.
                       CLIENT FACT 19 (G1 verbatim: 'If more dice are
                       rolled than requested, ignore from the
                       RIGHTMOST'): while a d6 ask is pending, a line
                       longer than the remaining need is trimmed from
                       the RIGHT; whole lines arriving after the need
                       is met are kept UNTRIMMED as cross-line surplus
                       feeding the step's next consumer (gesture-
                       granular acceptance; honesty rides the driver's
                       unconsumed() gate at step end).
                       Returns {'kept': [...], 'dropped': [...]} so
                       the router can chat the ignore notice.
                       Dwarven-ring reroll outcomes arrive HERE: the
                       reroll eats d6 values from this same stream
                       (ctx='dwarven_ring' on the starve) - no
                       separate face kind, no second answerer.
          'tile'       face string (or a HuntTile-like, or a
                       (face, pid, zone) tuple / {'face','pid','zone'}
                       dict) of ONE client-drawn hunt tile.
          'card'       (side_name, deck, cid) or (side_name, deck,
                       cid, pid): ONE client-drawn event card, cid
                       decoded from the commitment maps.
          'companion'  name string, or an iterable of acceptable names
                       (BOTH faces of dual-sided figures:
                       Strider/Aragorn, Grey/White), for ONE client
                       random-companion draw.
          'card_pick'  card id string, or an iterable of acceptable
                       ids (the driver's placeholder id for a blinded
                       human card), for ONE client random card pick
                       (WWSAT hand discard; funnel
                       game_logic.pick_random_card).
        (Provenance: the accept halves of ai_move_server.py
        _ingest_pending :12542-12801; d6 trim :12551-12621.)"""
        if kind == 'd6':
            if isinstance(payload, str):
                raise ValueError(
                    'feed d6: want a list of int faces, got a string '
                    '%r - the router parses lines, this module never '
                    'does' % (payload,))
            vals = [int(v) for v in payload]
            for v in vals:
                if not 1 <= v <= 6:
                    raise ValueError(
                        'feed d6: face %r outside 1..6 - upstream '
                        'parse defect, refusing to bank it' % (v,))
            kept, dropped = list(vals), []
            p = self._pending
            if p is not None and p.kind == 'd6':
                left = int(p.kw.get('need', 1)) - self._pending_got
                if left > 0 and len(kept) > left:
                    dropped = kept[left:]
                    kept = kept[:left]
                self._pending_got += len(kept)
                if self._pending_got >= int(p.kw.get('need', 1)):
                    self._pending = None
            tag = _side_name(side)
            self._feed['d6'].extend((tag, v) for v in kept)
            return {'kept': kept, 'dropped': dropped}
        if kind == 'tile':
            face, pid, zone = None, None, None
            if isinstance(payload, dict):
                face = payload.get('face')
                pid = payload.get('pid')
                zone = payload.get('zone')
            elif isinstance(payload, (tuple, list)):
                seq = list(payload) + [None, None, None]
                face, pid, zone = seq[0], seq[1], seq[2]
            else:
                face = payload
            if face is None:
                raise ValueError('feed tile: no face in %r' % (payload,))
            self._feed['tiles'].append((tile_face(face), pid, zone))
            if self._pending is not None \
                    and self._pending.kind == 'tile':
                self._pending = None
            return {'face': tile_face(face)}
        if kind == 'card':
            seq = list(payload)
            if len(seq) == 3:
                side_name, deck, cid = seq
                pid = None
            elif len(seq) == 4:
                side_name, deck, cid, pid = seq
            else:
                raise ValueError('feed card: want (side_name, deck, '
                                 'cid[, pid]), got %r' % (payload,))
            side_name = _side_name(side_name)
            deck = ('strategy' if str(deck).lower().startswith('strat')
                    else 'character')
            if (side_name, deck) not in _CARD_ATTR:
                raise ValueError('feed card: unknown deck %r'
                                 % ((side_name, deck),))
            key = (side_name, deck)
            self._feed['cards'].setdefault(key, []).append((cid, pid))
            p = self._pending
            if p is not None and p.kind == 'draws' \
                    and not self._missing_cards(p.kw['items']):
                self._pending = None
            return {'key': key, 'cid': cid}
        if kind == 'companion':
            if isinstance(payload, str):
                names = (payload,)
            else:
                names = tuple(str(n) for n in payload)
            if not names:
                raise ValueError('feed companion: empty name set')
            self._feed['companion'].append(names)
            if self._pending is not None \
                    and self._pending.kind == 'randomcomp':
                self._pending = None
            return {'names': names}
        if kind == 'card_pick':
            if isinstance(payload, str):
                cids = (payload,)
            else:
                cids = tuple(str(c) for c in payload)
            if not cids:
                raise ValueError('feed card_pick: empty id set')
            self._feed['card_pick'].append(cids)
            if self._pending is not None \
                    and self._pending.kind == 'card_pick':
                self._pending = None
            return {'cids': cids}
        # 'action_roll' is deliberately NOT feedable: rolled faces are
        # adopted engine-side (game_logic.adopt_rolled_faces) - one
        # answerer per quantity; this module only builds the ask.
        raise ValueError('feed: unknown kind %r (d6/tile/card/'
                         'companion; action_roll faces go through '
                         'game_logic.adopt_rolled_faces)' % (kind,))

    def _missing_cards(self, pairs):
        """The sublist of needed (side_name, deck) draws not covered by
        unconsumed feed items.
        (Provenance: ai_move_server.py:12082-12094.)"""
        from collections import Counter
        want = Counter(tuple(p) for p in pairs)
        have = Counter()
        for key, items in self._feed['cards'].items():
            have[key] = (len(items)
                         - self._feed_pos['cards'].get(key, 0))
        missing = []
        for key, n in want.items():
            for _ in range(max(0, n - have.get(key, 0))):
                missing.append(key)
        return missing

    # ---- directive builders --------------------------------------------
    def directives_for(self, call):
        """The Directive lines answering a NeedClientCall: '!CALL ...'
        client-executor lines and/or a chat ask for a HUMAN-side roll
        (G1: each side rolls its own dice; the human is ASKED in chat,
        the AI's own client is ORDERED via !CALL).  Pure builder - the
        router renders/whitelists chat and owns scan floors, the
        dry-pool self-heal and the balrog announce.
        (Provenance: ai_move_server.py:12378-12540, builder halves.)"""
        k = call.kind
        out = []
        if k == 'd6':
            n = int(call.kw.get('need', 1))
            his = (self.human_side is not None
                   and _side_name(call.kw.get('side'))
                   == self.human_side)
            if his:
                out.append(Directive('chat', 'roll %d %s'
                                     % (n, 'die' if n == 1
                                        else 'dice')))
            else:
                out.append(Directive('call', '!CALL combat_roll %d'
                                     % n))
        elif k == 'tile':
            out.append(Directive('call', '!CALL hunt_tile'))
        elif k == 'draws':
            for side_name, deck in call.kw['items']:
                out.append(Directive('call', '!CALL draw_%s_%s' % (
                    'fp' if side_name == 'FREE_PEOPLES' else 'sa',
                    'char' if deck == 'character' else 'strat')))
        elif k == 'action_roll':
            # G11 (corpus order): the SHADOW pool rolls FIRST - its
            # 'adds N eyes to the hunt' line rides the SA roll - then
            # the Free Peoples pool
            for side_name in sorted(call.kw['sides'], reverse=True):
                zone_code, _pids = call.kw['sides'][side_name]
                out.append(Directive(
                    'call', '!CALL roll_action_dice %s' % zone_code))
        elif k == 'randomcomp':
            # USER-MANDATED (2026-08-15): the Shadow ASKS out loud
            # before the client's built-in draws (parser-inert, ends
            # in '?')
            out.append(Directive('chat', 'random companion draw?'))
            out.append(Directive('call', '!CALL random_companion'))
        elif k == 'card_pick':
            # the client's OWN random FP-hand Character discard
            # ('Discard For WWSAT...', AiOpp execDirective 'wwsat':
            # its Math.random pick, Controls.452 grammar).  The
            # rendered line carries no identity; the router feeds the
            # ids from the engine hand it models.
            out.append(Directive('call', '!CALL wwsat'))
        else:
            raise RuntimeError('unknown NeedClientCall %r' % (k,))
        return out

    # ---- RNG-pinned legality probes -------------------------------------
    def pin(self):
        """Context manager for a legality probe (get_legal_actions)
        with a deterministic per-(step, probe-index) RNG pin.  The
        engine's _event_card_has_effect dry-runs card effects on the
        global RNG - for dice-rolling cards LEGALITY ITSELF is a roll.
        Pinning probe #N of a step to a fixed seed makes every retry
        re-enumerate the exact same legal set, no matter how much RNG
        the first attempt's MCTS consumed between probes.  Outside an
        open step: no-op.
        (Provenance: ai_move_server.py:12180-12200.)"""
        @contextlib.contextmanager
        def _cm():
            if not self._step_open:
                yield
                return
            ix = self._probe_ix
            self._probe_ix = ix + 1
            save = random.getstate()
            random.seed('probe:%d:%d' % (self._step_seed, ix),
                        version=2)
            try:
                yield
            finally:
                random.setstate(save)
        return _cm()

    # ---- journal (deterministic retry of decisions) ----------------------
    def journal_pick(self, actions, chooser, key=None):
        """Journal-aware chooser front end.  A step is retried across
        ticks while client randomness arrives; every retry must repeat
        the EXACT picks of the previous attempt, so only the first
        attempt runs `chooser` (which returns a ranked list; slot [0]
        is the decision).  A journaled pick that no longer matches any
        candidate is a hard inconsistency and halts loudly.  Outside
        an open step: straight pass-through.
        (Provenance: ai_move_server.py:976-1002.)"""
        if key is None:
            key = act_key
        if not self._step_open or not actions:
            return chooser(actions)
        ix = self._journal_ix
        if ix < len(self._journal):
            want = self._journal[ix]
            hit = next((a for a in actions if key(a) == want), None)
            if hit is None:
                self._halt(
                    'JOURNAL MISMATCH: recorded pick %r not among %d '
                    'legal candidates on retry - non-deterministic '
                    'step replay' % (want, len(actions)))
            self._journal_ix = ix + 1
            return [hit] + [a for a in actions if a is not hit]
        ranked = chooser(actions)
        if ranked:
            self._journal.append(key(ranked[0]))
            self._journal_ix = len(self._journal)
        return ranked

    # ---- consumers (called by the installed patches) ---------------------
    def _consume_d6(self, n, ctx='', side=None):
        """(Provenance: ai_move_server.py:11803-11813.)  G1: WHOSE dice
        these are rides the starve, so directives_for can ask the
        right person to roll."""
        want = _side_name(side)
        pos = self._feed_pos['d6']
        feed = self._feed['d6']
        # side-lane selection: a side-tagged entry serves only that
        # side's own rolls; untagged entries serve anyone; a side-less
        # consumer takes the stream in arrival order
        picks = [i for i, (tag, _v) in enumerate(feed)
                 if i not in pos
                 and (want is None or tag is None or tag == want)]
        if len(picks) < n:
            if ctx in ('hunt reroll', 'leader reroll') \
                    and self.reroll_decline_shim:
                # decline-equivalent (see __init__): consume what IS
                # fed, synthesize natural 1s for the rest
                vals = [feed[i][1] for i in picks]
                pos.update(picks)
                vals.extend([1] * (n - len(picks)))
                self.shim_uses.append((ctx, n - len(picks)))
                return vals
            self._starve(NeedClientCall('d6', need=n - len(picks),
                                        ctx=ctx, side=side))
        take = picks[:n]
        pos.update(take)
        return [feed[i][1] for i in take]

    def _consume_tile(self, state):
        """(Provenance: ai_move_server.py:11815-11894, MINUS the
        parser-authority heals B9/B9(c) - with the engine as the one
        state authority (design rule 1) the pool either holds the fed
        face or the census is genuinely broken and halts.)"""
        pos = self._feed_pos['tiles']
        if pos >= len(self._feed['tiles']):
            self._starve(NeedClientCall('tile'))
        face, pid, _zone = self._feed['tiles'][pos]
        tile = next((t for t in state.hunt_pool
                     if tile_face(t) == face), None)
        if tile is None:
            self._halt(
                'client drew hunt tile pid %s (face %r) but the '
                'engine pool holds %r - pool desync'
                % (pid, face,
                   sorted(tile_face(t) for t in state.hunt_pool)))
        self._feed_pos['tiles'] = pos + 1
        return tile

    def _consume_companion(self, candidates, ctx=''):
        """Forced random-companion pick: the CLIENT-drawn companion
        (Controls.637) instead of an engine-side draw.  Reached through
        the patched named funnel game_logic.pick_random_companion.
        The fed name set carries BOTH faces of dual-sided figures.
        (Provenance: ai_move_server.py:12009-12034; the pid->names
        resolution moved router-side with the log parsing.)"""
        pos = self._feed_pos['companion']
        if pos >= len(self._feed['companion']):
            self._starve(NeedClientCall('randomcomp', ctx=ctx))
        names = self._feed['companion'][pos]
        victim = next((c for c in candidates
                       if getattr(c, 'name', None) in names), None)
        if victim is None:
            self._halt(
                'client drew random companion %r but the engine '
                'candidate pool holds %r - pool desync (ctx %r)'
                % (names,
                   sorted(getattr(c, 'name', '?') for c in candidates),
                   ctx))
        self._feed_pos['companion'] = pos + 1
        return victim

    def _consume_card_pick(self, candidates, ctx=''):
        """Forced random CARD pick: the client's own random discard
        (rendered 'randomly discards a <deck> Card.' - identity never
        rendered; the driver feeds the id its own hand model assigns).
        Reached through the patched named funnel
        game_logic.pick_random_card."""
        pos = self._feed_pos['card_pick']
        if pos >= len(self._feed['card_pick']):
            self._starve(NeedClientCall('card_pick', ctx=ctx))
        cids = self._feed['card_pick'][pos]
        pick = next((c for c in candidates if str(c) in cids), None)
        if pick is None:
            self._halt(
                'client random card pick %r not in the engine '
                'candidate pool %r - hand desync (ctx %r)'
                % (cids, [str(c)[:40] for c in list(candidates)[:8]],
                   ctx))
        self._feed_pos['card_pick'] = pos + 1
        return pick

    def _consume_card(self, state, side, deck):
        """Forced draw of the CLIENT-drawn card: exact draw_event_card
        semantics (deck list -> hand, hand-limit flag, annotation,
        returns None) with the client's card instead of the engine's
        own random pop.  Deck-empty raises RuntimeError FIRST - the
        original contract game_logic.grant_draw relies on (an
        unfulfillable grant stamps NO debt); a starve raises
        NeedClientCall out of draw_event_card, which grant_draw
        answers by stamping state._owed_draws and re-raising - grant
        semantics live engine-side, this funnel never touches them.
        (Provenance: ai_move_server.py:12036-12079.)"""
        from game_state import Side
        deck_key = ('strategy'
                    if str(deck).lower().startswith('strat')
                    else 'character')
        attr = _CARD_ATTR[(side.name, deck_key)]
        deck_list = getattr(state, attr, [])
        if not deck_list:
            # same contract as the original: callers catch RuntimeError
            raise RuntimeError('%s empty' % attr)
        key = (side.name, deck_key)
        pos = self._feed_pos['cards'].get(key, 0)
        items = self._feed['cards'].get(key, [])
        if pos >= len(items):
            self._starve(NeedClientCall('draws', items=[key]))
        cid, pid = items[pos]
        self._feed_pos['cards'][key] = pos + 1
        if cid not in deck_list:
            self._halt(
                'client drew %s (pid %s) but the engine %s does not '
                'hold it - deck desync' % (cid, pid, attr))
        deck_list.remove(cid)
        hands = state.hands[side]
        hands.hand.append(cid)
        hands.needs_discard = len(hands.hand) > 6
        # provenance sentinel: every cid that ever entered a live hand,
        # so an unfunneled arrival is detectable (verify.py witness)
        self.cid_provenance.add(str(cid))
        try:
            from game_trace import emit_annotation
            emit_annotation(state, 'card_drawn', {
                'deck': deck_key, 'card_id': str(cid),
                'side': 'FP' if side is Side.FREE_PEOPLES else 'SH'})
        except Exception:
            pass

    def _consume_dwarven_face(self, state, side, die_idx):
        """CLIENT-native face for a Dwarven Ring reroll.

        The rule is a PHYSICAL procedure - the reroll 'repeats until
        the result is of a DIFFERENT face class than the original' -
        so it maps exactly onto rolling the real die again: each
        attempt eats one d6 from the SAME client feed the combat rolls
        use (ctx='dwarven_ring' on the starve) and indexes the die's
        real face multiset with it.  No new directive, no new client
        handler, no invented value; the result rides
        use_dwarven_ring's existing forced_face parameter.
        (Provenance: ai_move_server.py:11903-11956.)"""
        import game_logic as _gl
        die = state.dice[side][die_idx]
        orig = _gl._DWARVEN_FACE_CLASSES.get(die.current, frozenset())
        faces = list(die.faces)
        valid = [f for f in faces
                 if not (_gl._DWARVEN_FACE_CLASSES.get(f, frozenset())
                         & orig)]
        if not valid:
            raise RuntimeError(
                'No valid reroll result exists for this die')
        for _ in range(self.DWARVEN_REROLL_CAP):
            v = int(self._consume_d6(1, ctx='dwarven_ring')[0])
            face = faces[(v - 1) % len(faces)]
            if face in valid:
                print('    [dwarven] client d6 %d -> face %r (reroll '
                      'off %r)' % (v, getattr(face, 'name', face),
                                   getattr(die.current, 'name',
                                           die.current)))
                return face
        self._halt(
            'dwarven ring reroll consumed %d client d6 values without '
            'landing on a face outside the original class %r - face '
            'table %r cannot produce %r'
            % (self.DWARVEN_REROLL_CAP,
               sorted(str(c) for c in orig),
               [getattr(f, 'name', f) for f in faces],
               [getattr(f, 'name', f) for f in valid]))
