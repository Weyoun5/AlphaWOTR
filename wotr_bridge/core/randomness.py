"""The client-randomness funnel: every die, tile, card and random pick of the real game comes
from the CLIENT, through ONE class, ClientRandomness, that owns all of its state.

The engine's own random funnels are patched once per process (install) and divert to the
bank of client results while a real step runs (advance).  A step that needs a result the
bank does not hold raises NeedClientCall; the loop asks the client (directives_for), the
result is fed in (feed) and the step is tried again - exactly as before (the journal of
picks, the per-step random streams, the probe pins).  Not here: reading the client's lines
(the link calls feed once per result), and the faces of an action roll (the engine adopts
them through game_logic.adopt_rolled_faces; this module only builds the ask).  The random
discard is the client's own: its line feeds the 'card_pick' funnel, the card is never named.

Facts of the client this module keeps to:
  fact 19  d6 excess drops from the RIGHT within one roll line  (feed)
  G1       each side rolls its own dice and draws its own cards: the AI's
           are '!CALL's under its nick; the person's are made with his own
           hand, or by the client under HIS nick, as he chose (settings;
           directives_for is the one place that says which)
  G11      the SHADOW pool rolls first  (directives_for 'action_roll')
  scoping  real-trajectory only: funnels divert at _adv_depth > 0
           (inside advance()); MCTS clones run at depth 0 and keep the
           engine's own RNG  (install/advance)

ASCII only (cp1252 console).
"""
import contextlib
import copy
import random
from collections import Counter, namedtuple

from .settings import Settings

# R6-DEAD-CODE-4: src/ and wotr_bridge/ come from live/__init__.py's one path
# setup (R5-LIVE-PACKAGE-SYSPATH), which runs before any submodule in every
# package shape; this module keeps no copy of it.


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

    def __init__(self, kind, box=None, **kw):
        self.kind = kind
        self.kw = kw
        # the box of the person's (settings) that governs who makes HIS cards or action dice of this
        # request - said by whoever knows what caused it: 'turn' (the turn's start), 'effects' (a
        # step of a card of his own), None (none governs it); his d6 are always under 'dice'
        self.box = box
        super().__init__('%s %r' % (kind, kw))

    def makers(self):
        """The names of the sides whose own dice or cards it asks for (none: a tile, a lot - nobody's)."""
        kw = self.kw
        return {_side_name(s) for s in ([kw['side']] if kw.get('side') is not None else list(
            kw.get('sides', ())) + [s for s, _deck in kw.get('items', ())])}


_ELVEN_TILES = frozenset(('elven cloaks', 'elven rope'))


def tile_class(tile):
    """The tile identity a player can see: tile_face, with Elven Cloaks and
    Elven Rope as one class."""
    face = tile_face(tile)
    return 'elven cloaks' if face in _ELVEN_TILES else face


# A directive answering a NeedClientCall.  kind:
#   'call'  text is the full '!CALL ...' line for AiOpp
#   'chat'  text is the nick-less message BODY; the router renders it
#           through the single whitelisted chat exit (design rule 7)
#   'hand'  nothing is sent: the PERSON makes this part with his own hand, and the loop asks
#           him in its own words; text is what words.by_hand needs (how many dice, which deck)
Directive = namedtuple('Directive', ('kind', 'text'))

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


def _cursors():
    """The consume cursors of one attempt, at its start.  A consumer moves ONLY these, never the
    bank: a look-ahead that puts them back has taken nothing.  'cut': the dice of an early roll
    that a request passed over (its surplus on the right, or a roll too short for it)."""
    return {'d6': set(), 'cut': set(), 'tiles': 0, 'cards': {}, 'companion': 0, 'card_pick': 0}


# kind of client result -> (its bank, the kind of request that asks for it, plain words)
_KINDS = {'d6': ('d6', 'd6', 'd6'), 'tile': ('tiles', 'tile', 'Hunt tile'),
          'card': ('cards', 'draws', 'card'),
          'companion': ('companion', 'randomcomp', 'random Companion'),
          'card_pick': ('card_pick', 'card_pick', 'random card pick')}


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
# The patch targets are MODULE globals: the funnels install once per
# process and every closure consults the instance that is CURRENTLY
# deciding via _ACTIVE (a closure frozen on a stale instance reads
# depth 0 forever and falls through to raw random.randint).
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
      step_commit() / abort_step() close the step: what it did not use
                                   is dropped, by name (take_dropped)

    `human_side` (Side or name string, optional): the person's side.  What
    of a request is HIS - his dice, his cards, his action dice - he makes
    with his own hand or has the client make for him: `prefers`, the boxes
    he ticked (settings.Settings), asked by directives_for.
    """

    # cap on how many client d6 values one dwarven reroll may eat.  A
    # real die always has SOME face outside the original's class (the
    # engine checks `valid` is non-empty first), so the loop terminates
    # with probability 1 - the cap exists so a face-table bug becomes a
    # loud halt instead of an infinite drain on the client feed.
    DWARVEN_REROLL_CAP = 24

    def __init__(self, human_side=None, seed=None, prefers=None):
        self.human_side = _side_name(human_side)
        self.prefers = prefers or Settings()
        # WHO MAKES the person's part of the open request: (that request - its kind and number -, he
        # makes it with his own hand).  Decided when the request is first asked; it stands until
        # the request is answered or its step ends (directives_for)
        self.hand = (None, False)
        self.installed = False
        # last tripwire/desync halt message (STICKY: survives resets,
        # so a quarantined halt stays visible; instrument rule - a
        # component that cannot proceed fails loudly, never silently)
        self.sticky_halt = None
        # every card id that ever entered a live hand through the
        # funnel - verify.py's unfunneled-arrival witness reads this
        self.cid_provenance = set()
        # the bank: client results not yet used, one list per kind (cards: per deck).  Every
        # entry ENDS with a mark: 0 when the result answered the open request; when it came
        # early, the number of values of its result from this one on (a roll of 3: 3, 2, 1;
        # a tile: 1), so a request takes or passes over an early result as a whole.
        # d6 entries are (side of the roller, value, mark), consumed by a SET of indices (the
        # two sides' rolls interleave); the other banks are consumed from the front.
        self._feed = {'d6': [], 'tiles': [], 'cards': {},
                      'companion': [], 'card_pick': []}
        self._feed_pos = _cursors()
        self._dropped = []             # results dropped unused, not yet told (take_dropped)
        # the NeedClientCall currently starving the engine
        self._pending = None
        self._pending_got = 0          # d6 faces accepted toward it
        self.answered = 0              # requests answered in full: the number of the open one
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
        from . import emitter
        # the emitter planners that can SOURCE randomness (plan_roll,
        # plan_draw_card) must halt rather than invent - same fact as
        # the funnels: from this point the client owns randomness
        emitter.LIVE_MODE = True
        if _INSTALLED:
            return
        _INSTALLED = True
        import game_logic as gl
        import game_state as gs
        import combat as cb
        import event_effects as ee
        import onslaught_choices as oc

        orig_gl_batch = gl.roll_d6_batch
        orig_cb_batch = cb.roll_d6_batch
        orig_tile = gl.pick_hunt_tile
        orig_pop = gl.pop_hunt_tile
        orig_draw = gl.draw_event_card
        orig_dwarf = gl.use_dwarven_ring
        orig_pick_comp = gl.pick_random_companion
        orig_pick_card = gl.pick_random_card
        orig_pre_combat = cb._apply_pre_combat_for_side
        orig_post_casualty = cb._apply_post_casualty_for_side
        orig_onslaught = cb.apply_onslaught_choice
        orig_onslaught_advance = oc._advance
        orig_event_card = ee.play_event_card
        orig_dead_men = ee.effect_dead_men_of_dunharrow

        def _gl_batch(n, ctx='', side=None):
            a = _active_live()
            if a is not None:
                # The engine's Hunt helper predates side-tagged batches.
                # Shadow owns both the initial Hunt roll and its rerolls,
                # regardless of which side initiated the Fellowship move.
                if side is None and ctx in ('hunt roll', 'hunt reroll', 'shelob d6 damage'):
                    side = gs.Side.SHADOW
                scope = getattr(a, '_roll_context', None) or {}
                if side is None:
                    side = scope.get('side')
                if ctx == 'card effect' and scope:
                    ctx = 'event card'
                return a._consume_d6(n, ctx, side=side)
            # R2-TR-test-fixwave-B8-01: the inactive fall-through must be
            # TRANSPARENT.  When a second copy of this module is imported
            # under another package name (live.* vs wotr_bridge.live.*,
            # the bridge test suite) its funnel wraps this one; dropping
            # `side` here made the inner, active copy see side=None and
            # ask the wrong player to roll (G1).  A side-less call keeps
            # the old call shape for originals that take no `side`.
            if side is None:
                return orig_gl_batch(n, ctx)
            return orig_gl_batch(n, ctx, side=side)

        def _cb_batch(n, ctx='', side=None):
            a = _active_live()
            if a is not None:
                scope = getattr(a, '_roll_context', None) or {}
                return a._consume_d6(n, scope.get('ctx', ctx),
                                     side=side if side is not None else scope.get('side'),
                                     role=scope.get('role'))
            # R2-TR-test-fixwave-B8-01: forward the dice owner (see _gl_batch)
            if side is None:
                return orig_cb_batch(n, ctx)
            return orig_cb_batch(n, ctx, side=side)

        @contextlib.contextmanager
        def _effect_scope(combat, is_attacker, timing):
            a = _active_live()
            if a is None:
                yield
                return
            previous = getattr(a, '_roll_context', None)
            role = 'attacker' if is_attacker else 'defender'
            card = getattr(combat, role + '_card', None)
            a._roll_context = dict(side=getattr(combat, role + '_side'),
                                   role=role, ctx=timing + ' ' + str(getattr(card, 'combat_name', '')))
            try:
                yield
            finally:
                a._roll_context = previous

        def _pre_combat(combat, is_attacker):
            with _effect_scope(combat, is_attacker, 'pre-combat'):
                return orig_pre_combat(combat, is_attacker)

        def _post_casualty(combat, is_attacker, choices=None):
            with _effect_scope(combat, is_attacker, 'post-casualty'):
                return orig_post_casualty(combat, is_attacker, choices)

        def _onslaught(*args, **kwargs):
            a = _active_live()
            if a is None:
                return orig_onslaught(*args, **kwargs)
            previous = getattr(a, '_roll_context', None)
            a._roll_context = dict(side=kwargs.get('own_side', args[3] if len(args) > 3 else None),
                                   ctx='post-casualty Onslaught')
            try:
                return orig_onslaught(*args, **kwargs)
            finally:
                a._roll_context = previous

        def _onslaught_advance(state, pc):
            # Staged unit payments use a different engine entry point from
            # apply_onslaught_choice. Its final combat_roll is the card's
            # additional 4+ attack, not another ordinary combat round.
            a = _active_live()
            if a is None:
                return orig_onslaught_advance(state, pc)
            payment = pc['_onslaught_payment']
            role, _ = oc._roles(payment)
            previous = getattr(a, '_roll_context', None)
            a._roll_context = dict(side=pc[role + '_side'], role=role,
                                   ctx='post-casualty Onslaught')
            try:
                return orig_onslaught_advance(state, pc)
            finally:
                a._roll_context = previous

        @contextlib.contextmanager
        def _owner_scope(side):
            a = _active_live()
            if a is None:
                yield
                return
            previous = getattr(a, '_roll_context', None)
            a._roll_context = dict(side=side)
            try:
                yield
            finally:
                a._roll_context = previous

        def _event_card(state, side, *args, **kwargs):
            with _owner_scope(side):
                return orig_event_card(state, side, *args, **kwargs)

        def _dead_men(*args, **kwargs):
            with _owner_scope(gs.Side.FREE_PEOPLES):
                return orig_dead_men(*args, **kwargs)

        def _owned_card_handler(handler):
            # B8-04: ai.apply_action resolves a chained 'direct'
            # card_option stage by calling CARD_EFFECT_HANDLERS[name]
            # directly (the card already left the hand), and
            # event_effects re-enters a handler after its casualty
            # pending the same way - play_event_card's owner scope is
            # not on the stack there, so the card's d6 batch reached
            # the funnel side-less and the AI's client was ordered to
            # roll the HUMAN's card dice (G1).  The card's player owns
            # its dice, exactly as on the play_event_card path; an
            # enclosing scope (play_event_card, a combat effect) wins,
            # and a card-less pseudo-handler stays unscoped as before.
            import functools

            @functools.wraps(handler)
            def _owned(state, card, *args, **kwargs):
                a = _active_live()
                side = getattr(card, 'side', None)
                if (a is None or getattr(a, '_roll_context', None) is not None
                        or not isinstance(side, gs.Side)):
                    return handler(state, card, *args, **kwargs)
                with _owner_scope(side):
                    return handler(state, card, *args, **kwargs)
            return _owned

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
            # the NAMED companion funnel: ONE patched site, the engine
            # names the draw itself
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
        cb._apply_pre_combat_for_side = _pre_combat
        cb._apply_post_casualty_for_side = _post_casualty
        cb.apply_onslaught_choice = _onslaught
        oc._advance = _onslaught_advance
        ee.play_event_card = _event_card
        ee.effect_dead_men_of_dunharrow = _dead_men
        # B8-04: every registered card handler carries its card's owner
        # scope (in place: ai.py and event_effects look handlers up in
        # this same dict at call time)
        for _key, _handler in list(ee.CARD_EFFECT_HANDLERS.items()):
            ee.CARD_EFFECT_HANDLERS[_key] = _owned_card_handler(_handler)
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
        interleaved MCTS consumed."""
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
        journal-mismatch), journal cursor and feed cursors reset."""
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
        self._feed_pos = _cursors()

    def step_commit(self):
        """The engine step is done: close it (_close).  -> (did a result of the client go
        into it, the Hunt tiles it drew)"""
        used = self._feed_pos
        took = (bool(used['d6'] or used['tiles'] or used['companion'] or used['card_pick']
                     or any(used['cards'].values())), [t[0] for t in self._feed['tiles'][:used['tiles']]])
        self._close('step done')
        return took

    def probe(self):
        """Context manager for a look-ahead (the matcher asks what a move would open): the
        engine and the step machine may run inside and nothing is consumed - the cursors, the
        journal, the open request and both random streams are as before afterwards."""
        @contextlib.contextmanager
        def _cm():
            saved = {name: getattr(self, name) for name in (
                '_step_open', '_journal_ix', '_probe_ix', '_step_rng_cur', '_step_rng_base',
                '_step_outer_rng', '_step_seed', '_pending', '_pending_got', '_adv_depth', 'hand')}
            journal, feed_pos = list(self._journal), copy.deepcopy(self._feed_pos)
            outer, own = random.getstate(), self._rng.getstate()
            try:
                yield
            finally:
                for name, value in saved.items():
                    setattr(self, name, value)
                self._journal, self._feed_pos = journal, feed_pos
                random.setstate(outer)
                self._rng.setstate(own)
        return _cm()

    @property
    def step_open(self):
        """True while an engine step is open (begin_attempt without a
        step_commit/abort_step)."""
        return bool(self._step_open)

    def abort_step(self, reason=''):
        """Close a step WITHOUT committing it (a take-back, a changed board answer, a refused
        choice, an error that came again; a first error only ends the attempt).  Nothing of
        it counts as used; the journal goes, so that the next step's chooser runs (a kept
        pick would be replayed against another menu).  Safe on a closed step."""
        self._feed_pos = _cursors()
        self._close('step abandoned (%s)' % reason)

    def _close(self, how):
        """THE rule of the end of a step: what it used leaves the bank - and so does what it
        did NOT use, by name.  A result answers the step it came for and never a later one:
        left in the bank it is taken by the next step of its kind, which then shows no roll
        or draw of its own (and a tile whose face has left the pool since stops every
        retry).  Only what the PERSON made before he was asked stays - he may roll, or draw
        his card, for a request he sees coming, steps ahead (the card of an action whose die
        an Elven Ring turns first), while the client makes nothing for the AI except on a
        call; when his early result is void is the server's rule (drop_early)."""
        lost, early = Counter(), 0
        for kind, words, bank, used, passed, makers in self._lanes():
            stay = []
            for i, e in enumerate(bank):
                if i in used:
                    early += bool(e[-1])
                elif e[-1] and i not in passed and makers[i] is not None \
                        and makers[i] == self.human_side:
                    stay.append(e)
                else:
                    lost[words] += 1        # (a card and a tile are pieces of the client's board)
                    self._dropped.append(str(e[1]) if kind in ('card', 'tile') and e[1] is not None else None)
            bank[:] = stay
        self._feed_pos = _cursors()
        self._step_open = False
        self._journal, self._journal_ix, self._probe_ix = [], 0, 0
        self._pending, self._pending_got, self.hand = None, 0, (None, False)
        if early:
            self._say('%s: it used %d result(s) the client produced before they were asked'
                      % (how, early))
        for name, n in sorted(lost.items()):
            if n:
                self._say('%s: %d x %s dropped, not used by it' % (how, n, name))

    def take_dropped(self):
        """The client results dropped unused at a step's end since the last call, one entry each:
        the piece of a card or a tile, else None (a die, Companion, pick).  Said once by the caller."""
        gone, self._dropped = self._dropped, []
        return gone

    def drop_early(self, side):
        """The server's rule ends the wait of `side`'s early results (the AI acts, a battle or a
        Hunt opens): its dice leave the bank - returns how many (for the caller's sentence) -
        and so does a card it drew early, which is dropped as unused (take_dropped).
        Between attempts only: the cursors of a running attempt index the bank."""
        tag, n = _side_name(side), 0
        if self._feed_pos != _cursors():
            return 0
        for kind, _words, bank, _used, _passed, makers in self._lanes():
            keep = [e for e, maker in zip(bank, makers) if maker != tag]
            if kind == 'card':      # (a piece of the client's board: it goes back as every unused result)
                self._dropped += [str(e[1]) if e[1] is not None else None for e, m in zip(bank, makers) if m == tag]
            else:
                n += len(bank) - len(keep)
            bank[:] = keep
        return n

    def _say(self, text):
        """The funnel's own log line (the server's console): no client result is kept
        early, used early or dropped without one."""
        print('    [funnel] %s' % text, flush=True)

    # ---- pending + feeding ---------------------------------------------
    def pending(self):
        """The NeedClientCall currently starving the engine, or None."""
        return self._pending

    def _starve(self, exc):
        """The engine needs a client result the bank does not hold: `exc` becomes the open
        request (the results that come in for it are its own) and goes up to the caller."""
        self._pending = exc
        self._pending_got = 0
        raise exc

    def draws_request(self, pairs):
        """The request for the draws among `pairs` (side name, deck) the bank does not hold
        yet, or None when it holds them all.  The caller orders these draws from the client
        itself (the turn-start draws); made here, the request is the open one like any
        other, and the cards that come in answer it."""
        missing = self._missing_cards(pairs)
        if not missing:
            return None
        self._pending, self._pending_got = NeedClientCall('draws', box='turn', items=missing), 0
        return self._pending

    def feed(self, kind, payload, side=None, undone=None):
        """Supply ONE client result (the caller hands each in exactly once); what becomes
        of it is the one rule of _arrive.  Returns {'kept', 'dropped', 'early', 'second'}:
        `dropped` and `second` are for the caller to say.  undone: the piece of a card of this
        deck whose draw was undone in the client - its result, if no step used it, is void
        (this one is the draw).

          'd6'         list of int faces of ONE roll line; `side` = the side of whoever
                       ROLLED it (None when the caller cannot tell: AI vs AI).  A Dwarven
                       Ring reroll eats d6 of this same stream.
          'tile'       face string, a HuntTile-like, (face, pid, zone) or a dict of them
          'card'       (side_name, deck, cid[, pid]): cid as the client's commitment
                       names it
          'companion'  a name, or the names of both faces of a two-sided figure
          'card_pick'  a card id, or the ids it may be (the client's random discard)"""
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
            tag = _side_name(side)
            return self._arrive(kind, vals, [(tag, v) for v in vals], maker=tag)
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
            face = tile_face(face)
            return self._arrive(kind, [face], [(face, pid, zone)])
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
            if undone is not None:
                bank, used = self._lane(kind, (side_name, deck))[:2]
                bank[len(used):] = [e for e in bank[len(used):] if e[1] != undone]
            return self._arrive(kind, [cid], [(cid, pid)], key=(side_name, deck))
        if kind == 'companion':
            if isinstance(payload, str):
                names = (payload,)
            else:
                names = tuple(str(n) for n in payload)
            if not names:
                raise ValueError('feed companion: empty name set')
            return self._arrive(kind, [names], [(names,)])
        if kind == 'card_pick':
            if isinstance(payload, str):
                cids = (payload,)
            else:
                cids = tuple(str(c) for c in payload)
            if not cids:
                raise ValueError('feed card_pick: empty id set')
            return self._arrive(kind, [cids], [(cids,)])
        # 'action_roll' is deliberately NOT feedable: rolled faces are
        # adopted engine-side (game_logic.adopt_rolled_faces) - one
        # answerer per quantity; this module only builds the ask.
        raise ValueError('feed: unknown kind %r (d6/tile/card/'
                         'companion; action_roll faces go through '
                         'game_logic.adopt_rolled_faces)' % (kind,))

    # ---- the owner of a client result: one rule for every kind -------------
    # A result is the answer to ONE request.  _arrive binds it to the open request or
    # keeps it as early; _take hands it to the request that uses it; _close drops, by
    # name, what the step did not use.  Nothing else puts a result in or takes one out.
    def _lane(self, kind, key=None):
        """One bank with the cursors of the running attempt: (entries, indices used, indices
        passed over, who made each entry - None where the client's line does not say)."""
        f, p, name = self._feed, self._feed_pos, _KINDS[kind][0]
        if kind == 'd6':
            return f['d6'], p['d6'], p['cut'], [e[0] for e in f['d6']]
        bank = f[name] if key is None else f[name].setdefault(key, [])
        return bank, range(p[name] if key is None else p[name].get(key, 0)), (), [None] * len(bank)

    def _lanes(self):
        """Every bank as _lane gives it, with its kind and its plain name in front - and, for the
        end of a step, WHOSE each entry is: a card is its deck's side's (nobody else draws it)."""
        for kind, (name, _request, words) in _KINDS.items():
            for key in (list(self._feed[name]) if kind == 'card' else [None]):
                bank, used, passed, makers = self._lane(kind, key)
                yield (kind, words if key is None else '%s %s %s' % (key + (words,)), bank, used, passed,
                       makers if key is None else [key[0]] * len(bank))

    def owed(self):
        """What the open request still lacks: {deck of a draw, or None: how many}."""
        p = self._pending
        if p is None:
            return {}
        if p.kind == 'draws':
            return Counter(self._missing_cards(p.kw['items']))
        return {None: max(0, int(p.kw.get('need', 1)) - self._pending_got)}

    def _arrive(self, kind, values, rows, maker=None, key=None):
        """THE rule of arrival (`rows`: the dice of one roll line, or the one tile, card,
        Companion or card pick).  The result answers the open request of its kind and
        maker: it is kept for that request as far as the request still needs it, and what
        is beyond is handed back ('If more dice are rolled than requested, ignore from the
        RIGHTMOST').  Or nobody asked: it came EARLY and waits for the next request of its
        kind and maker - one such result per maker; a second is handed back whole (a
        double click is never the next roll).  Every case is said in the log.  `second`: kept
        early while an ASKED result of its maker still lies in the bank (he rolled twice)."""
        bank, used, passed, makers = self._lane(kind, key)
        p, what = self._pending, '%s%s' % (_KINDS[kind][2], ' of %s' % maker if maker else '')
        mine = p is not None and p.kind == _KINDS[kind][1] and (
            maker is None or p.kw.get('side') is None or maker == _side_name(p.kw['side']))
        need = self.owed().get(key, 0) if mine else 0
        n, early = len(rows), not need
        second = early and maker is not None and any(not e[-1] and makers[i] == maker for i, e in enumerate(bank))
        if need:
            n = min(n, need)
            bank.extend(r + (0,) for r in rows[:n])
            self._pending_got += n
            if not sum(self.owed().values()):
                self._pending, self.answered = None, self.answered + 1
            if n < len(rows):
                self._say('%d x %s more than asked: ignored' % (len(rows) - n, what))
        elif any(e[-1] and makers[i] == maker and i not in used and i not in passed
                 for i, e in enumerate(bank)):
            n = 0
            self._say('%s came again before any was asked: ignored' % what)
        else:
            bank.extend(r + (n - k,) for k, r in enumerate(rows))
            self._say('%s came before it was asked: it may answer the next request for '
                      'it, once' % what)
        return {'kept': values[:n], 'dropped': values[n:], 'early': early, 'second': bool(second and n)}

    def _take(self, kind, n, maker=None, key=None):
        """THE rule of use: the next n results of `kind` (made by `maker`; a result with
        no maker answers anyone), in the order they came.  A result that answered a request
        of this step is taken as it is.  An EARLY result answers this request as a whole or
        not at all: with at least as many values as are still needed, its first ones count
        and the rightmost are ignored; with fewer, it was not meant for this request and is
        passed over.  Only the cursors move, never the bank: a look-ahead that puts the
        cursors back has taken nothing.  Returns the entries taken; fewer than n: the rest
        must be asked for."""
        bank, used, passed, makers = self._lane(kind, key)
        take, over, i = [], [], 0
        while i < len(bank) and len(take) < n:
            line, need = bank[i][-1], n - len(take)
            other = maker is not None and makers[i] is not None and makers[i] != maker
            if other or i in used or i in passed:
                i += 1
            elif not line:
                take.append(i)
                i += 1
            else:
                whole = list(range(i, i + line))
                take += whole[:need] if line >= need else []
                over += whole[need:] if line >= need else whole
                i += line
        pos, name = self._feed_pos, _KINDS[kind][0]
        if kind == 'd6':
            pos['d6'].update(take)
            pos['cut'].update(over)
        elif take or over:
            if key is None:
                pos[name] = max(take + over) + 1
            else:
                pos[name][key] = max(take + over) + 1
        return [bank[i] for i in take]

    def _missing_cards(self, pairs):
        """The sublist of needed (side_name, deck) draws not covered by
        unconsumed feed items."""
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
        """THE Directives answering a NeedClientCall, and THE one place that says WHO MAKES each
        part of it.  What is nobody's or the AI's the client is ORDERED to make ('call').  What is
        the PERSON's own - his d6, his cards, his action dice - the client makes for him too when
        the box that governs it is ticked (prefers), and under HIS nick: '!CALL own_roll' for his
        d6, the draw and action-roll calls as they are; when it is not, he makes it with his own
        hand ('hand': nothing is sent, the loop asks him once and takes the result whenever it
        comes).  The box: 'dice' for his d6 always; for his cards and action dice the one the
        request carries (call.box) - 'turn' at the turn's start, 'effects' in a step of a card of
        his own, None where none governs it (a click on "draw a card": the client makes it).
        DECIDED ONCE, when the request is first asked: self.hand stands while that request is
        open - a box he ticks meanwhile governs the next one - and only his word "finish" hands
        the open one over to the client (the loop sets self.hand).  Pure builder otherwise."""
        k = call.kind
        his = self.human_side is not None and self.human_side in call.makers()
        if self.hand[0] != (k, self.answered):
            self.hand = ((k, self.answered), his and not self.prefers['dice' if k == 'd6' else call.box])
        mine = lambda side: self.hand[1] and _side_name(side) == self.human_side
        out = []
        if k == 'd6':
            n = int(call.kw.get('need', 1))
            out.append(Directive('hand', n) if mine(call.kw.get('side')) else Directive(
                'call', '!CALL %s %d' % ('own_roll' if his else 'combat_roll', n)))
        elif k == 'tile':
            out.append(Directive('call', '!CALL hunt_tile'))
        elif k == 'draws':
            for side_name, deck in call.kw['items']:
                out.append(Directive('hand', deck) if mine(side_name) else Directive('call', '!CALL draw_%s_%s' % (
                    'fp' if side_name == 'FREE_PEOPLES' else 'sa', 'char' if deck == 'character' else 'strat')))
        elif k == 'action_roll':
            # G11 (corpus order): the SHADOW pool rolls FIRST - its
            # 'adds N eyes to the hunt' line rides the SA roll - then
            # the Free Peoples pool
            for side_name in sorted(call.kw['sides'], reverse=True):
                zone_code, _pids = call.kw['sides'][side_name]
                out.append(Directive('hand', None) if mine(side_name) else Directive(
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
        an open step: straight pass-through."""
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
    def _consume_d6(self, n, ctx='', side=None, role=None):
        """n d6 of `side` from the bank, or a starve that carries WHOSE dice they are (each
        side rolls its own: the server asks the right roller).  A side's dice answer only
        that side's rolls; a die with no side answers anyone; a side-less request takes the
        stream in arrival order."""
        got = self._take('d6', n, maker=_side_name(side))
        if len(got) < n:
            metadata = dict(need=n - len(got), ctx=ctx, side=side)
            if role is not None:
                metadata['role'] = role
            self._starve(NeedClientCall('d6', **metadata))
        return [e[1] for e in got]

    def _consume_tile(self, state):
        """(Provenance: ai_move_server.py:11815-11894, MINUS the
        parser-authority heals B9/B9(c) - with the engine as the one
        state authority (design rule 1) the pool either holds the fed
        face or the census is genuinely broken and halts.)"""
        got = self._take('tile', 1)
        if not got:
            call = NeedClientCall('tile')
            # what the engine's pool and used pile hold at this very draw (the
            # engine may have refilled the pool inside this step): the server
            # makes the client's pool the same before it orders the draw
            call.pool = [tile_class(t) for t in state.hunt_pool]
            call.used = [tile_class(t) for t in state.hunt_discard]
            self._starve(call)
        face, pid = got[0][:2]
        tile = next((t for t in state.hunt_pool
                     if tile_face(t) == face), None)
        if tile is None:
            # Elven Cloaks and Elven Rope are one tile to the player (same
            # face, damage 0); the CARDS stay distinct
            tile = next((t for t in state.hunt_pool
                         if tile_class(t) == tile_class(face)), None)
        if tile is None:
            self._halt(
                'client drew hunt tile pid %s (face %r) but the '
                'engine pool holds %r - pool desync'
                % (pid, face,
                   sorted(tile_face(t) for t in state.hunt_pool)))
        return tile

    def _consume_companion(self, candidates, ctx=''):
        """Forced random-companion pick: the CLIENT-drawn companion
        (Controls.637) instead of an engine-side draw.  Reached through
        the patched named funnel game_logic.pick_random_companion.
        The fed name set carries BOTH faces of dual-sided figures.
        (Provenance: ai_move_server.py:12009-12034; the pid->names
        resolution moved router-side with the log parsing.)"""
        got = self._take('companion', 1)
        if not got:
            self._starve(NeedClientCall('randomcomp', ctx=ctx))
        names = got[0][0]
        victim = next((c for c in candidates
                       if getattr(c, 'name', None) in names), None)
        if victim is None:
            self._halt(
                'client drew random companion %r but the engine '
                'candidate pool holds %r - pool desync (ctx %r)'
                % (names,
                   sorted(getattr(c, 'name', '?') for c in candidates),
                   ctx))
        return victim

    def _consume_card_pick(self, candidates, ctx=''):
        """Forced random CARD pick: the client's own random discard
        (rendered 'randomly discards a <deck> Card.' - identity never
        rendered; the driver feeds the id its own hand model assigns).
        Reached through the patched named funnel
        game_logic.pick_random_card."""
        got = self._take('card_pick', 1)
        if not got:
            self._starve(NeedClientCall('card_pick', ctx=ctx))
        cids = got[0][0]
        pick = next((c for c in candidates if str(c) in cids), None)
        if pick is None:
            self._halt(
                'client random card pick %r not in the engine '
                'candidate pool %r - hand desync (ctx %r)'
                % (cids, [str(c)[:40] for c in list(candidates)[:8]],
                   ctx))
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
        semantics live engine-side, this funnel never touches them."""
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
        got = self._take('card', 1, key=key)
        if not got:
            self._starve(NeedClientCall('draws', items=[key]))
        cid, pid = got[0][:2]
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
        use_dwarven_ring's existing forced_face parameter."""
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
            v = int(self._consume_d6(1, ctx='dwarven_ring', side=side)[0])
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
