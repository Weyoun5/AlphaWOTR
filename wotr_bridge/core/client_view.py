"""ClientView — the pid-level CLIENT model the emitter planners read.

The old driver fed the emitter planners a log_replay.LogReplayer `r`
rebuilt from the raw command stream every tick (183 full re-parses in
one 200s leg).  The planners' actual read surface is FIVE fields
(measured census 2026-08-19 over wotr_bridge/emitter.py):

    r.state                        engine-shaped GameState
    r.piece_region                 pid -> zone name
    r.die_id_map                   pid -> (Side, die index)
    r.ai_drawn_pids                pids the AI's own draws produced
    r.get_unit_type_from_piece_id  pid -> unit type (swap-aware)

This adapter provides exactly that surface from CLIENT-STATED sources
only (rendered-baseline directive):
    * state          = THE engine state (sole authority, I1)
    * piece_region   = the client's own board dump, as the loop read it
                       for this poll (link.ClientLink.read_board)
    * die_id_map     = pid->(side, idx) bindings made at ROLL ADOPTION:
                       the client's roll line pairs pids with faces in
                       order, and adopt_rolled_faces lands those faces
                       on engine dice in the same order — the binding
                       is the client's own statement, never inferred
    * dynamic_types  = pid type swaps tracked from rendered kill/place
                       pairs (A17 / A25-R2 6b — the A28 class)
    * ai_drawn_pids  = fed by the randomness funnel's card results

No parsing, no rules, no second authority: every field is either the
engine or a client statement.
"""
import copy


class ClientView(object):
    def __init__(self, state_fn):
        self._state_fn = state_fn        # () -> engine GameState
        self.piece_region = {}           # pid -> zone name (client)
        self.die_id_map = {}             # pid -> (Side, idx)
        # BASE action-die pieces, known at birth (client fact 11): the
        # pre-roll die-carrying emissions (the D3 eye allocation) need a
        # binding before any roll has bound a pid.  Bonus dice (358/359,
        # 360-362) enter the pool later and stay roll-adoption-bound.
        from game_state import Side
        for _i in range(7):
            self.die_id_map['%03d' % (99 + _i)] = (Side.SHADOW, _i)
        for _i in range(4):
            self.die_id_map[str(210 + _i)] = (Side.FREE_PEOPLES, _i)
        self.ai_drawn_pids = set()
        self.dynamic_types = {}          # pid -> swapped unit type
        self.card_pid_map = {}           # cid -> pid (commitment decode
        #                                  inverse; known cards of either seat)
        self.fellowship_location = None  # pawn continuity for planners

    def probe(self, state=None):
        """A copy for a probe: dicts of its own (a renderer may write them) and, when `state`
        is given, that state in the place of the game's."""
        view = copy.copy(self)
        for name in ('piece_region', 'shadow_stronghold_regions', 'die_id_map', 'piece_mark'):
            if hasattr(view, name):
                setattr(view, name, dict(getattr(view, name)))
        if state is not None:
            view._state_fn = lambda: state
        return view

    # ---- the emitter read surface ------------------------------------
    @property
    def state(self):
        return self._state_fn()

    @property
    def piece_types(self):
        """emitter's own _unit_type reads `replayer.piece_types`
        (emitter.py:704) — WITHOUT this alias A17 swap-awareness
        silently degrades to static typing for every planner pick
        (LIFT-D finding, 2026-08-19)."""
        return self.dynamic_types

    def card_pid(self, cid):
        """cid -> client pid for authenticated cards of either seat
        (commitment-decode inverse; fact 10: the static alphabetical map is WRONG once
        the hands are claimed).  None = unknown (emission HOLDS)."""
        return self.card_pid_map.get(str(cid))

    def get_unit_type_from_piece_id(self, pid, side_hint=None):
        """Swap-aware pid type (A17): dynamic_types consulted FIRST —
        passing None measured the STATIC type and was wrong for every
        downgraded piece (the 428/428 lesson)."""
        import log_replay as lr
        return lr.get_unit_type_from_piece_id(pid, self.dynamic_types,
                                              side_hint)


