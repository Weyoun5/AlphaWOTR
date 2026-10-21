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
    * piece_region   = the client's own board dump (client_board.
                       read_board), refreshed on demand
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
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, '..', '..', 'src'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from . import client_board


class ClientView(object):
    def __init__(self, state_fn, dump_path=None):
        self._state_fn = state_fn        # () -> engine GameState
        self.dump_path = dump_path
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

    # ---- client-stated updates ----------------------------------------
    def refresh_dump(self):
        """Re-read the client's board dump (torn-safe).  Returns True
        when a stable dump landed."""
        if not self.dump_path:
            return False
        d = client_board.read_board(self.dump_path)
        if d is None:
            return False
        from .shadow_siege_slots import project_dump
        self.native_piece_region = dict(d)
        from .temporary_combat_figures import project_dump as project_combat_figures
        self.piece_region = project_combat_figures(self, project_dump(self, d))
        return True

    def bind_die(self, pid, side, idx):
        """Registration slots are UNIQUE per (side, idx).  BL-25
        (shgame3): the client's Allocate Eye picked pid 105 while the
        static seed still held 105 -> reg6; binding without evicting
        left TWO pids on one reg and none on another, and
        die_piece_for's last-wins inversion then shifted every
        published index (idx6 fell off the end and the whole seat
        wedged).  Binding P to a slot evicts any other pid on that
        slot; P's own old slot is released by the dict overwrite."""
        pid = str(pid)
        key = (side, int(idx))
        old = self.die_id_map.get(pid)
        for _p2 in [p for p, k in self.die_id_map.items()
                    if p != pid and k == key]:
            if old is not None and old != key:
                # SWAP: the displaced pid takes P's old slot, so a
                # bind never reduces registration coverage (a
                # transient hole would publish pid:null)
                self.die_id_map[_p2] = old
                old = None
            else:
                del self.die_id_map[_p2]
        self.die_id_map[pid] = key

    def note_drawn(self, pid):
        if pid:
            self.ai_drawn_pids.add(str(pid))
