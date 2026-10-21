"""Spectator mode — full-game corpus ingestion (tier 1 of the USER's
corpus-replay validation, 2026-08-19: "fix one human decisions = AI
... run the entire thing through the harness and see it all works").

The driver plays NEITHER seat: every window from BOTH sides flows
parse -> segment -> match -> engine-apply, with the game's own
recorded randomness fed into the funnels per attempt (the RNG-only
forcing discipline of the corpus closed loop, applied to the LIVE
stack).  The input feed is REAL CLIENT OUTPUT — either a fresh AiFeed
capture of the client's native replay mode (I4-grade) or one of the
already-captured pairs feeds (offline breadth).  Nothing is injected
into a live game; the log is the client's own replay input.

This is a FINDINGS GENERATOR, not a pass/fail gate: every window the
harness cannot match/hold/apply cleanly becomes a census row.  The
census is the work queue for the remaining harness gaps, ordered by
what real games actually contain.
"""
import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, '..', '..', 'src'))
_BRIDGE = os.path.abspath(os.path.join(_HERE, '..'))
for _p in (_SRC, _BRIDGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from . import gesture_match
from . import live_parser
from .driver import DriverHalt, HandModel, LiveDriver


# _RandCtl is re-exported for test_live_spectator (it builds a ctl
# through this module)
from .driver import _RandCtl  # noqa: F401  (re-export)


class SpectatorDriver(LiveDriver):
    """Both seats external.  Overrides only what spectating changes;
    the base ingestion (parser, segmenter, undo ledger, turn-start
    protocol) runs untouched."""

    def __init__(self, engine_state, feed_path, randomness):
        import game_state as gs
        LiveDriver.__init__(self, engine_state,
                            gs.Side.SHADOW,        # placeholder; unused
                            feed_path, randomness=randomness,
                            nick='<spectator>')
        self.spectator = True
        self.hand_models = {
            gs.Side.FREE_PEOPLES: HandModel(engine_state,
                                            gs.Side.FREE_PEOPLES),
            gs.Side.SHADOW: HandModel(engine_state, gs.Side.SHADOW),
        }
        # PER-TURN RESYNC (USER 2026-08-19: "resync every turn, so we
        # can narrow down where divergences occur and what is broken
        # vs messy piece movement"): a Mirror — the corpus-proven
        # client-vocabulary board model — runs alongside on the SAME
        # deduped record stream; every Turn marker diffs engine vs
        # mirror and the rows localize the first broken turn.  Undo
        # events rebuild the mirror from the settled record list (the
        # Mirror has no unapply; undos are rare, rebuilds are cheap).
        import rendered_replay as rr
        self._rr = rr
        self.mirror = rr.Mirror()
        self.mirror_recs = []       # settled records the mirror holds
        self.turn_reports = []      # [(turn_no, rows)]
        self.first_divergent_turn = None

    # ---- the mirror shadow + per-turn resync ---------------------------
    def _route(self, ev, payload):
        """Mirror-track + per-turn resync, then the BASE routing shell
        (defer/buffer head + rng-retry tail live in LiveDriver._route;
        the spectator's own copy of that head is deleted — WAVE-G)."""
        self._mirror_track(ev, payload)
        if ev == 'phase' and payload.get('op') == 'turn':
            self._turn_resync(payload.get('turn'))
        return LiveDriver._route(self, ev, payload)

    def _pending_buffer_applies(self, dp):
        """BOTH seats are external in spectator mode: every open
        pending resolves from the feed (the base restricts buffering
        to the human side's pendings)."""
        return True

    _FEL_TOKENS = ('the fellowship', 'fsp', 'corruption')

    @classmethod
    def _mirror_alias_rec(cls, rec):
        """ARMY pieces parked on Mordor-track boxes physically sit on
        GORGOROTH (the matcher's zone_key alias); the Mirror only
        aliases the FELLOWSHIP's mordor zones, so armies staged there
        vanished from its board and every later turn resynced dirty
        (games 2023-01-18/-25/-31: turn-1 'Barad-Dur to Mordor 5 -
        The Crack of Doom' parks).  Rewrite a COPY for the mirror;
        fellowship records keep their mordor zones (fel_mordor
        tracking feeds on them)."""
        op = rec.get('op')
        obj = str(rec.get('obj', '') or '').strip().lower()
        if op == 'move' and obj not in cls._FEL_TOKENS:
            out = None
            for fld in ('from', 'to'):
                z = str(rec.get(fld, '') or '')
                if z.strip().lower().startswith('mordor'):
                    if out is None:
                        out = dict(rec)
                    out[fld] = 'Gorgoroth'
            return out or rec
        if op == 'kill':
            z = str(rec.get('zone', '') or '')
            if z.strip().lower().startswith('mordor') and not any(
                    str(p).strip().lower() in cls._FEL_TOKENS
                    for p in rec.get('pieces', [])):
                out = dict(rec)
                out['zone'] = 'Gorgoroth'
                return out
        return rec

    def _mirror_apply(self, rec):
        if rec.get('op') in ('move', 'kill', 'flip'):
            rec = self._mirror_alias_rec(rec)
            try:
                self.mirror.apply(rec)
            except Exception as e:
                self.findings.append(('mirror_apply', '%r %r'
                                      % (e, rec.get('norm', rec))))
            self.mirror_recs.append(rec)

    def _mirror_pop_tail(self, norm):
        if self.mirror_recs \
                and self.mirror_recs[-1].get('norm') == norm:
            self.mirror_recs.pop()
            self.mirror = self._rr.Mirror()
            for r in self.mirror_recs:
                try:
                    self.mirror.apply(r)
                except Exception:
                    pass

    def _mirror_track(self, ev, payload):
        if ev == 'phase':
            self._mirror_apply(payload)
        elif ev == 'open':
            self._mirror_apply(payload.records[0])
        elif ev in ('update', 'closed'):
            self._mirror_apply(payload.records[-1])
        elif ev == 'withdrawn':
            # an open window's tail record was undone pre-close: the
            # mirror applied it on open/update — take it back
            self._mirror_pop_tail(str(payload.get('norm', '')))
        elif ev == 'undo':
            inner = str(payload.get('inner', '')).strip()
            _k, norm = live_parser.normalize(inner)
            self._mirror_pop_tail(norm)
            # non-tail undos: the driver's own ledger alarms; the
            # mirror keeps its state (a divergence row will localize)

    @staticmethod
    def _engine_army_rows(st):
        from board import sanitize
        rows = {}
        for k, a in st.armies.items():
            cnt = Counter()
            for u in a.units:
                cnt[(str(u.nation.name).upper(),
                     str(u.kind.name).upper())] += 1
            nl = len(getattr(a, 'leaders', []))
            if cnt or nl:
                rows[sanitize(k)] = (cnt, nl)
        return rows

    @staticmethod
    def _mirror_army_rows(snap):
        rows = {}
        for reg, (units, leaders) in snap['armies'].items():
            # the mirror keys client zone text; canonicalize the
            # definite-article variants ('the grey havens') so a
            # naming difference can never masquerade as a divergence.
            # MERGE on collision — the mirror may hold the setup
            # garrison under the canonical key and an arrival under
            # the article variant (overwrite erased the garrison and
            # forged three divergence rows, game 2023-01-13 T3)
            reg = gesture_match.region_lookup(reg) or reg
            cnt, nl = rows.get(reg, (Counter(), 0))
            cnt = Counter(cnt)
            for nat, kind in units:
                cnt[(str(nat).upper(), str(kind).upper())] += 1
            rows[reg] = (cnt, nl + len(leaders))
        return rows

    def _turn_resync(self, turn_no):
        """Engine vs mirror at a turn boundary.  Row kinds:
        'count'      army unit-count mismatch      (REAL breakage)
        'leaders'    leader-count mismatch         (real or named-vs-
                                                    generic messiness)
        'fellowship' corruption/hidden             (real)
        'marker_cosmetic'  FS-PROGRESS-only mismatch: the client's FSP
                     marker is BOOKKEEPING (players leave it behind
                     after a reveal — game 2023-01-22 left it at 5);
                     the pawn/corruption/hidden rows carry the board
                     truth.  Reported, but NOT counted into
                     first_divergent_turn (USER's broken-vs-messy
                     directive: fdt reflects BREAKAGE).
        'political'  track/active mismatch         (real)
        first_divergent_turn counts NON-cosmetic rows only.
        """
        rows = []
        snap = self.mirror.snapshot()
        e = self._engine_army_rows(self.es)
        m = self._mirror_army_rows(snap)
        for reg in sorted(set(e) | set(m)):
            ec, el = e.get(reg, (Counter(), 0))
            mc, ml = m.get(reg, (Counter(), 0))
            for key in set(ec) | set(mc):
                if ec.get(key, 0) != mc.get(key, 0):
                    rows.append(('count', '%s %s: engine=%d mirror=%d'
                                 % (reg, key, ec.get(key, 0),
                                    mc.get(key, 0))))
            if el != ml:
                rows.append(('leaders', '%s: engine=%d mirror=%d'
                             % (reg, el, ml)))
        fel = snap['fel']
        fs = self.es.fellowship
        for name, ev_, mv in (
                ('progress', fs.progress, fel['progress']),
                ('corruption', fs.corruption, fel['corruption']),
                ('hidden', bool(fs.hidden), bool(fel['hidden']))):
            if ev_ != mv:
                kind = ('marker_cosmetic' if name == 'progress'
                        else 'fellowship')
                rows.append((kind, '%s: engine=%r mirror=%r'
                             % (name, ev_, mv)))
        self.turn_reports.append((turn_no, rows))
        real = [r for r in rows if r[0] != 'marker_cosmetic']
        self.cosmetic_rows = (getattr(self, 'cosmetic_rows', 0)
                              + (len(rows) - len(real)))
        if real and self.first_divergent_turn is None:
            self.first_divergent_turn = turn_no

    # ---- both seats draw --------------------------------------------
    def _on_draw(self, rec):
        import game_logic as gl
        import game_state as gs
        what = str(rec.get('what', '')).lower()
        side = (gs.Side.FREE_PEOPLES if what.startswith('free')
                else gs.Side.SHADOW if what.startswith('shadow')
                else None)
        if side is None:
            # non-deck phase draws are RANDOMNESS the engine will ask
            # for (hunt tiles after a fellowship move, random
            # companions): bank them for the funnels
            self._feed_draw_rec(rec)
            return []
        deck = 'character' if 'character' in what else 'strategy'
        key = (side, deck)
        n = self.turn_draws.get(key, 0)
        owed = (n == 0
                or gl.settle_owed_draw(self.es, side, deck))
        if not owed:
            self.findings.append(('unowed_draw', '%s %s (n=%d)'
                                  % (side.name, deck, n)))
            return []
        self.turn_draws[key] = n + 1
        # ONE roller-nick answerer (driver._bind_roller_nick): a nick
        # drawing for BOTH sides is poisoned to untagged.  A one-client
        # stream stamps the local nick on every draw and every '<game>
        # X rolls' line; a last-writer binding here side-tags every
        # later roll to the prefix's last drawing side and the other
        # side's combat roll starves forever (D1 DEFERRAL WEDGE).
        self._bind_roller_nick(rec.get('player'), side)
        self._snapshot([rec.get('norm', '')], 'draw:%s' % side.name)
        try:
            self.hand_models[side].human_draw(deck)
        except DriverHalt as e:
            self.findings.append(('draw_halt', str(e)))
        # D4: the hand-limit discard is an ENGINE pending — open it so
        # the discard gesture that follows the over-draw resolves
        # through the pending buffer (measured: the un-opened pending
        # left the 'kills ... in Shadow Hand' record stranded in the
        # declare buffer)
        gl.open_hand_limit_discard(self.es)
        if self.phase == 'draws' and all(
                self.turn_draws.get((s, d))
                for s in self.hand_models
                for d in ('character', 'strategy')):
            self.phase = 'declare'
            # arm the engine's own declaration opportunity (D2: the
            # Fellowship phase) — declare gestures resolve through the
            # pending buffer; silence resolves as the 'pass' decline
            # when the game demonstrably moves on (eyes/roll/window)
            self.es._declare_phase = True
        return []

    # ---- no AI seat ---------------------------------------------------
    def _ai_turn(self):
        return []

    def _on_eye(self, rec):
        # both seats are external: EVERY 'allocates an eye.' is the
        # recorded Shadow player's Phase-3 allocation.  Eye allocation
        # follows the declaration opportunity (D3 order): an armed
        # declare with no gesture settles as decline first.
        self._settle_declare_decline('eye allocation began')
        if self.phase in ('draws', 'declare', 'eyes'):
            self.phase = 'eyes'
        if rec.get('op') == 'alloc_eye':
            if not self._allocate_one_eye():
                self.findings.append(('eye_alloc_failed',
                                      str(rec.get('raw', ''))[:80]))
        return []

    def _on_action_roll(self, rec):
        self._settle_declare_decline('action roll began')
        return LiveDriver._on_action_roll(self, rec)


    def _on_recover(self, rec):
        """Spectator rollover: unlike the live driver (which must not
        grow a roster it has no physical die drag for — the J1 lesson)
        the spectator's feed IS the client's own piece movement, so
        the engine grows its bonus dice itself (Saruman/WK/Mouth):
        without this the turn-2 roll carried 7 faces onto a 6-die
        roster and the A-die windows matched nothing."""
        import ai
        import game_logic as gl
        if self.game_over is not None:
            return []
        if self.deferred is not None:
            self._retry_blocked()
        if self.deferred is not None:
            self._expire_deferred('unresolved at rollover')
        self._flush_deferred_tail()
        if ai.describe_pending(self.es) is not None:
            # the turn is ending: whatever pending is still open gets
            # its silence-resolution or an attributed expiry NOW,
            # never a ride across the boundary
            self._try_pending()
        if ai.describe_pending(self.es) is not None:
            self._expire_pending('unresolved at rollover')
        self._snapshot([rec.get('norm', '')], 'rollover')
        ended = gl.begin_turn_reset(self.es, self.es.turn + 1,
                                    deal_cards=False, bonus_dice=True)
        if ended:
            self.game_over = gl.check_all_victory_conditions(self.es)
            return [('note', 'game over: %s (%s)' %
                     (self.game_over['victory'], self.game_over['victory_type']))]
        self.phase = 'draws'
        self.turn_draws = {}
        return []

    def _on_chat(self, rec):
        from . import chat as chat_mod
        cmd = chat_mod.classify_command(rec.get('body', ''))
        kind = cmd[0] if cmd else None
        if kind == 'fp_ok' and self.phase == 'declare':
            self._settle_declare_decline("'fp ok'")
            self.phase = 'eyes'
            return []
        # decision answers may belong to EITHER seat's open pending
        import ai
        dp = ai.describe_pending(self.es)
        if dp is not None:
            act = chat_mod.match_choice(rec.get('body', ''),
                                        dp['actions'])
            if act is not None:
                try:
                    return self._commit_human(act, via='chat')
                except DriverHalt as e:
                    self.findings.append(('chat_apply_halt', str(e)))
        return []

    # ---- windows: either seat, rng-fed matching -----------------------
    def _retry_ring_waiver_pick(self, w, side):
        """ENGINE GAP (reported, engine FROZEN): the A7 ring waiver is
        honored by ai._consume_die_for_face but NOT by game_logic.
        _pick_die, so executors that call _pick_die directly
        (muster_minion: game_logic.py:4198) refuse the very action the
        menu offered on a ring-waived die (game 2023-01-28 T1: Shadow
        spent the captured Narya to muster Saruman on a P die — menu
        yes, apply 'Selected die cannot be used').  ONE retry with
        _pick_die wrapped to honor the waiver EXACTLY as ai.py:10816-
        10824 does; census row keeps the engine debt.  Fix owed:
        one waiver-aware die consumer."""
        import game_logic as gl
        orig = gl._pick_die
        uses = []

        def patched(state, side_, wanted, die_idx):
            try:
                return orig(state, side_, wanted, die_idx)
            except RuntimeError:
                dice = state.dice.get(side_, [])
                if die_idx is not None and 0 <= die_idx < len(dice):
                    d = dice[die_idx]
                    if (not d.used
                            and gl.ring_waiver_active(state, side_)
                            and gl.ring_waiver_permits(side_,
                                                       d.current)):
                        d.used = True
                        gl.ring_waiver_spend(state, side_)
                        uses.append(die_idx)
                        return d
                raise
        n_matched = self.window_stats['matched']
        gl._pick_die = patched
        try:
            self._match_window_now(w, side)
        finally:
            gl._pick_die = orig
        if self.window_stats['matched'] > n_matched:
            self.findings.append((
                'engine_gap_ring_waiver_pick_die',
                '%s die=%s waiver-consumed %r (shim-committed; engine '
                'fix owed: one waiver-aware die consumer)'
                % (side.name, w.die_face, uses)))
        return True

    def _on_window(self, w):
        import ai
        import game_state as gs
        side = (gs.Side.SHADOW if w.die_side == 'SHADOW'
                else gs.Side.FREE_PEOPLES)
        # a NEW window closing means the client moved on: anything
        # still blocked is now attributable — expire it visibly
        if self.deferred is not None:
            self._retry_blocked()
        if self.deferred is not None:
            self._expire_deferred('rng never arrived before the next '
                                  'window')
            self._flush_deferred_tail()
        if ai.describe_pending(self.es) is not None:
            self._try_pending()
        if ai.describe_pending(self.es) is not None:
            self._expire_pending('unresolved at next window')
        # an expiry that RESOLVED the pending (silent no-op, forced
        # discard) unblocks this window - hold only when one is still
        # genuinely open
        if ai.describe_pending(self.es) is not None:
            w_norms = [r.get('norm', '') for r in w.records]
            self.uncommitted.extend(w_norms)
            self.window_stats['hold'] += 1
            self.findings.append(('window_while_pending',
                                  '%s die=%s :: %s'
                                  % (side.name, w.die_face,
                                     (w_norms[1] if len(w_norms) > 1
                                      else w_norms)[:100])))
            return []
        # card plays name their titles: reveal-swap the hand
        # placeholders BEFORE matching (L4 wiring).  Combat cards
        # chosen through the client chooser reveal into WHICHEVER hand
        # the chosen-zone names (the defender picks inside the
        # attacker's window).
        pre_facts = gesture_match.window_facts(w)
        w._reveals = getattr(w, '_reveals', [])
        for title in pre_facts.cards_played:
            cid = self.hand_models[side].reveal(title)
            if cid is None:
                self.findings.append(('reveal_failed', repr(title)))
            else:
                w._reveals.append((side, cid))
        for tag, title in pre_facts.combat_cards:
            cc_side = (gs.Side.FREE_PEOPLES if tag == 'FP'
                       else gs.Side.SHADOW)
            cid = self.hand_models[cc_side].reveal(title)
            if cid is None:
                self.findings.append(('reveal_failed',
                                      'combat %r' % (title,)))
            else:
                w._reveals.append((cc_side, cid))
        self._feed_window_rng(w)
        return self._match_window_now(w, side)

    def _match_window_now(self, w, side):
        res = gesture_match.match_window(self.es, side, w,
                                         rng=self.rctl)
        w_norms = [r.get('norm', '') for r in w.records]
        if res.status == 'match':
            if self._commit_chain(res.chain, w_norms,
                                  'window:%s' % side.name):
                self.window_stats['matched'] += 1
            else:
                self.window_stats['hold'] += 1
                self._unreveal(w)
            return []
        if res.status == 'hold' and res.starved:
            w._starve_note = res.note
            note = res.note or ''
            tried = getattr(w, '_shims_tried', set())
            w._shims_tried = tried
            # DEFER only for randomness that genuinely renders AFTER
            # the window terminator (hunt rolls/tiles on fellowship
            # moves, random companions, random discards).  Battle
            # randomness renders INSIDE its window in every specimen —
            # a combat-roll starve is some candidate probing a battle
            # that is not this gesture, and deferring on it turned
            # plain non-matches into deferral noise on every army-move
            # window near a siege.  The HUNT-ROLL kind gets the same
            # precision: a die_killed window with NO fellowship
            # content can never be awaiting its own hunt roll - the
            # starving candidates are fellowship-move probes, not this
            # gesture.
            facts = gesture_match.window_facts(w)
            fs_windowed = bool(facts.fel_moves or facts.fel_flips
                               or facts.fs_progress)
            if w.closed == 'die_to_hunt' \
                    or ('d6:hunt roll' in note and fs_windowed) \
                    or 'tile' in note or 'randomcomp' in note \
                    or 'card_pick' in note:
                self.deferred = (w, side)
                return []
            if 'draws:' in note and 'draws' not in tried:
                tried.add('draws')
                self._retry_skip_draw_grants(w, side)
                return []
            if 'reroll' in note and 'reroll' not in tried:
                # decline-equivalent shim, same as the expiry path
                tried.add('reroll')
                n_matched = self.window_stats['matched']
                self.rand.reroll_decline_shim = True
                try:
                    self._match_window_now(w, side)
                finally:
                    self.rand.reroll_decline_shim = False
                if self.window_stats['matched'] > n_matched:
                    self.findings.append((
                        'engine_gap_reroll_declined',
                        '%s die=%s (shim-committed; engine fix owed: '
                        'optional re-roll as a learned decision) :: %s'
                        % (side.name, w.die_face,
                           str(w_norms[1] if len(w_norms) > 1
                               else w_norms)[:100])))
                return []
            # battle-internal starve with no shim left: a non-match
            res = gesture_match.MatchResult(
                'none', applies=res.applies, note=res.note)
        if res.status == 'none':
            tried = getattr(w, '_shims_tried', set())
            w._shims_tried = tried
            facts = gesture_match.window_facts(w)
            if facts.ring_moves and 'ring_pick' not in tried:
                tried.add('ring_pick')
                if self._retry_ring_waiver_pick(w, side):
                    return []
        if res.status == 'none':
            fb = self._relaxed_draw_match(w, side, w_norms)
            if fb is not None:
                return fb
            fb = self._relaxed_nation_match(w, side, w_norms)
            if fb is not None:
                return fb
            fb = self._pullback_match(w, side, w_norms)
            if fb is not None:
                return fb
            self._name_illegal_muster(w, side)
        self.uncommitted.extend(w_norms)
        self.window_stats[res.status if res.status in
                          self.window_stats else 'hold'] += 1
        self.findings.append((res.status,
                              '%s die=%s%s n_recs=%d :: %s'
                              % (side.name, w.die_face,
                                 '(' + (res.note or '') + ')',
                                 len(w.records),
                                 (w_norms[1] if len(w_norms) > 1
                                  else w_norms)[:100])))
        self._unreveal(w)
        self.rand.reset_feed()
        return []

    def _relaxed_draw_match(self, w, side, w_norms):
        """Client-permitted misplay lane: the window rendered MORE deck
        draws than any legal chain grants (proven real on game
        2023-01-13's raw log: two Controls.509 draws, distinct pids,
        on 'draw one Strategy card' plays).  Commit the rules-LEGAL
        subset and put the surplus on the books by name.  Returns the
        route result on commit, None to fall through to the normal
        failure bookkeeping."""
        facts = gesture_match.window_facts(w)
        if not facts.draws:
            return None
        res = gesture_match.match_window(self.es, side, w,
                                         rng=self.rctl,
                                         relax_draws=True)
        if res.status != 'match':
            return None
        engd = gesture_match.delta_facts(self.es, res.post).draws
        extra = {k: v - engd.get(k, 0)
                 for k, v in facts.draws.items()
                 if v - engd.get(k, 0)}
        if self._commit_chain(res.chain, w_norms,
                              'window:%s(relaxed-draws)' % side.name):
            self.window_stats['matched'] += 1
            self.findings.append((
                'client_unowed_draw',
                '%s die=%s drew %r beyond the effect grant '
                '(engine committed the legal subset)'
                % (side.name, w.die_face, extra)))
        else:
            self.window_stats['hold'] += 1
            self._unreveal(w)
        return []

    def _relaxed_nation_match(self, w, side, w_norms):
        """ENGINE GAP fallback lane (reported, engine FROZEN):
        game_logic.move_army's split take() picks units in LIST ORDER
        and carries no per-nation composition, so a mixed-nation drag
        (1 Isengard + 1 Sauron regular from Moria, game 2023-01-18 T2)
        is UNREACHABLE from the menu.  Retry nation-blind; on match
        COMMIT the closest legal composition and put the composition
        residue on the books by name (the client_unowed_draw
        precedent: commit the legal core, name the divergence).  Fix
        owed: composition-aware move_army split."""
        facts = gesture_match.window_facts(w)
        nations = set()
        for (_reg, t), v in facts.armies.items():
            if v:
                nations.add(t.split(':', 1)[0])
        if len(nations) < 2:
            return None         # single-nation rows always reachable
        res = gesture_match.match_window(self.es, side, w,
                                         rng=self.rctl,
                                         relax_nations=True)
        if res.status != 'match':
            return None
        eng = gesture_match.delta_facts(self.es, res.post)
        diff = {}
        for k in set(facts.armies) | set(eng.armies):
            d = facts.armies.get(k, 0) - eng.armies.get(k, 0)
            if d:
                diff[k] = d
        if self._commit_chain(res.chain, w_norms,
                              'window:%s(relaxed-nations)'
                              % side.name):
            self.window_stats['matched'] += 1
            self.findings.append((
                'engine_gap_army_composition',
                '%s die=%s composition residue %r (shim-committed '
                'closest legal composition; engine fix owed: '
                'per-nation counts in move_army split)'
                % (side.name, w.die_face, diff)))
        else:
            self.window_stats['hold'] += 1
            self._unreveal(w)
        return []

    def _pullback_match(self, w, side, w_norms):
        """ENGINE GAP fallback lane (reported, engine FROZEN): after a
        defender retreat the engine AUTO-advances the WHOLE attacking
        army (no partial-advance pending; victor_advance after a
        fought battle does offer counts), but p.31 lets the attacker
        advance any part — the client renders the stay-behind as a
        full group move plus a smaller PULL-BACK drag of the reverse
        pair (game 2023-01-28 T2: Minas Morgul -> North Ithilien, one
        Sauron Regular back).  STRICT matching already failed (a
        legitimate two-half Army-die shuttle would have matched);
        retry with the pull-back record removed, commit the engine's
        full-advance state, and put the stay-behind residue on the
        books by name.  Fix owed: partial advance after retreat via
        the victor_advance menu."""
        import copy
        pairs = {}
        for r in w.body_records():
            if r.get('op') != 'move':
                continue
            s = gesture_match.zone_key(r.get('from'))
            d = gesture_match.zone_key(r.get('to'))
            if s[0] == 'region' and d[0] == 'region':
                pairs.setdefault((s[1], d[1]), []).append(r)

        def _npieces(rec):
            import rendered_replay as rr
            obj = str(rec.get('obj', '') or '')
            toks = (rr.split_group(obj[len('group ('):-1])
                    if obj.startswith('group (') and obj.endswith(')')
                    else [obj])
            n = 0
            for t in toks:
                pp = rr.parse_piece(t)
                n += pp[3] if pp is not None else 1
            return n
        for (s, d), rl in pairs.items():
            fwd = pairs.get((d, s))
            if not fwd or len(rl) != 1:
                continue
            back, n_back = rl[0], _npieces(rl[0])
            n_fwd = sum(_npieces(r) for r in fwd)
            if n_back >= n_fwd:
                continue        # not a pull-back shape
            w2 = copy.copy(w)
            w2.records = [r for r in w.records if r is not back]
            res = gesture_match.match_window(self.es, side, w2,
                                             rng=self.rctl)
            if res.status != 'match':
                continue
            if self._commit_chain(res.chain, w_norms,
                                  'window:%s(pullback)' % side.name):
                self.window_stats['matched'] += 1
                self.findings.append((
                    'engine_gap_partial_advance',
                    '%s die=%s stay-behind %r (%s -> %s pull-back; '
                    'engine committed the full advance; fix owed: '
                    'partial advance after retreat)'
                    % (side.name, w.die_face,
                       str(back.get('obj', ''))[:60], s, d)))
            else:
                self.window_stats['hold'] += 1
                self._unreveal(w)
            return []
        return None

    def _name_illegal_muster(self, w, side):
        """NAME the rules-illegal client recruit (no commit is honest:
        there is no legal subset — the die was spent on the illegal
        act).  Specimen: game 2023-01-31 T2, Elven Elite recruited at
        Woodland Realm with Elves one step short of At War (clause
        CL-084/p.26: recruiting requires At War; VASSAL enforces
        nothing)."""
        import game_state as gs
        import rendered_replay as rr
        facts = gesture_match.window_facts(w)
        if facts.cards_played or facts.combat_cards or facts.draws:
            # a CARD window: card effects place units regardless of
            # At-War (Thranduil's Archers class) — the window failed
            # for some other reason; naming it a dice-muster misplay
            # would be a false accusation
            return
        rows = []
        for rec in w.body_records():
            if rec.get('op') != 'move':
                continue
            src = gesture_match.zone_key(rec.get('from'))
            dst = gesture_match.zone_key(rec.get('to'))
            if src[0] != 'reinf' or dst[0] != 'region':
                continue
            obj = str(rec.get('obj', '') or '')
            toks = (rr.split_group(obj[len('group ('):-1])
                    if obj.startswith('group (') and obj.endswith(')')
                    else [obj])
            for t in toks:
                pp = rr.parse_piece(t)
                if pp is None or pp[0] != 'unit':
                    continue
                nat = gesture_match._nation_enum(pp[1])
                if nat is None:
                    continue
                if self.es.political.boxes.get(nat, 0) < 3:
                    rows.append((getattr(nat, 'value', nat),
                                 dst[1], t))
        if rows:
            caveat = ''
            if self.first_divergent_turn is not None:
                # engine political may lag behind the client once the
                # boards have parted — the naming is then evidence,
                # not a verdict
                caveat = (' [post-divergence turn: engine political '
                          'may lag the client]')
            self.findings.append((
                'client_illegal_muster',
                '%s die=%s recruit(s) for not-At-War nation(s) %r - '
                'rules-illegal (p.26/CL-084), nothing committed%s'
                % (side.name, w.die_face, rows, caveat)))

    # ---- pending phase-buffer + blocked-item retry ----------------------
    def _on_undo(self, ev, rec):
        """Head kept for the PENDING BUFFER only: a buffered-but-
        uncommitted pending record undone by the client dissolves,
        exactly like the segmenter's window tail-pop (game 2023-01-22:
        an undone FSP reveal flip stayed in the hunt buffer and doubled
        the track reset).  The DEFERRED-window head is the base's now
        (WAVE-E hoist; shadow deleted WAVE-G)."""
        if ev == 'undo' and self.pending_buf is not None \
                and self.pending_buf.records:
            inner = str(rec.get('inner', '')).strip()
            _k, norm = live_parser.normalize(inner)
            if str(self.pending_buf.records[-1].get('norm', '')) \
                    == norm:
                self.pending_buf.records.pop()
                if not self.pending_buf.records:
                    self.pending_buf = None
                return []
        return LiveDriver._on_undo(self, ev, rec)

    # ---- report --------------------------------------------------------
    def census(self):
        if self.deferred is not None:
            self._expire_deferred('feed ended')
        return {'windows': dict(self.window_stats),
                'pending': dict(self.pending_stats),
                'findings': list(self.findings),
                'alarms': list(self.alarms),
                'unknown_lines': list(self.unknown_lines),
                'turn': getattr(self.es, 'turn', None),
                'commits': len(self.commits),
                'turn_reports': [(t, len([x for x in r
                                          if x[0] != 'marker_cosmetic'
                                          ]), r[:6])
                                 for t, r in self.turn_reports],
                'cosmetic_rows': getattr(self, 'cosmetic_rows', 0),
                'first_divergent_turn': self.first_divergent_turn}


def ingest_feed_file(path, max_windows=None, seed=1234):
    """Run one captured feed file (real client output) through the
    spectator driver.  Returns the census dict."""
    import board as board_mod
    import game_logic
    import game_state as gs
    import random as _random
    from . import randomness as rnd_mod
    _random.seed(seed)
    st = gs.setup_initial_game(board_mod.load_board())
    game_logic.initialize_decks(st)
    rand = rnd_mod.ClientRandomness(seed=seed)
    rand.install()
    drv = SpectatorDriver(st, path, rand)
    done = 0
    while True:
        try:
            new = drv.feed.poll()
        except Exception as e:
            drv.findings.append(('feed_error', repr(e)))
            break
        if not new:
            break
        for raw in new:
            rec = live_parser.parse(raw)
            if rec.get('unknown'):
                drv.unknown_lines.append(raw)
            for ev, payload in drv.seg.push(rec):
                try:
                    drv._route(ev, payload)
                except DriverHalt as e:
                    drv.findings.append(('halt', str(e)))
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as e:
                    # a NeedClientCall (BaseException) or engine error
                    # leaking out of routing is a CENSUS ROW, never a
                    # crash of the whole soak (game 2023-01-13 died on
                    # an escaped draws-starve)
                    drv.findings.append(('route_error', '%s %r'
                                         % (ev, e)))
                    try:
                        drv.rand.reset_feed()
                    except Exception:
                        pass
                if ev == 'closed':
                    done += 1
                    if max_windows and done >= max_windows:
                        return drv.census()
    return drv.census()
