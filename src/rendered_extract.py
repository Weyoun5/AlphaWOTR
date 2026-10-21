"""RENDERED-LOG BC extraction (v27) â€” engine-driven, low-false-positive.

Sources every decision from the CLIENT-RENDERED replay (the authoritative
gesture resolution) via rendered_replay.parse_game: a per-action-die-batch
DECISION + RNG stream plus a client-truth MIRROR snapshot (closed_loop
_board_snapshot schema) for every batch.  For each batch the REAL engine is
resync'd to the batch's rendered truth-BEFORE (rendered_census
._state_from_snapshot) and driven so that every kept decision is LEGALITY-
MATCHED against ai.get_legal_actions â€” bookkeeping batches matched, resolver
batches (hunt/combat) driven through game_logic under forced RNG
(rendered_census._drive_hunt / _drive_combat).  Because each batch is
independently resync'd to the client-authoritative mirror, an engine
divergence in one batch NEVER contaminates a later batch (no engine state is
carried forward) â€” divergent batches only drop their own samples.

USER directive: PURITY over recall.  When in doubt EXCLUDE; never patch/guess.
Every admission gate is per-game; any failure quarantines the whole game with
a NAMED reason (quarantine.jsonl in OUT_DIR).

Reuse-by-import (never edited): ai.encode_state / encode_action /
get_legal_actions / _get_action_region_index; extract_corpus.pack_npz /
corpus_rows / stem_of; game_logic; board; rendered_replay; rendered_census.

Artifacts (resumable â€” games with an existing .npz or .error.json skip):
  <OUT_DIR>/<stem>.npz              per-game ragged sample pack
  <OUT_DIR>/manifest-shard<I>.jsonl one JSON line per game
  <OUT_DIR>/quarantine.jsonl        one line per quarantined game
  <OUT_DIR>/extraction_progress.md  driver heartbeat
  (OUT_DIR default: data/bc_corpus_v37_rendered; override RX_OUT_DIR)

Usage:
  python src/rendered_extract.py --pilot 200        # first N ok corpus stems
  python src/rendered_extract.py --canary           # the 21 verdict games
  python src/rendered_extract.py --driver --workers 7
  python src/rendered_extract.py --shard 0 --nshards 7   (internal)
"""
import argparse
import csv
import gc
import json
import os
import re
import subprocess
import sys
import time
import traceback
from raw_log_reader import RawLogEncodingError, read_raw_log_text
from collections import Counter

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('PYTHONHASHSEED', '0')

DATA = os.path.join(ROOT, 'data')
# NOTE: data/bc_corpus_v27 is already occupied by the TEACHER's v27
# extraction (extract_corpus.py, ~172+ samples/game, its own shard
# manifests).  This rendered pipeline is a SEPARATE, per-die-batch,
# engine-driven build, so it defaults to its own directory to avoid
# colliding with (and being resume-skipped against) the teacher output.
# Override with RX_OUT_DIR.
OUT_DIR = os.environ.get('RX_OUT_DIR') or \
    os.path.join(DATA, 'bc_corpus_v37_rendered')
# Corpus location: explicit override, else the v1 dir if it still exists,
# else the v2 sweep output.
RLOG_CORPUS = os.environ.get('RX_RLOG_CORPUS') or (
    os.path.join(DATA, 'rendered_logs_corpus')
    if os.path.exists(os.path.join(DATA, 'rendered_logs_corpus'))
    else os.path.join(DATA, 'rendered_logs_corpus_v2'))
RLOG_CANARY = os.path.join(DATA, 'rendered_logs')
# v27 List-2 emitter master switch.  RX_NO_LIST2=1 reproduces the pre-v27
# behaviour EXACTLY (no eye_allocate/draw_card/elven_ring/census), so an A/B on
# identical inputs proves the additions leave the admit/quarantine verdict
# byte-identical (they touch no gate/ledger, only additive sample emission).
_LIST2 = not os.environ.get('RX_NO_LIST2')
# CLOSURE wave master switch (spec 1.216 final five).  RX_NO_CLOSURE=1
# reproduces the pre-closure behaviour EXACTLY (no table-card hunt interposes,
# no declaration emitter, onslaught magnitudes stay censused), so an A/B on
# identical inputs proves the closure additions leave every prior emit-kind
# count + the admit/quarantine/divergence verdict byte-identical.
_CLOSURE = _LIST2 and not os.environ.get('RX_NO_CLOSURE')
RAW_LOGS = os.path.join(DATA, 'game_logs')
CORPUS_MANIFEST = os.path.join(RLOG_CORPUS, 'manifest.csv')
PROGRESS = os.path.join(OUT_DIR, 'extraction_progress.md')

# --------------------------------------------------------------------------
# die letter -> engine face
# --------------------------------------------------------------------------
_LETTER2FACE = None


def _letter2face():
    global _LETTER2FACE
    if _LETTER2FACE is None:
        import game_state as gs
        F = gs.ActionDieFace
        _LETTER2FACE = {'C': F.CHARACTER, 'A': F.ARMY, 'M': F.MUSTER,
                        'P': F.EVENT, 'H': F.MUSTER_ARMY, 'E': F.EYE,
                        'W': F.WILL_OF_WEST}
    return _LETTER2FACE


# rendered "draws X" deck name -> VASSAL deck code (matches raw Game.200x)
_DECK_CODE = {'free character card': '2004', 'free strategy card': '2005',
              'shadow character card': '2006', 'shadow strategy card': '2007'}
# strict draw-TO-HAND pattern (the hand destination Controls.45/46 tail
# distinguishes a real card draw from other Controls.509 deck events)
_RAW_DRAW_RE = re.compile(r'<~Controls\.509~>(\d+)<~Controls\.510~>'
                          r'<~Game\.(200[4-7])~><~Controls\.511~>'
                          r'<~Controls\.(45|46)~>')
# rendered "draws <deck> Card." line (scanned directly â€” draws before the
# first die-staging of a turn never reach a batch's rng stream)
_RENDERED_DRAW_LINE_RE = re.compile(r'^\S+\s+draws\s+(.+?)\.\s*$')

# gate-2 whitelisted no-op "other"-op line templates (proven no board effect):
# viewing a hand and deselecting a piece are pure UI gestures.  'deselects a'
# matches the occasional truncated 'deselects a' line (no trailing 'card.').
_G2_NOOP_RE = re.compile(
    r'^\S+\s+(views\s+the\b|deselects\s+a\b)', re.I)
# gate-2 MODELED decision-bearing "other"-op classes (recognized, not unknown)
_G2_DIECHANGE_RE = re.compile(r'^\S+\s+changes\s+(Shadow|Free)\s+Action\s+Die')
_G2_RANDDISC_RE = re.compile(r'^\S+\s+randomly\s+discards\s+a\s+')

# shadow-side nations (pool-conservation ownership)
_SHADOW_NATS = {'SAURON', 'ISENGARD', 'SOUTHRON_EASTERLINGS'}

# authoritative physical figure inventory per nation (from
# piece_table_authoritative.AUTH_ARMY_PIECES) â€” the hard conservation
# ceiling: no more figures of a nation/kind can ever be ON the board than
# physically exist (killed figures leave the board; re-mustered ones return
# from reinforcements, so ON-BOARD alone is always <= the pool for legal
# play).  A parse error that places phantom figures or fails to remove
# killed ones breaks this.  Nazgul (8 shared Sauron figures) counted apart.
_NATION_POOL = {
    'DWARVES': {'REGULAR': 5, 'ELITE': 5, 'LEADER': 4},
    'ELVES': {'REGULAR': 5, 'ELITE': 10, 'LEADER': 4},
    'GONDOR': {'REGULAR': 15, 'ELITE': 5, 'LEADER': 4},
    'THE_NORTH': {'REGULAR': 10, 'ELITE': 5, 'LEADER': 4},
    'ROHAN': {'REGULAR': 10, 'ELITE': 5, 'LEADER': 4},
    'ISENGARD': {'REGULAR': 12, 'ELITE': 6, 'LEADER': 0},
    'SOUTHRON_EASTERLINGS': {'REGULAR': 24, 'ELITE': 6, 'LEADER': 0},
    'SAURON': {'REGULAR': 36, 'ELITE': 6, 'LEADER': 0},
}
_MAX_NAZGUL = 8


# ==========================================================================
# HAND-STATE LEDGER (v27 event-card-play enablement)
# --------------------------------------------------------------------------
# USER directive: "every single decision the AI makes has equivalent matching
# data in the logs".  Event-card PLAYS are the biggest unemitted class; to
# emit them the reconstructed engine state needs the TRUE per-side hand so the
# play_event menu is legal-matched (and hands enrich EVERY sample's encoding).
#
# The ledger reconstructs each side's hand purely from the logs:
#   * draws IN   -> raw .log password-decoded draw stream, paired 1-1 IN ORDER
#                   with the rendered generic 'draws <deck> Card' lines (gate3
#                   already proves that deck-code order is identical, so the
#                   per-deck FIFO assignment is exact).
#   * plays/OUT  -> rendered move records whose FROM zone is a Hand (the card
#                   TITLE names the identity); works for event plays (-> Current
#                   FP/SA Card), table plays (-> BoardN) and every other exit.
#   * named IN   -> rendered move records whose TO zone is a Hand (card returns).
#   * random OUT -> the rendered 'randomly discards a <deck> Card' line carries
#                   no identity; resolved via the raw forced-discard command
#                   (<~Controls.452~>PID ... <~Controls.45|46~> ... <~Game.2039~>)
#                   decoded through the SAME CardDecoder commitment brute-force.
#   * undo       -> a hand-affecting move that is undone is reversed in-ledger.
#
# The rendered '<player> views the <side> event hand.' lines are ledger sync
# ANCHORS (positions); the raw log carries NO viewed-hand contents (verified),
# so they cannot cross-check identity today -- a future #HAND-instrumented
# rendered stream (separate agent) will, and the hand SOURCE is pluggable:
# _HandLedger exposes hand_at(side, line); any alternate source implementing
# the same interface can replace it.
#
# Consistency gate: a named play that removes a card NOT in the reconstructed
# hand, or a draw-FIFO underflow, is a hard desync -> quarantine
# (gate3_hand_desync).  An UNRESOLVABLE random discard is a soft ambiguity ->
# the affected side's later play_event samples drop (purity), never a guess.
# ==========================================================================
import bisect as _bisect

# raw forced-(random)-discard: a hand PID sent to the discard pile (Game.2039)
_RAW_DISCARD_RE = re.compile(r'<~Controls\.452~>(\d+)<~Controls\.453~>'
                             r'<~Controls\.(45|46)~><~Controls\.454~>'
                             r'<~Game\.2039~>')
# raw voluntary EXCESS-hand discard (hand > 6 at end of draw phase): a hand PID
# selected out of hand 45/46 (Controls.296/297).  Its identity never renders
# (the rendered line is the generic 'kills <deck> Card in <side> Hand') and it
# is discarded FACE-DOWN (no reveal), so the commitment decode is the ONLY
# identity source.  The hand-marked command count matches the rendered generic
# hand-kill count exactly (verified), so a per-hand FIFO aligns 1-1 in order.
_RAW_EXCESS_RE = re.compile(r'<~Controls\.296~>(\d+)\s*<~Controls\.297~>'
                            r'<~Controls\.(45|46)~>')
# W1-003: raw hand->Casualties DRAG (a hand-limit discard made by dragging
# the card onto the FP/SA Casualties box, Game.2039/2041).  The rendered
# line is the identity-less 'moves <Side> <Deck> Card from <Side> Hand to
# <FP|SA> Casualties'; the raw command carries the hand PID, decodable
# through the same commitment route as the excess discard.  One raw
# command per rendered generic drag line, per hand (verified 1-1 on every
# affected card-cover game) -> a per-hand FIFO.  Face-DOWN (1.261).
_RAW_CAS_RE = re.compile(r'<~Controls\.281~>(\d+)\s*<~Controls\.282~>'
                         r'<~Controls\.(45|46)~><~Controls\.283~>'
                         r'<~Game\.(?:2039|2041)~>')
_CAS_ZONES = {'fp casualties', 'sa casualties'}
# hand zone spellings (rendered)
_HAND_ZONE_FP = 'free peoples hand'
_HAND_ZONE_SH = 'shadow hand'
_HAND_ZONES = {_HAND_ZONE_FP, _HAND_ZONE_SH}


def _zone_norm(zone):
    """Canonical lowercase zone name with any trailing parenthetical
    annotation stripped ('Shadow Hand (chosen)' -> 'shadow hand'): the
    VASSAL 'chooses a card' menu tags the hand zone that way (W1-001)."""
    return re.sub(r'\s*\([^)]*\)\s*$', '', str(zone).strip().lower())
# recognized event/table PLAY-AREA destinations (a hand-exit here == the die
# was spent buying the card; casualties/region/dice exits still leave the hand
# but are not treated as a die-decision play to emit)
_PLAY_AREA_DEST = {'current fp card', 'current sa card'}
_BOARD_DEST_RE = re.compile(r'^board\d*$')
_RENDERED_VIEW_RE = re.compile(
    r'^\S+\s+views\s+the\s+(free peoples|shadow)\s+event\s+hand', re.I)

_PW_CACHE = None
_PW_NORM = None           # normalized-stem -> (fp,sa) (space/underscore tolerant)
_RAW_LOG_INDEX = None     # normalized-stem -> raw .log path
_CARD_NAME_LOOKUP = None
_LEDGER = None            # module-global per-game ledger (single-thread worker)


def _norm_stem(s):
    """Collapse space/underscore runs + lowercase (rendered stems underscore a
    player name the raw .log / reports CSV keep spaced â€” 'The board is set')."""
    return re.sub(r'[ _]+', '_', str(s)).strip('_').lower()


def _passwords():
    """{stem: (fp_password, sa_password)} from the reports CSV (log_file
    basename -> stem).  Cached, plus a normalized-key index."""
    global _PW_CACHE, _PW_NORM
    if _PW_CACHE is None:
        _PW_CACHE = {}
        _PW_NORM = {}
        csvp = os.path.join(DATA, 'gameReports_with_passwords.csv')
        try:
            with open(csvp, encoding='utf-8') as f:
                for r in csv.DictReader(f):
                    fp = (r.get('free_password') or '').strip()
                    sa = (r.get('shadow_password') or '').strip()
                    if not (fp and sa):
                        continue
                    st = os.path.splitext(os.path.basename(
                        r.get('log_file', '')))[0]
                    if st and st not in _PW_CACHE:
                        _PW_CACHE[st] = (fp, sa)
                        _PW_NORM.setdefault(_norm_stem(st), (fp, sa))
        except Exception:
            pass
    return _PW_CACHE


def _password_for(stem):
    """Passwords for a stem, tolerating the rendered/raw space-vs-underscore
    player-name spelling (canary stems underscore; the CSV keeps spaces)."""
    pw = _passwords().get(stem)
    if pw is not None:
        return pw
    return (_PW_NORM or {}).get(_norm_stem(stem))


def _raw_log_path(stem):
    """Resolve a rendered stem to its raw .log path, tolerating the
    space-vs-underscore player-name spelling.  Returns None when no raw log
    exists (graceful â€” gate3/ledger degrade, never a false match: the unique
    timestamp prefix makes the normalized key collision-free).  Excludes
    _replaynav_clean derivatives."""
    p = os.path.join(RAW_LOGS, stem + '.log')
    if os.path.exists(p):
        return p
    global _RAW_LOG_INDEX
    if _RAW_LOG_INDEX is None:
        _RAW_LOG_INDEX = {}
        try:
            for fn in os.listdir(RAW_LOGS):
                if not fn.endswith('.log') or 'replaynav' in fn.lower():
                    continue
                _RAW_LOG_INDEX.setdefault(_norm_stem(fn[:-4]),
                                          os.path.join(RAW_LOGS, fn))
        except Exception:
            pass
    return _RAW_LOG_INDEX.get(_norm_stem(stem))


def _title_to_cid(title):
    """Rendered card TITLE ('Event/Combat' dual title) -> card_id, tolerant of
    the '/'-joined combat subtitle and accents/punctuation.  None on miss."""
    global _CARD_NAME_LOOKUP
    if not title:
        return None
    import ai
    if _CARD_NAME_LOOKUP is None:
        _CARD_NAME_LOOKUP = ai._build_card_name_lookup()
    cid = ai._lookup_card_id(_CARD_NAME_LOOKUP, title)
    if cid:
        return cid
    head = str(title).split('/')[0].strip()
    return ai._lookup_card_id(_CARD_NAME_LOOKUP, head) if head else None


class _HandLedger:
    """Per-side hand reconstruction with a line-indexed history.

    hand_at('fp'|'sh', line) -> ordered list of card_ids in that side's hand
    just BEFORE `line` (so a card played AT/after that line is still present).
    """

    def __init__(self):
        from rendered_history import DeckHistory
        self.decks = DeckHistory(_deck_catalogues())
        # running hands (ordered by acquisition), snapshotted on every change
        self._fp = []
        self._sh = []
        self._lines = []          # sorted change-line numbers
        self._snap_fp = []        # parallel: hand tuple after that line
        self._snap_sh = []
        # P0-a (DET census 2026-08-30 + USER hidden-info ruling
        # 2026-08-31 "discards from hand are still kept face down"):
        # line-indexed FACE-UP out-of-hand history.  ONLY the classes
        # the ENGINE files face-up enter here - cards PLAYED to a play
        # area / Board slot (event plays, revealed combat cards CL-217,
        # table plays).  Hand-limit discards, random/effect discards
        # and the EX-8 payment pair are client-MASKED and stay face-
        # down: the ledger KNOWS them (raw omniscience) but must not
        # leak them into any encoding.  out_at() feeds the state's
        # discard_face_up (mirror of the engine's replayed pile).
        self._out_fp = []
        self._out_sh = []
        self._snap_out_fp = []
        self._snap_out_sh = []
        # 1.261: each side's PRIVATE face-down discards (hand-limit /
        # random / effect discards, the EX-8 payment pair, WWSAT) -
        # installed into that side's hands[side].discarded_face_down so
        # the per-side encoding fill (public + OWN face-down) can light
        # the owner's spent cards without leaking them to the opponent.
        self._outfd_fp = []
        self._outfd_sh = []
        self._snap_outfd_fp = []
        self._snap_outfd_sh = []
        self.desync = None        # (reason, detail) -> hard quarantine
        # per-(side, deck-letter) UNKNOWN-removal ambiguity: an event removed a
        # card of that deck whose identity we could not resolve (excess/other
        # generic discard).  A phantom of that deck may linger in the known
        # list, so play_event of that deck DROPS from that point (purity).
        self.amb = {'fp': {'C': 0, 'S': 0}, 'sh': {'C': 0, 'S': 0}}
        # W1-002: the ambiguity is LINE-INDEXED.  A phantom of that deck
        # can only sit in the reconstructed hand from the unresolved
        # removal line until the named play that resolves it (or game
        # end) - a window [open_line, close_line).  Plays of that deck
        # outside every window are from a clean hand; the old game-wide
        # flag dropped them retroactively (2023-01-20 L77/409/535/578
        # for a kill at L733).
        self.amb_win = {'fp': {'C': [], 'S': []}, 'sh': {'C': [], 'S': []}}
        # slot memory: play-area / board slot zone -> the card_id sitting there
        # (so a GENERIC 'moves <deck> Card from Current X Card to Hand' return
        # re-adds the correct identity even without a title).
        self.slot = {}
        # every card_id that was EVER in each side's hand.  A named play of a
        # card that is ever-held but not present NOW is an undo/return artifact
        # (implicit undo of an 'auto> ... (chosen)' play, a slot return whose
        # line we skipped, ...) -> SOFT drop.  A play of a card that was NEVER
        # held is the genuine desync -> HARD quarantine.
        self.ever = {'fp': set(), 'sh': set()}
        # VOLUNTARY hand-discard decisions (spec 1.216 META: "the discarded
        # card identity is a critical decision" â€” NEVER auto/random).  Each
        # entry: (side_tag, line_no, card_id_or_None).  Random (forced)
        # discards are RNG, not decisions, and are NOT recorded here.
        self.discard_events = []
        # counters (manifest telemetry)
        self.n_draws = 0
        self.n_named_out = 0
        self.n_named_in = 0
        self.n_rand_disc = 0
        self.n_rand_unresolved = 0
        self.n_excess_disc = 0
        self.n_cas_disc = 0       # W1-003 honoured hand->Casualties drags
        self.n_disposal = 0       # W1-002 in-hand kills of a card already
        #                           played face-up (the play's disposal)
        self.n_undone_kills = 0   # W1-002 reverted 'undoes ... kills' lines
        self.n_undone_draws = 0   # W1-024 reverted 'undoes ... draws' lines
        self.n_soft_desync = 0
        self.n_views = 0
        self.enabled = True

    def _record(self, line_no):
        self._lines.append(line_no)
        self._snap_fp.append(tuple(self._fp))
        self._snap_sh.append(tuple(self._sh))
        self._snap_out_fp.append(tuple(self._out_fp))
        self._snap_out_sh.append(tuple(self._out_sh))
        self._snap_outfd_fp.append(tuple(self._outfd_fp))
        self._snap_outfd_sh.append(tuple(self._outfd_sh))

    def mark_deck_ambiguous(self, side_tag, deck_letter, line_no=None):
        self.amb[side_tag][deck_letter] += 1
        # W1-002: open a window at this line (0 = unknown line -> the
        # window covers the whole game, the old behaviour)
        self.amb_win[side_tag][deck_letter].append(
            [line_no if line_no is not None else 0, None])

    def unmark_deck_ambiguous(self, side_tag, deck_letter, line_no):
        """W1-002: an UNDO of the unresolved removal that opened the
        window at line_no - the phantom never entered the hand."""
        wins = self.amb_win[side_tag][deck_letter]
        for k in range(len(wins) - 1, -1, -1):
            if wins[k][0] == line_no and wins[k][1] is None:
                del wins[k]
                self.amb[side_tag][deck_letter] = max(
                    0, self.amb[side_tag][deck_letter] - 1)
                return True
        return False

    def _close_deck_ambiguity(self, side_tag, deck_letter, line_no):
        """W1-002: a named exit of an absent card of that deck resolves
        the EARLIEST open window (the phantom WAS this card) at line_no."""
        for w in self.amb_win[side_tag][deck_letter]:
            if w[1] is None:
                w[1] = line_no
                return

    def is_deck_ambiguous(self, side_tag, deck_letter):
        return self.amb[side_tag].get(deck_letter, 0) > 0

    def is_deck_ambiguous_at(self, side_tag, deck_letter, lo, hi=None):
        """W1-002: does an ambiguity window of (side, deck) intersect the
        line span [lo, hi]?  A phantom of that deck may then sit in the
        hand the span's samples see -> their play_event drops (purity).
        Plays outside every window are clean and EMIT."""
        if hi is None:
            hi = lo
        for w in self.amb_win[side_tag].get(deck_letter, []):
            if w[0] <= hi and (w[1] is None or w[1] > lo):
                return True
        return False

    def add(self, side_tag, cid, line_no):
        (self._fp if side_tag == 'fp' else self._sh).append(cid)
        if cid is not None:
            self.ever[side_tag].add(cid)
            # a card back in hand is no longer out (undo / slot return)
            for out in ((self._out_fp, self._outfd_fp)
                        if side_tag == 'fp'
                        else (self._out_sh, self._outfd_sh)):
                if cid in out:
                    out.remove(cid)
        self._record(line_no)

    def remove_named(self, side_tag, cid, line_no, deck_letter=None,
                     face_up=False):
        """A rendered NAMED hand-exit: the card must be present.  Absent but
        (a) EVER held (undo of an 'auto> (chosen)' play / an unlogged return),
        or (b) the relevant deck is already ambiguous (an earlier unresolved
        generic removal) -> SOFT drop, don't quarantine.  Absent AND never held
        -> HARD desync (a genuine reconstruction inconsistency).
        face_up=True (P0-a): the exit is a PLAY to a public zone - the one
        class that reaches the face-up out history."""
        hand = self._fp if side_tag == 'fp' else self._sh
        if cid in hand:
            hand.remove(cid)
            if face_up:
                (self._out_fp if side_tag == 'fp'
                 else self._out_sh).append(cid)
            else:
                # 1.261: a named non-public exit is the owner's PRIVATE
                # face-down discard
                (self._outfd_fp if side_tag == 'fp'
                 else self._outfd_sh).append(cid)
            self._record(line_no)
            return True
        if cid is None:
            return False          # unresolved title: handled by caller
        dl = deck_letter or ('C' if 'Character' in str(cid) else 'S')
        if self.is_deck_ambiguous(side_tag, dl):
            # resolve the earlier ambiguity: this named card WAS the phantom
            self.amb[side_tag][dl] = max(0, self.amb[side_tag][dl] - 1)
            self._close_deck_ambiguity(side_tag, dl, line_no)   # W1-002
            self.n_soft_desync += 1
            return False
        if cid in self.ever[side_tag]:
            # ever-held but gone now: an undo/return artifact, not a desync
            self.n_soft_desync += 1
            return False
        if self.desync is None:
            self.desync = ('gate3_hand_desync',
                           {'line': line_no, 'side': side_tag,
                            'missing': str(cid)[:40],
                            'hand': [str(c)[:24] for c in hand][:8]})
        return False

    def remove_random(self, side_tag, cid, line_no, deck_letter=None):
        """A random discard resolved via raw.  If unresolved / not in hand,
        mark that (side, deck) ambiguous rather than hard-quarantine."""
        hand = self._fp if side_tag == 'fp' else self._sh
        if cid is not None and cid in hand:
            # NEVER face-up (P0-a hidden-info): random/excess discards
            # are client-masked and physically face-down; they enter
            # the owner's PRIVATE face-down history (1.261)
            hand.remove(cid)
            (self._outfd_fp if side_tag == 'fp'
             else self._outfd_sh).append(cid)
            self._record(line_no)
            return True
        self.n_rand_unresolved += 1
        if deck_letter:
            self.mark_deck_ambiguous(side_tag, deck_letter, line_no)
        return False

    def remove_generic(self, side_tag, deck_letter, line_no):
        """An identity-less hand removal (excess-hand discard 'kills <deck>
        Card in <side> Hand').  We cannot know WHICH card left, so mark the
        (side, deck) ambiguous -> its later play_event samples drop (purity,
        never a guess).  The known list keeps the (now phantom) card until a
        later named play of that deck resolves the count."""
        self.n_excess_disc += 1
        self.mark_deck_ambiguous(side_tag, deck_letter, line_no)

    def hand_at(self, side_tag, line_no):
        lines = self._lines
        if not lines:
            return []
        i = _bisect.bisect_left(lines, line_no) - 1
        if i < 0:
            return []
        snap = self._snap_fp if side_tag == 'fp' else self._snap_sh
        return list(snap[i])

    def out_at(self, side_tag, line_no):
        """P0-a: cards of side_tag proven OUT of hand (played/discarded,
        not returned) just BEFORE line_no - the face-up pile surface the
        ledger can prove."""
        lines = self._lines
        if not lines:
            return []
        i = _bisect.bisect_left(lines, line_no) - 1
        if i < 0:
            return []
        snap = self._snap_out_fp if side_tag == 'fp' else self._snap_out_sh
        return list(snap[i])

    def outfd_at(self, side_tag, line_no):
        """1.261: side_tag's own FACE-DOWN discards (hand-limit / random /
        effect / EX-8 payment / WWSAT) just BEFORE line_no - private to
        that side's rows."""
        lines = self._lines
        if not lines:
            return []
        i = _bisect.bisect_left(lines, line_no) - 1
        if i < 0:
            return []
        snap = (self._snap_outfd_fp if side_tag == 'fp'
                else self._snap_outfd_sh)
        return list(snap[i])


def _raw_draw_ids(stem, fp_pw, sa_pw):
    """Ordered raw draws -> {deck_code: [card_id,...]} FIFOs + a
    (piece_id-indexed) CardDecoder for on-demand forced-discard decode.

    Returns (fifos, decoder) or (None, None) when the raw log/passwords are
    unusable (graceful empty-hand fallback, never a quarantine)."""
    import ai
    import log_replay as lr
    path = _raw_log_path(stem)
    if not path or not os.path.exists(path):
        return None, None
    try:
        with open(path, encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        dec = lr.CardDecoder()
        dec.fp_password = fp_pw
        dec.sa_password = sa_pw
        dec.preprocess_log(lines)
    except Exception:
        return None, None
    global _CARD_NAME_LOOKUP
    if _CARD_NAME_LOOKUP is None:
        _CARD_NAME_LOOKUP = ai._build_card_name_lookup()
    fifos = {'2004': [], '2005': [], '2006': [], '2007': []}
    for line in lines:
        for m in _RAW_DRAW_RE.finditer(line):
            pid, deck_code = m.group(1), m.group(2)
            try:
                info = dec.handle_draw(pid, deck_code, m.group(3))
            except Exception:
                info = None
            cid = None
            if info and info.get('decoded'):
                cid = ai._lookup_card_id(_CARD_NAME_LOOKUP,
                                         info.get('card_name', ''))
            fifos.setdefault(deck_code, []).append(cid)
    return fifos, dec


def _decode_hand_pid(decoder, pid):
    """A hand PIECE id -> (card_id, deck_letter) via the commitment deck
    (works for face-DOWN discards that are never 'reveal'-ed).  (None, None)
    for a non-card pid / decode failure."""
    import ai
    import log_replay as lr
    if 106 <= pid <= 157:
        ptype = lr._FP_PID_TYPES[pid - 106]
        base = 'FP'
    elif 158 <= pid <= 209:
        ptype = lr._SA_PID_TYPES[pid - 158]
        base = 'SA'
    else:
        return None, None
    dl = 'C' if ptype == 'C' else 'S'
    dt = base + (' Character' if ptype == 'C' else ' Strategy')
    try:
        cn = decoder.decode_piece(pid, dt)
        nm = decoder.get_card_name(cn, dt)
        cid = ai._lookup_card_id(_CARD_NAME_LOOKUP, nm) if nm else None
    except Exception:
        cid = None
    return cid, dl


def _raw_hand_out_ids(stem, decoder, regex):
    """Ordered raw hand-OUT commands (random OR excess discard) matching
    `regex` (group1=pid, group2=hand 45/46) -> {'45': [(cid, deck),...],
    '46': [...]} FIFOs, aligned 1-1 with the corresponding rendered lines."""
    out = {'45': [], '46': []}
    path = _raw_log_path(stem)
    if decoder is None or not path or not os.path.exists(path):
        return out
    with open(path, encoding='utf-8', errors='ignore') as f:
        blob = f.read()
    for m in regex.finditer(blob):
        cid, dl = _decode_hand_pid(decoder, int(m.group(1)))
        out[m.group(2)].append((cid, dl))
    return out


def _build_hand_ledger(rendered_path, stem):
    """Walk the rendered %-lines in order, driving the per-side hand ledger
    from the raw-decoded draw/discard FIFOs.  Returns a _HandLedger (with
    .desync set on a hard inconsistency) or None when passwords are absent
    (empty-hand fallback: board samples still emit, no play_event)."""
    import rendered_replay as rr
    if os.environ.get('RX_NO_LEDGER'):
        return None                      # baseline (empty-hand) comparison mode
    pw = _password_for(stem)
    if not pw:
        return None
    fifos, dec = _raw_draw_ids(stem, pw[0], pw[1])
    if fifos is None:
        return None
    disc = _raw_hand_out_ids(stem, dec, _RAW_DISCARD_RE)     # random discards
    excess = _raw_hand_out_ids(stem, dec, _RAW_EXCESS_RE)    # excess discards
    cas = _raw_hand_out_ids(stem, dec, _RAW_CAS_RE)          # W1-003 drags
    disc_ptr = {'45': 0, '46': 0}
    excess_ptr = {'45': 0, '46': 0}
    cas_ptr = {'45': 0, '46': 0}
    # W1-003: per-side stack of (line, cid) generic drags honoured so far,
    # so an 'undoes last action' of a generic drag can return THAT card
    cas_done = {'fp': [], 'sh': []}
    # W1-002: per-side stack of in-hand KILL lines honoured so far, each a
    # list of (kind, payload) per killed piece - 'card' (cid removed from
    # the hand, discard event recorded), 'amb' (deck letter, window
    # opened), 'skip' (nothing changed) - so an 'undoes last action :
    # kills ... in <Side> Hand' can revert exactly what that line did.
    # The raw excess FIFO is NOT rewound: every rendered kill line,
    # undone or not, has its own raw Controls.296 command (145 == 145
    # over the 24 card-cover games).
    kill_done = {'fp': [], 'sh': []}
    # W1-024: per-side stack of (line, cid, deck code) draws honoured so
    # far, so an 'undoes last action : %X draws <Deck> Card' removes THAT
    # card again.  The raw draw FIFO is NOT rewound: the undone draw has
    # its own raw command (1501 rendered draws == 1501 raw FIFO entries
    # over the 24 card-cover games, undone draws included) and the next
    # draw of that deck is a new raw command naming its own card.
    draw_done = {'fp': [], 'sh': []}
    led = _HandLedger()

    with open(rendered_path, encoding='utf-8', errors='ignore') as f:
        raw_lines = [ln.rstrip('\n') for ln in f]

    def _norm_zone(zone):
        # strip a trailing parenthetical annotation ('Shadow Hand (chosen)')
        return re.sub(r'\s*\([^)]*\)\s*$', '', str(zone).strip().lower())

    def _hand_side_of(zone):
        z = _norm_zone(zone)
        if z == _HAND_ZONE_FP:
            return 'fp'
        if z == _HAND_ZONE_SH:
            return 'sh'
        return None

    def _deck_zone_code(zone):
        text = _norm_zone(zone)
        if 'deck' not in text:
            return None
        side = 'free' if 'free' in text else ('shadow' if 'shadow' in text else None)
        kind = 'character' if 'character' in text else ('strategy' if 'strategy' in text else None)
        return _DECK_CODE.get(f'{side} {kind} card') if side and kind else None

    def _deck_letter(text):
        t = str(text).lower()
        if 'character' in t:
            return 'C'
        if 'strategy' in t:
            return 'S'
        return None

    def _kill_sig(piece):
        # W1-002: the signature of one killed hand piece - a named card
        # by identity, a generic '<deck> Card' by deck letter - so an
        # 'undoes last action : kills ...' finds the kill line it reverts
        cid = _title_to_cid(piece)
        if cid is not None:
            return ('N', cid)
        return ('G', _deck_letter(piece))

    for i, raw in enumerate(raw_lines, 1):
        s = raw.strip()
        if not s.startswith('%'):
            continue
        body = s[1:].strip()
        if body.startswith('<'):
            continue
        rec = rr.parse_line(body)
        if rec is None:
            continue
        op = rec.get('op')
        # ---- view anchor -------------------------------------------------
        if op == 'other' and _RENDERED_VIEW_RE.match(body):
            led.n_views += 1
            continue
        # ---- draw IN -----------------------------------------------------
        if op == 'draw':
            what = str(rec.get('what', '')).strip().lower()
            code = _DECK_CODE.get(what)
            if not code:
                continue
            fifo = fifos.get(code) or []
            cid = fifo.pop(0) if fifo else '__underflow__'
            if cid == '__underflow__':
                if led.desync is None:
                    led.desync = ('gate3_hand_desync',
                                  {'line': i, 'draw_fifo_underflow': code})
                cid = None
            side_tag = 'fp' if code in ('2004', '2005') else 'sh'
            deck_token = led.decks.draw(code, cid, i)
            led.add(side_tag, cid, i)
            led.n_draws += 1
            draw_done[side_tag].append((i, cid, code, deck_token))
            continue
        # ---- random discard OUT (raw Controls.452, commitment-resolved) ---
        if op == 'other' and _G2_RANDDISC_RE.match(body):
            # side from the DECK PHRASE only - a whole-line substring
            # scan matches 'free' inside player names ('Freek')
            _rd_m9 = re.search(
                r'randomly\s+discards\s+an?\s+(free|shadow)\b',
                body, re.I)
            side_tag = None
            if _rd_m9:
                side_tag = ('fp' if _rd_m9.group(1).lower() == 'free'
                            else 'sh')
            dl = _deck_letter(body.lower())
            if side_tag is None:
                # unresolvable side: consume NO pointer (purity â€” the
                # affected samples drop later, never a guess)
                continue
            hand_code = '45' if side_tag == 'fp' else '46'
            seq = disc.get(hand_code, [])
            ptr = disc_ptr[hand_code]
            cid = seq[ptr][0] if ptr < len(seq) else None
            disc_ptr[hand_code] = ptr + 1
            led.n_rand_disc += 1
            led.remove_random(side_tag, cid, i, dl)
            continue
        # ---- excess-hand discard: 'kills <deck> Card in <side> Hand' ------
        # generic (identity-less, face-DOWN) removal when hand > 6; resolved
        # via the raw Controls.296 excess-discard FIFO (commitment-decoded),
        # aligned 1-1 per hand.  A named card in a hand-zone kill resolves by
        # title; a FIFO underflow / deck-mismatch falls back to deck-ambiguity.
        if op == 'kill':
            zside = _hand_side_of(rec.get('zone', ''))
            if zside is not None:
                hand_code = '45' if zside == 'fp' else '46'
                done = []                                   # W1-002
                out_pub = led._out_fp if zside == 'fp' else led._out_sh
                for piece in rec.get('pieces', []):
                    cid = _title_to_cid(piece)
                    if cid is not None:
                        # a NAMED in-hand kill is a voluntary discard whose
                        # identity is already known (spec 1.216 discard emit)
                        # W1-002: only when the card really left the hand
                        # (an absent card changes nothing -> nothing to
                        # undo, no event)
                        if led.remove_named(zside, cid, i):
                            led.discard_events.append((zside, i, cid))
                            done.append(('card', cid))
                        else:
                            done.append(('skip', cid))
                        led.n_named_out += 1
                        continue
                    dl = _deck_letter(piece)
                    seq = excess.get(hand_code, [])
                    ptr = excess_ptr[hand_code]
                    if ptr < len(seq):
                        ecid, edl = seq[ptr]
                        excess_ptr[hand_code] = ptr + 1
                        led.n_excess_disc += 1
                        if dl and edl and dl != edl:
                            led.mark_deck_ambiguous(zside, dl, i)   # misaligned
                            led.discard_events.append((zside, i, None))
                            done.append(('amb', dl))
                        elif ecid is not None and ecid in out_pub:
                            # W1-002: the FIFO names a card ALREADY PLAYED
                            # face-up (2023-01-20 L730 Kindred of
                            # Glorfindel '(chosen)', L733 'kills Free
                            # Strategy Card in Free Peoples Hand') - the
                            # play's disposal, not a hand discard: the hand
                            # is unchanged, no discard decision, no
                            # ambiguity (the old code marked the deck
                            # ambiguous for the whole game)
                            led.n_disposal += 1
                            for zk in [z for z, c in led.slot.items()
                                       if c == ecid]:
                                led.slot.pop(zk, None)
                            done.append(('skip', ecid))
                        elif led.remove_random(zside, ecid, i, dl or edl):
                            # excess-hand discard = the owner's CHOICE (the
                            # commitment decode supplies the identity)
                            led.discard_events.append((zside, i, ecid))
                            done.append(('card', ecid))
                        else:
                            # not in the reconstructed hand: remove_random
                            # opened the ambiguity window at this line;
                            # the identity is NOT trusted (drop beats wrong)
                            led.discard_events.append((zside, i, None))
                            done.append(('amb', dl or edl))
                    elif dl:
                        led.remove_generic(zside, dl, i)         # underflow
                        led.discard_events.append((zside, i, None))
                        done.append(('amb', dl))
                kill_done[zside].append(
                    (i, done, tuple(_kill_sig(p)
                                    for p in rec.get('pieces', []))))
            continue
        # ---- named hand moves (play OUT / return IN) ---------------------
        if op == 'move':
            frm_raw = _norm_zone(rec.get('from', ''))
            to_raw = _norm_zone(rec.get('to', ''))
            frm = _hand_side_of(rec.get('from', ''))
            to = _hand_side_of(rec.get('to', ''))
            obj = rec.get('obj', '')
            returned_deck = _deck_zone_code(rec.get('to', ''))
            if returned_deck:
                led.decks.return_card(returned_deck, _title_to_cid(obj), i)
            if frm is not None and to != frm:
                cid = _title_to_cid(obj)
                if cid is None and to_raw in _CAS_ZONES:
                    # W1-003: identity-less hand->Casualties DRAG = a
                    # hand-limit discard whose card the raw Controls.281
                    # command names (per-hand FIFO, commitment-decoded).
                    # Mirrors the excess-kill path: resolved -> the
                    # owner's PRIVATE face-down history + a discard
                    # decision; FIFO underflow / deck mismatch -> that
                    # (side, deck) goes ambiguous and the event is a
                    # None (drop beats wrong).
                    hand_code = '45' if frm == 'fp' else '46'
                    dl = _deck_letter(obj)
                    seq = cas.get(hand_code, [])
                    ptr = cas_ptr[hand_code]
                    if ptr < len(seq):
                        ccid, cdl = seq[ptr]
                        cas_ptr[hand_code] = ptr + 1
                        led.n_cas_disc += 1
                        if dl and cdl and dl != cdl:
                            led.mark_deck_ambiguous(frm, dl, i)   # misaligned
                            led.discard_events.append((frm, i, None))
                        elif led.remove_random(frm, ccid, i, dl or cdl):
                            led.discard_events.append((frm, i, ccid))
                            cas_done[frm].append((i, ccid))
                        else:
                            # not in the reconstructed hand: remove_random
                            # already marked the deck ambiguous
                            led.discard_events.append((frm, i, None))
                    elif dl:
                        led.remove_generic(frm, dl, i)          # underflow
                        led.discard_events.append((frm, i, None))
                    continue
                # P0-a hidden-info gate: only a PLAY to a public zone
                # (play area / Board slot - event plays, revealed
                # combat cards CL-217, table plays) is face-up; a named
                # hand exit to any other zone (casualties, discard
                # boxes) mirrors a face-down discard (USER 2026-08-31)
                _pub9 = bool(to_raw in _PLAY_AREA_DEST
                             or _BOARD_DEST_RE.match(to_raw))
                led.remove_named(frm, cid, i, face_up=_pub9)
                led.n_named_out += 1
                # remember the play-area / board SLOT contents so a later
                # generic 'from Current X Card to Hand' return re-adds correctly
                if cid is not None and _pub9:
                    led.slot[to_raw] = cid
            elif to is not None and frm != to:
                cid = _title_to_cid(obj)
                if cid is None:
                    cid = led.slot.get(frm_raw)   # generic return from a slot
                if cid is not None:
                    led.add(to, cid, i)
                    led.n_named_in += 1
                    led.slot.pop(frm_raw, None)
            continue
        # ---- undo of a hand-affecting move -------------------------------
        if op == 'undo':
            inner = rr.parse_line(rec.get('inner', '') or '')
            if inner and inner.get('op') == 'kill':
                # W1-002: undo of an in-hand KILL.  The last honoured kill
                # of that side with the same deck signature is reverted:
                # its cards return to the hand and their discard decisions
                # die with it; a window it opened closes as if never
                # opened.  The raw FIFO pointer stays (the re-kill has
                # its own raw command naming the same card again:
                # 2023-09-11 L615/616 -> L627, L618-621 -> L628/629;
                # 2024-05-21 L1102/1103 -> L1194; 2023-08-15 L340/341).
                uside = _hand_side_of(str(inner.get('zone', '')))
                if uside is not None:
                    usig = tuple(_kill_sig(p) for p in inner.get('pieces', []))
                    stack = kill_done[uside]
                    for k in range(len(stack) - 1, -1, -1):
                        k_line, k_done, k_sig = stack[k]
                        if k_sig != usig:
                            continue
                        del stack[k]
                        led.n_undone_kills += 1
                        for kind, payload in reversed(k_done):
                            if kind == 'card':
                                led.add(uside, payload, i)
                                ev = (uside, k_line, payload)
                                if ev in led.discard_events:
                                    led.discard_events.remove(ev)
                            elif kind == 'amb':
                                if payload:
                                    led.unmark_deck_ambiguous(
                                        uside, payload, k_line)
                                ev = (uside, k_line, None)
                                if ev in led.discard_events:
                                    led.discard_events.remove(ev)
                        break
                continue
            if inner and inner.get('op') == 'draw':
                # W1-024: undo of a DRAW - the card the draw branch put
                # into the hand leaves it again (it never stayed: the
                # #HAND oracle lists the hand without it - 2024-05-21
                # L285 The Shadow Lengthens, 2023-01-20 L932 Orc Patrol).
                # Only event-card draws (deck code) qualify; a hunt-tile
                # 'draws <tile>' undo is not a hand event.
                ucode = _DECK_CODE.get(
                    str(inner.get('what', '')).strip().lower())
                if ucode:
                    uside = 'fp' if ucode in ('2004', '2005') else 'sh'
                    stack = draw_done[uside]
                    for k in range(len(stack) - 1, -1, -1):
                        d_line, d_cid, d_code, deck_token = stack[k]
                        if d_code != ucode:
                            continue
                        del stack[k]
                        led.decks.undo_draw(deck_token, i)
                        hand = led._fp if uside == 'fp' else led._sh
                        for h in range(len(hand) - 1, -1, -1):
                            if hand[h] == d_cid:
                                del hand[h]
                                break
                        led._record(i)
                        led.n_undone_draws += 1
                        break
                continue
            if not inner or inner.get('op') != 'move':
                continue
            frm = _hand_side_of(str(inner.get('from', '')))
            to = _hand_side_of(str(inner.get('to', '')))
            cid = _title_to_cid(inner.get('obj', ''))
            returned_deck = _deck_zone_code(inner.get('to', ''))
            if returned_deck:
                led.decks.draw(returned_deck, cid, i)
            if (frm is not None and to != frm and cid is None
                    and _norm_zone(str(inner.get('to', ''))) in _CAS_ZONES):
                # W1-003: undo of a GENERIC drag -> the last honoured drag
                # of that side returns to hand and its discard decision
                # dies with it (an unhonoured drag left nothing to undo)
                if cas_done[frm]:
                    d_line, d_cid = cas_done[frm].pop()
                    led.add(frm, d_cid, i)
                    ev = (frm, d_line, d_cid)
                    if ev in led.discard_events:
                        led.discard_events.remove(ev)
                continue
            if frm is not None and to != frm and cid is not None:
                # undo a PLAY -> card returns to hand; the play's slot
                # memory must die with it (a later generic return from
                # that slot would re-add the undone card's identity)
                led.add(frm, cid, i)
                led.slot.pop(_norm_zone(str(inner.get('to', ''))), None)
            elif to is not None and frm != to and cid is not None:
                # undo a RETURN -> card leaves hand again
                hand = led._fp if to == 'fp' else led._sh
                if cid in hand:
                    hand.remove(cid)
                    led._record(i)
            continue
        if led.desync is not None:
            break
    return led


def _cid_deck_key(cid):
    """card_id -> state.discard_face_up deck key (ids encode side+deck:
    FREE_PEOPLES_/SHADOW_ prefix, _Character_/_Strategy_ infix)."""
    s = str(cid)
    return (('fp_' if s.startswith('FREE_PEOPLES') else 'sh_')
            + ('character' if '_Character_' in s else 'strategy'))


def _install_ledger_discards(st, line):
    """P0-a (DET census 2026-08-30) + 1.261: install the ledger-proven
    discard state.  PUBLIC face-up piles (state.discard_face_up) get
    only the publicly-named play exits (event plays, revealed combat
    cards CL-217, table plays); each side's PRIVATE face-down discards
    (hand-limit / random / effect / EX-8 payment / WWSAT) go to that
    side's hands[side].discarded_face_down, which the per-side encoding
    fill lights ONLY for the owner (USER 2026-08-31: "the shadow net
    should see shadow's private knowledge").  NO menu generator reads
    either (grep-verified: every other touch is a writer or a search-
    side overlay), so installing here cannot change any legal set or
    drive verdict.  Additive merge: in-segment engine discards are
    preserved."""
    from game_state import Side
    if _LEDGER is None or not getattr(_LEDGER, 'enabled', False):
        return
    dfu = getattr(st, 'discard_face_up', None)
    if not isinstance(dfu, dict):
        return
    for tag, side in (('fp', Side.FREE_PEOPLES), ('sh', Side.SHADOW)):
        for cid in _LEDGER.out_at(tag, line):
            k = _cid_deck_key(cid)
            if k in dfu and cid not in dfu[k]:
                dfu[k].append(cid)
        pi = st.hands.get(side)
        fd = getattr(pi, 'discarded_face_down', None)
        if fd is None:
            continue
        for cid in _LEDGER.outfd_at(tag, line):
            if cid not in fd:
                fd.append(cid)
    st._rx_dfu_stamped = True


def _install_ledger_hands(st, batch, at_line=None):
    """Install both sides' reconstructed hands (as of the batch's die-stage
    line, or `at_line` when given - mid-battle draws move the hand WITHIN
    a batch: T9 Grond/Breaking/Cruel Weather were drawn between rounds)
    into the engine state.  ENRICHES every sample's encoding (card
    embedding slots + own/opp hand-size feats were EMPTY before) and exposes
    the true play_event alternatives in get_legal_actions.  No-op when no
    ledger (empty-hand fallback)."""
    from game_state import Side
    # ring-inventory stamp for states built OUTSIDE _recon_state (the
    # combat family rides rc._state_from_snapshot directly): menus must
    # see true ring counts before get_legal_actions runs
    if not getattr(st, '_rx_rings_stamped', False):
        _rg9 = (batch.get('_rx_rings') or {}).get('post')
        if _rg9 is not None:
            _stamp_rings(st, _rg9)
    # W1-025: the FP hero flags for the same outside-_recon_state states
    # (combat-family rows encoded gw/ac = 0 after the transformation)
    if not getattr(st, '_rx_heroes_stamped', False):
        _stamp_hero_flags(st, (batch.get('_rx_heroes') or {}).get('pre'))
    # W1-011: FP bid-token holdings for the same outside-_recon_state
    # states (the snapshot never carried the tokens: every menu lacked
    # the use_token options while they were held)
    if not getattr(st, '_rx_tokens_stamped', False):
        _tk9 = (batch.get('_rx_tokens') or {}).get('pre')
        if _tk9 is not None:
            _stamp_tokens(st, _tk9)
    if _LEDGER is None or not getattr(_LEDGER, 'enabled', False):
        return
    line = at_line if at_line is not None else (batch.get('start_line') or 0)
    _ensure_decks(st, line=line)
    from rendered_history import decision_boundary
    st._rx_decision_time = decision_boundary(line)
    st._rx_decision_scope = 'explicit' if at_line is not None else 'batch'
    st.hands[Side.FREE_PEOPLES].hand = list(_LEDGER.hand_at('fp', line))
    st.hands[Side.SHADOW].hand = list(_LEDGER.hand_at('sh', line))
    _install_ledger_discards(st, line)


# ==========================================================================
# TABLE-CARD (PERSISTENT-CARD) LEDGER  (CLOSURE wave, spec 1.216 items 1-4)
# --------------------------------------------------------------------------
# USER (2026-07-24): "TABLE CARDS ARE FULLY VISIBLE in the rendered stream
# (played by NAME to BoardN slots, killed by name)".  We reconstruct, purely
# from the rendered gestures, which persistent/table cards sit on the table at
# any line â€” the mirror snapshot never carried this (only armies/pol/fel/ctrl),
# which is why the prior wave DEFERRED the four hunt-interpose card decisions.
#
# A card is ON the table from its Hand->play-area (Current FP/SA Card or BoardN)
# play, through any number of position-noise slot shuffles (table<->table stays
# ON), until it is KILLED by name (the USE gesture for the hunt cards) or moved
# off-table to a non-play zone.  The reconstructed persistent_cards state is
# installed into the engine so game_logic._offer_hunt_reduction /
# _resolve_hunt_tile open their REAL interposes and get_legal_actions produces
# the same use/decline menu self-play faces (legality-matched, emit-or-drop).
#
# Scope: the four FP hunt-interpose cards (all FREE_PEOPLES side).  Extending to
# any other PersistentCardType is a one-line addition to _PERSISTENT_TITLE_MAP.
# ==========================================================================
# rendered card TITLE substring -> gs.PersistentCardType attribute name
_PERSISTENT_TITLE_MAP = (
    ('balrog of moria', 'BALROG_OF_MORIA'),
    ('mithril coat', 'MITHRIL_COAT'),
    ('axe and bow', 'AXE_AND_BOW'),
    ('horn of gondor', 'HORN_OF_GONDOR'),
    ("wizard's staff", 'WIZARD_STAFF'),
    ('wizard s staff', 'WIZARD_STAFF'),
    # 1.149 KILLABLE TABLE CARDS (USER 2026-08-20: "give it its own
    # category and extract the training signal"): without these the
    # victim is not on the reconstructed table, so
    # ai._get_table_kill_actions offers NOTHING and the human's removal
    # can never be matched.
    ('a power too great', 'POWER_TOO_GREAT'),
    ('power of tom bombadil', 'TOM_BOMBADIL'),
    ('palantir of orthanc', 'PALANTIR_OF_ORTHANC'),
    ('palant', 'PALANTIR_OF_ORTHANC'),      # accented spellings
    ("denethor's folly", 'DENETHORS_FOLLY'),
    ('denethor s folly', 'DENETHORS_FOLLY'),
)
_HUNT_PTYPES = ('MITHRIL_COAT', 'AXE_AND_BOW', 'HORN_OF_GONDOR', 'WIZARD_STAFF')
_ROUTE_PTYPES = {'BALROG_OF_MORIA': 'SHADOW'}
# the killable set, with the SIDE that owns each (the hunt four are all
# FP, which is why _install_persistent_cards hardcoded that side)
_KILLABLE_PTYPES = {
    'POWER_TOO_GREAT': 'FREE_PEOPLES',
    'TOM_BOMBADIL': 'FREE_PEOPLES',
    'PALANTIR_OF_ORTHANC': 'SHADOW',
    'DENETHORS_FOLLY': 'SHADOW',
}
_TABLE = None                # module-global per-game table ledger
_REACTIVE_TABLE = None       # normalized gestures plus compulsory declaration cleanup
_RX_STATS = None             # module-global stats handle (census)


def _title_to_ptype(title):
    """Rendered card TITLE -> PersistentCardType attribute name, or None."""
    t = str(title).lower()
    for sub, name in _PERSISTENT_TITLE_MAP:
        if sub in t:
            return name
    return None


def _zone_is_table(zone):
    """A play-area / board table slot (persistent-card home)."""
    z = re.sub(r'\s*\([^)]*\)\s*$', '', str(zone).strip().lower())
    return z in _PLAY_AREA_DEST or bool(_BOARD_DEST_RE.match(z))


class _TableLedger:
    """Line-indexed persistent-card presence, reconstructed from gestures.

    events: sorted [(line, ptype_name, kind)] with kind in
    'on' / 'off_kill' / 'off_move'.  A KILL is the hunt-card USE gesture
    (Axe/Horn discard-to-reduce, Wizard prevent, Mithril redraw); a plain
    move off-table is position noise / a Crack-of-Doom active-card relocation
    (Mithril keep), NOT a use."""

    def __init__(self):
        self.events = []
        self.cid = {}            # ptype -> resolved card_id (from the ON title)
        self.n_on = 0
        self.n_off_kill = 0
        self.n_off_move = 0

    def add(self, line, ptype, kind):
        self.events.append((int(line), ptype, kind))
        if kind == 'on':
            self.n_on += 1
        elif kind == 'off_kill':
            self.n_off_kill += 1
        else:
            self.n_off_move += 1

    def on_at(self, ptype, line):
        """True iff `ptype` is on the table just BEFORE `line`."""
        state = False
        for (ln, pt, k) in self.events:      # events built in line order
            if ln >= line:
                break
            if pt == ptype:
                state = (k == 'on')
        return state

    def killed_in(self, ptype, lo, hi):
        """The card was KILLED (used) within [lo, hi)."""
        return any(lo <= ln < hi and pt == ptype and k == 'off_kill'
                   for (ln, pt, k) in self.events)


def _build_table_ledger(rendered_path):
    """Walk the rendered %-lines in order, tracking each tracked persistent
    card's ON/OFF transitions.  Returns a _TableLedger (empty when nothing
    tracked was ever played)."""
    import rendered_replay as rr
    led = _TableLedger()
    try:
        with open(rendered_path, encoding='utf-8', errors='ignore') as f:
            raw_lines = [ln.rstrip('\n') for ln in f]
    except Exception:
        return led
    for i, raw in enumerate(raw_lines, 1):
        s = raw.strip()
        if not s.startswith('%'):
            continue
        body = s[1:].strip()
        if body.startswith('<'):
            continue
        rec = rr.parse_line(body)
        if rec is None:
            continue
        op = rec.get('op')
        if op == 'move':
            pt = _title_to_ptype(rec.get('obj', ''))
            if pt is None:
                continue
            frm_t = _zone_is_table(rec.get('from', ''))
            to_t = _zone_is_table(rec.get('to', ''))
            if to_t and not frm_t:
                led.add(i, pt, 'on')
                if pt not in led.cid:
                    led.cid[pt] = _title_to_cid(rec.get('obj', ''))
            elif frm_t and not to_t:
                led.add(i, pt, 'off_move')
        elif op == 'kill':
            for p in rec.get('pieces', []):
                pt = _title_to_ptype(p)
                if pt is not None:
                    led.add(i, pt, 'off_kill')
    led.events.sort()
    return led


def _install_persistent_cards(st, line, killable=False, route_only=False):
    """Install every tracked persistent card ON the table at `line` into the
    reconstructed state's persistent_cards (game_state.PersistentCardState).
    ENRICHES encode_state (card_status = 1.0 for on-table cards) AND exposes
    the hunt-reduction interposes to get_legal_actions.  No-op without a
    table ledger."""
    from game_state import PersistentCard, PersistentCardType, Side
    if _TABLE is None:
        return
    pcs = getattr(st, 'persistent_cards', None)
    if pcs is None:
        return
    # killable=False by DEFAULT.  The killable set is only needed where
    # an ACTION menu is built (a 1.149 removal costs an action die, so
    # it can never be an option inside a hunt/combat sub-decision);
    # installing the four extra card types on the hunt/combat drive
    # paths changes hunt outcomes and makes the driven hunt diverge
    # from the log, so it is opt-in.
    for pt_name in (tuple(_ROUTE_PTYPES)
                    + (() if route_only else tuple(_HUNT_PTYPES))
                    + (tuple(_KILLABLE_PTYPES) if killable and not route_only else ())):
        if not _TABLE.on_at(pt_name, line):
            continue
        ptype = getattr(PersistentCardType, pt_name)
        if pcs.get_by_type(ptype) is not None:
            continue
        cid = _TABLE.cid.get(pt_name) or ('__%s__' % pt_name.lower())
        # the hunt four are all FP; the
        # killable set spans both sides and the OWNER decides who may
        # remove it, so the side has to be right
        _sd = _KILLABLE_PTYPES.get(pt_name, _ROUTE_PTYPES.get(pt_name, 'FREE_PEOPLES'))
        pcs.add_card(PersistentCard(
            card_id=str(cid), card_name=ptype.value,
            side=(Side.SHADOW if _sd == 'SHADOW'
                  else Side.FREE_PEOPLES), card_type=ptype))


# ==========================================================================
# corpus enumeration
# ==========================================================================
def _ok_stems():
    """Stems the JVM sweep rendered successfully (manifest status == ok) and
    whose .rendered.txt is on disk, in manifest order."""
    out = []
    if not os.path.exists(CORPUS_MANIFEST):
        return out
    with open(CORPUS_MANIFEST, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if (r.get('status') or '').strip() != 'ok':
                continue
            stem = r['stem'].strip()
            if os.path.exists(os.path.join(RLOG_CORPUS, stem + '.rendered.txt')):
                out.append(stem)
    return out


def _z_meta():
    """{stem: {winner_side, victory, turns}} via extract_corpus.corpus_rows
    (base game, both passwords, curation exclusions)."""
    import extract_corpus as ec
    rows = ec.corpus_rows()               # keyed by <stem>.log
    return {ec.stem_of(name): m for name, m in rows.items()}


def _canary_files():
    import glob
    fs = sorted(glob.glob(os.path.join(RLOG_CANARY, 'GAME*.rendered.txt')))
    return [f for f in fs if 'branstonpickle_Compote.rendered' not in f]


def _canary_stem(path):
    m = re.match(r'^GAME\d+_(.*)\.rendered\.txt$', os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


# ==========================================================================
# gate helpers
# ==========================================================================
def _scan_lines_gate2(rendered_path):
    """Gate 2: every %-line must be a parsed decision/no-op or a MODELED
    class.  Returns (ok, reason, catalog Counter).  Unknown 'other' -> not-ok
    (+cataloged).

    UNDO/REDO (amendment 1.216) are now MODELED, not quarantined: parse_line
    tags them op='undo'/'redo' and rendered_replay.resolve_undo_redo collapses
    them to the settled state before the mirror ever runs, so a redo game is
    RESOLVED (gate2_redo removed â€” this is what raises the admission rate)."""
    import rendered_replay as rr
    cat = Counter()
    unknown = Counter()
    with open(rendered_path, encoding='utf-8', errors='ignore') as f:
        for raw in f:
            s = raw.strip()
            if not s.startswith('%'):
                continue
            body = s[1:].strip()
            if body.startswith('<'):
                continue
            rec = rr.parse_line(body)
            if rec is None:
                continue
            op = rec.get('op')
            if op == 'undo':
                cat['modeled_undo'] += 1
                continue
            if op == 'redo':
                cat['modeled_redo'] += 1
                continue
            if op != 'other':
                continue
            if _G2_NOOP_RE.match(body):
                cat['noop_view/deselect'] += 1
            elif _G2_DIECHANGE_RE.match(body):
                cat['modeled_die_change'] += 1
            elif _G2_RANDDISC_RE.match(body):
                cat['modeled_random_discard'] += 1
            else:
                tmpl = re.sub(r'\d+', 'N', re.sub(r'^\S+\s+', '', body))[:50]
                unknown[tmpl] += 1
    if unknown:
        top = unknown.most_common(1)[0][0]
        return False, 'gate2_unparsed:%s' % top, cat + unknown
    return True, None, cat


def _rendered_draw_decks(rendered_path):
    """Ordered VASSAL deck codes for every card draw, scanned directly from
    the rendered %-lines (batch rng loses turn-start draws that precede the
    first die-staging)."""
    seq = []
    with open(rendered_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s.startswith('%'):
                continue
            body = s[1:].strip()
            if body.startswith('<'):
                continue   # chat body, never a draw line (gate-2 twin)
            m = _RENDERED_DRAW_LINE_RE.match(body)
            if m:
                code = _DECK_CODE.get(m.group(1).strip().lower())
                if code:
                    seq.append(code)
    return seq


def _raw_draw_decks(stem):
    """Ordered VASSAL deck codes decoded from the raw .log (card draws only).
    Returns None when the raw log is absent."""
    path = _raw_log_path(stem)
    if not path or not os.path.exists(path):
        return None
    seq = []
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            for m in _RAW_DRAW_RE.finditer(line):
                seq.append(m.group(2))
    return seq


def _gate3_draws(rendered_path, stem):
    """Gate 3: the rendered card-draw stream pairs 1:1 IN ORDER with the raw
    decoded draws.  Order-exact is required (falls back to count on a rare
    interleave); returns (ok, note)."""
    rnd = _rendered_draw_decks(rendered_path)
    raw = _raw_draw_decks(stem)
    if raw is None:
        return False, 'gate3_no_rawlog'
    if rnd == raw:
        return True, 'ordered(n=%d)' % len(rnd)
    cr, cw = Counter(rnd), Counter(raw)
    if cr != cw:
        diff = {k: [cr.get(k, 0), cw.get(k, 0)]
                for k in set(cr) | set(cw) if cr.get(k) != cw.get(k)}
        return False, 'gate3_draw_mismatch:%s' % json.dumps(diff)
    return False, 'gate3_draw_order:n=%d' % len(rnd)


def _conservation_ok(snap):
    """Gate 6: deployed figures plus permanent FP losses fit physical supply.
    Summed across ALL regions, so it is
    stronghold/siege/co-location agnostic (per-region stacking + besieged
    garrison caps are the engine's own legality, enforced by the drive, not
    re-derived here where the co-located model would false-fire).  Returns
    (ok, detail)."""
    per_nat = {}                # nation -> {REGULAR,ELITE,LEADER}
    nazgul = 0
    for _key, (units, leaders) in snap['armies'].items():
        for (nat, kind) in units:
            per_nat.setdefault(nat, Counter())[kind] += 1
        for ld in leaders:
            if ld[0] == 'LEADER':
                per_nat.setdefault(ld[1], Counter())['LEADER'] += 1
            elif ld[0] == 'NAZGUL':
                nazgul += 1
    if nazgul > _MAX_NAZGUL:
        return False, 'nazgul %d>%d' % (nazgul, _MAX_NAZGUL)
    for nat, counts in snap.get('fpc', {}).items():
        if nat not in ('ELVES', 'DWARVES', 'GONDOR', 'THE_NORTH', 'ROHAN'):
            return False, 'invalid_casualty_nation:' + str(nat)
        for kind, count in counts.items():
            if kind not in ('regular', 'elite', 'leader') or type(count) is not int or count < 0:
                return False, 'invalid_casualty_count:' + str(nat)
            per_nat.setdefault(nat, Counter())[kind.upper()] += count
    for nat, d in per_nat.items():
        cap = _NATION_POOL.get(nat)
        if not cap:
            continue
        for kind in ('REGULAR', 'ELITE', 'LEADER'):
            if d.get(kind, 0) > cap.get(kind, 0):
                return False, '%s.%s %d>%d' % (nat, kind, d[kind], cap[kind])
    return True, None


def _z_from_csv_board(final_snap, meta):
    """Gate 7 (rulings 1.212 + 1.214): CSV winner is the authoritative label;
    the final rendered board is a cross-check flag only.  Returns
    (z_tag, z_source, conflict)."""
    ws = str(meta.get('winner_side', '')).lower()
    csv_tag = 'fp' if ws.startswith('free') else 'sh'
    fel = final_snap['fel']
    corr = fel.get('corruption', 0)
    # count military VP off the rendered control set + captured settlements is
    # lossy; use the decided-board safety net (corruption ring win only, the
    # unambiguous board signal) â€” military VP needs the engine tally.
    board_tag = 'sh' if corr >= 12 else None
    if board_tag and board_tag != csv_tag:
        return csv_tag, 'board_conflict', True
    return csv_tag, 'csv', False


# ==========================================================================
# per-batch engine drive + legality-matched sample emission
# ==========================================================================
def _army_move_records(batch):
    """DISTINCT (from_region, to_region) region->region figure moves in the
    batch, IN ORDER, plus whether any move carried a plain ARMY UNIT.  A
    single army move renders as several lines (units, leaders, split groups)
    all sharing one (from,to) â€” those coalesce to one decision; an Army die
    legitimately moves up to two DIFFERENT armies (two distinct pairs).
    Returns [(from, to, has_unit)] deduped in first-seen order."""
    import rendered_replay as rr
    seen = {}
    order = []
    for r in batch['records']:
        if r.get('op') != 'move':
            continue
        obj = str(r.get('obj', ''))
        ol = obj.strip().lower()
        if ol in ('the fellowship', 'fsp', 'corruption', 'nothing') or \
                ol.endswith('control marker') or \
                ol.startswith('victory points marker'):
            continue
        fk = rr.zone_kind(r.get('from', ''))
        tk = rr.zone_kind(r.get('to', ''))
        if fk[0] not in ('region', 'stronghold') or \
                tk[0] not in ('region', 'stronghold'):
            continue
        if fk[1] == tk[1]:
            continue
        mg = re.match(r'^group\s*\((.*)\)$', obj, re.I)
        toks = rr.split_group(mg.group(1)) if mg else [obj]
        has_unit = has_fig = False
        for t in toks:
            if t.strip().lower() == 'the fellowship':
                continue
            p = rr.parse_piece(t)
            if p and p[0] == 'unit':
                has_unit = has_fig = True
            elif p and p[0] in ('nazgul', 'leader', 'named'):
                has_fig = True
        if not has_fig:
            continue
        key = (fk[1], tk[1])
        if key not in seen:
            seen[key] = has_unit
            order.append(key)
        elif has_unit:
            seen[key] = True
    return [(f, t, seen[(f, t)]) for (f, t) in order]


def _political_nation(batch):
    """The single nation advanced/activated this batch, or None."""
    import rendered_replay as rr
    nats = set()
    for r in batch['records']:
        if r.get('op') == 'move':
            tk = rr.zone_kind(r.get('to', ''))
            fk = rr.zone_kind(r.get('from', ''))
            if tk[0] == 'political' or fk[0] == 'political':
                p = rr.parse_piece(r.get('obj', ''))
                if p and p[0] in ('nation', 'leader', 'unit'):
                    nats.add(p[1])
                elif str(r.get('obj', '')).strip().lower() in rr.NATION_NAME:
                    nats.add(rr.NATION_NAME[str(r.get('obj', '')).strip().lower()])
        elif r.get('op') == 'flip' and 'political' in \
                (r.get('zone', '') or '').lower():
            for part in str(r.get('obj', '')).split(','):
                nat = rr.NATION_NAME.get(part.strip().lower())
                if nat:
                    nats.add(nat)
    return next(iter(nats)) if len(nats) == 1 else None


# companion display name -> (level, leadership); setup_initial_game values
_COMP_LEVELS = {'gandalf the grey': (3, 1), 'strider': (3, 1),
                'boromir': (2, 1), 'legolas': (2, 1), 'gimli': (2, 1),
                'meriadoc': (1, 1), 'peregrin': (1, 1),
                'merry': (1, 1), 'pippin': (1, 1),
                'gandalf the white': (3, 1), 'aragorn': (3, 2)}


def _retype_companions(st):
    """Census-recon army leaders are plain Leader objects; re-type named
    COMPANIONS with the engine's own class (RB-02 sibling, ONE
    implementation for _recon_state and the combat drive).  A
    Companion's leadership equals a plain Leader's, so combat math is
    unchanged - typing only enables the Companion-dispatched paths
    (1.254 target enumeration, shine, crown)."""
    import game_state as _gs9
    import rendered_replay as _rr9c
    for _a9 in st.armies.values():
        for _i9, _l9 in enumerate(_a9.leaders):
            if type(_l9).__name__ != 'Leader':
                continue
            _nm9 = str(getattr(_l9, 'name', '') or '').lower()
            _cn9 = _rr9c.COMP_ALIAS.get(_nm9, _nm9).lower()
            _lv9 = _COMP_LEVELS.get(_cn9) or _COMP_LEVELS.get(_nm9)
            if _lv9:
                _a9.leaders[_i9] = _gs9.Companion(
                    name=_rr9c.COMP_ALIAS.get(_nm9,
                                              getattr(_l9, 'name')),
                    level=_lv9[0], leadership=_lv9[1])


def _recon_state(snap, turn=None):
    """rendered_census._state_from_snapshot (imported, never edited) COMPLETED:
    the census reconstruction leaves fellowship.region a bare sanitized STRING,
    but the engine models it as a board.Region object (game_logic reads
    ``state.fellowship.region.key`` â€” can_crown_aragorn crashed the Will-of-West
    special enumeration on the string form).  We finish the reconstruction here
    (import-only: rendered_census is not modified) by promoting the region to
    the real Region object, so a Will/Character die enumerates cleanly."""
    import rendered_census as rc
    import board
    st = rc._state_from_snapshot(snap)
    if isinstance(snap, dict) and snap.get('_rx_hunt_observation') is not None:
        st._rx_hunt_observation = tuple(snap['_rx_hunt_observation'])
        from rendered_history import decision_boundary
        st._rx_decision_time = decision_boundary(st._rx_hunt_observation[0])
        st._rx_decision_scope = 'snapshot'
        if _LEDGER is not None and getattr(_LEDGER, 'enabled', False):
            _ensure_decks(st, line=st._rx_hunt_observation[0])
    # skew-gate P0: the census snapshot carries no turn number, so
    # every call site threads the batch turn - the encode_state turn
    # features (cols 3339/3419: turn/15, turn/25) must VARY across the
    # store.
    if turn:
        try:
            st.turn = max(1, int(turn))
        except (TypeError, ValueError):
            pass
    # decision-time elven-ring inventories (ledger-stamped by
    # _annotate_ring_ledger into the snapshot; setup defaults FP=3/SH=0
    # are wrong for every state after the first ring use)
    _rg9 = snap.get('_rx_rings') if isinstance(snap, dict) else None
    if _rg9 is not None:
        _stamp_rings(st, _rg9)
    reg = st.fellowship.region
    if isinstance(reg, str):
        try:
            st.fellowship.region = board.load_board().region(reg)
        except Exception:
            pass                # leave the string; only WotW crown needs .key
    # USER truth-table fix (T2 transform): minion in-play flags from board
    # presence - can_transform_gandalf_white requires a minion "is or has
    # been in play" and the census recon never set the flags (Saruman on
    # the board at Orthanc, flag False -> transform menu empty).
    _names = set()
    for _a9 in st.armies.values():
        for _l9 in _a9.leaders:
            _nm9 = str(getattr(_l9, 'name', '') or '')
            if _nm9:
                _names.add(_nm9.lower())
    for _attr, _keys in (('saruman', ('saruman',)),
                         ('witch_king', ('witch-king', 'witch king',
                                         'the witch-king')),
                         ('mouth_of_sauron', ('mouth of sauron',))):
        _mo9 = getattr(st, _attr, None)
        if _mo9 is not None and any(
                any(k in n for n in _names) for k in _keys):
            _mo9.in_play = True
            _mo9.was_ever_in_play = True
    # W1-025 (RB-02 sibling): the two FP hero flags the engine sets once
    # and never clears (gandalf_the_white.in_play / aragorn_crowned.crowned)
    _stamp_hero_flags(st, snap.get('_rx_heroes') if isinstance(snap, dict)
                      else None, _names)
    # W1-011: FP bid-token holdings at the decision point (ledger-stamped
    # by _annotate_token_ledger; the census snapshot has no token model,
    # so every state read 'no tokens' and no menu offered use_token)
    _tk9 = snap.get('_rx_tokens') if isinstance(snap, dict) else None
    if _tk9 is not None:
        _stamp_tokens(st, _tk9)
    # RB-02: the census recon
    # types named minions as plain Leaders, but the engine dispatches on
    # type name 'Minion' (nazgul-chain include_wk hops, WK-army
    # followups, Saruman immobility, minion sweeps).  Re-type in place
    # with the engine's own class so every minion path enumerates.
    # GAME19 T3 (crown): separated COMPANIONS get the same treatment -
    # can_crown_aragorn / companion moves dispatch on Companion.
    import game_logic as _gl9
    import game_state as _gs9
    for _a9 in st.armies.values():
        for _i9, _l9 in enumerate(_a9.leaders):
            if type(_l9).__name__ != 'Leader':
                continue
            _nm9 = str(getattr(_l9, 'name', '') or '').lower()
            if ('witch' in _nm9 or 'mouth of sauron' in _nm9
                    or 'saruman' in _nm9):
                _mn9 = _gl9.Minion(nation=_gs9.Nation.SAURON,
                                   name=getattr(_l9, 'name'))
                if 'saruman' in _nm9:
                    _mn9.immobile = True
                _a9.leaders[_i9] = _mn9
    _retype_companions(st)
    # Gandalf the Grey no longer among comps AND not on the board = fell
    # (transform precondition path for dead-Grey)
    return st


# WHICH-DIE MARGINALIZATION matcher context (USER 2026-07-27): set by
# _mk_state_with_die for the current batch, cleared by the batch loop.
# _match_index uses it to disambiguate a params-level multi-match by
# the die FACE the human actually spent (the same action can appear
# under several faces in a full-pool menu).
_MM_FACE = None
_MM_SIDE = None
_MM_STATE = None

# Ring-inventory ledger context: _annotate_ring_ledger stamps
# batch['_rx_rings'] + both truth snapshots; _recon_state /
# _install_ledger_hands read them so every reconstructed state carries
# decision-time elven-ring inventories (the census snapshot holds no
# ring state).
# _RX_RINGS_CUR backstops census-driver-internal states per batch.
_RX_RINGS_CUR = None
# P0-a: current batch start line - _emit_sample's discard-pile backstop
# for states that never saw _install_ledger_hands (resolver-path drives)
_RX_LINE_CUR = None
# Public bag provenance is independent of replay's singleton RNG scaffolding.
_RX_HUNT_HISTORY = None
# 1.268 (USER 2026-09-03, sponsor pick (a) positive SET): alternate legal
# indices that are EQUALLY right for the sample being emitted (a ring take
# whose K enabling faces no consequence pins down).  Set by the emitter
# immediately before its _emit_sample call, consumed and cleared there;
# lands in the sample as 'alt', in the npz as alt_idx / alt_offsets, in the
# ledger as alt_acts, and in train_corpus._menu_rows as extra positives of
# the listwise multi-positive CE (-log(sum_pos p)).  Never a weight.
_RX_ALT_HIDX = None
# X-audit B1 (2026-08-31): per-game {turn: n} pre-roll eye allocations /
# rolled-Eye counts, set by extract_one for _emit_sample's hunt-box
# backstop (X3192 sat at 0 corpus-wide without it).
_RX_EYEALLOC = None
_RX_EYEROLL = None
# W4-032 (sponsor condition 1, 2026-09-03): set by an emitter right before
# its _emit_sample call when the chosen action is a Saruman ability variant
# that the v41 action encoder cannot tell from its plain Orthanc twin;
# consumed and cleared at the top of _emit_sample into sample['alias_v41']
# (the C2-004 _RX_ALT_HIDX pattern).  extract_corpus.pack_npz refuses such a
# sample under a pre-v43 tag unless the caller acknowledges it.
_RX_ALIAS_V41 = None


def _mk_state_with_die(batch, install_hands=False):
    """Reconstruct the engine state at the batch's rendered truth-before
    and stage the acting side's FULL dice context (v31 which-die
    marginalization, USER 2026-07-27): every remaining unused rolled
    face plus the turn's already-spent faces as used dice.  The legal
    menu therefore spans ALL the player's live options â€” the true
    marginal decision surface â€” instead of being conditioned on the
    face the human happened to spend.  The batch's own face leads the
    pool; _match_index resolves cross-face duplicates by face.
    Falls back to single-die staging when pool annotation is missing.
    Returns (state, side) or (None, reason).

    install_hands=True installs the ledger-reconstructed hands (for EMITTED
    samples: enriches the encoding + exposes play_event alternatives).  The
    resolver DRIVE path leaves it False so the certified hunt/combat drivers
    run with empty hands exactly as before (canary-preserving)."""
    global _MM_FACE, _MM_SIDE, _MM_STATE
    import game_state as gs
    from game_state import Side
    face = _letter2face().get(batch['die_face'])
    if face is None or face == gs.ActionDieFace.EYE:
        return None, 'die_face(%s)' % batch['die_face']
    side = Side.SHADOW if batch['die_side'] == 'SHADOW' else Side.FREE_PEOPLES
    try:
        st = _recon_state(batch['truth_before'], batch.get('turn'))
    except Exception as e:
        return None, 'recon(%s)' % str(e)[:30]
    faces_of = gs.SH_FACES if side == Side.SHADOW else gs.FP_FACES

    def _mk_die(f, used):
        d = gs.ActionDie(faces_of, side)
        d.current = f
        d.used = used
        return d

    l2f = _letter2face()
    key = 'SHADOW' if side == Side.SHADOW else 'FREE_PEOPLES'
    poolinfo = (batch.get('_rx_pools') or {}).get(key)
    dice = [_mk_die(face, False)]           # the human's die leads
    if isinstance(poolinfo, dict):
        rest = list(poolinfo.get('unused') or [])
        # remove ONE instance of the batch face (it leads the list)
        bl = str(batch['die_face']).upper()
        if bl in rest:
            rest.remove(bl)
        for L in rest:
            f = l2f.get(L)
            if f is not None and f is not gs.ActionDieFace.EYE:
                dice.append(_mk_die(f, False))
        for L in (poolinfo.get('used') or []):
            f = l2f.get(L)
            if f is not None:
                dice.append(_mk_die(f, True))
    st.dice = {Side.FREE_PEOPLES: [], Side.SHADOW: []}
    st.dice[side] = dice
    # encode-time swap context for _emit_sample (both sides' pools)
    st._rx_full_pools = batch.get('_rx_pools')
    _MM_FACE, _MM_SIDE, _MM_STATE = face, side, st
    if install_hands:
        _install_ledger_hands(st, batch)
        # TABLE STATE ON EVERY EMITTED SAMPLE (USER 2026-08-20: "we
        # need to make sure that we consider every action that occurs
        # while these cards are on the board = not choosing to discard
        # at that moment").
        #
        # THIS IS WHERE THE PRIOR COMES FROM.  A table kill can only
        # appear in a menu when the victim is modelled on the table;
        # installing it ONLY on the kill path would have taught the
        # net that the action exists exclusively in the states where it
        # was taken - a 100% prior on a rare move.  Installed here, the
        # removal shows up as an UNCHOSEN option in every decision made
        # while the card sits on the board, which is exactly the
        # negative evidence that makes the prior small and honest.
        # Denethor's Folly is deliberately INCLUDED here (it is
        # excluded only from being LABELLED as chosen), so it earns the
        # low prior the USER asked for instead of an undefined one.
        #
        # The resolver DRIVE path passes install_hands=False and stays
        # byte-identical, so the certified hunt/combat canary is
        # untouched.
        _install_persistent_cards(st, batch.get('start_line') or 0,
                                  killable=True)
        # populate decks so the Event-die draw_card menu (character vs
        # strategy) is legal-matchable; leaves the resolver DRIVE path (which
        # passes install_hands=False) byte-identical to before.
        if _LIST2:
            _ensure_decks(st)
    return st, side


def _effective_lead_letter(batch):
    """Resolve the die funding the observed action, without overriding probes."""
    if _RING_PROBE is not None:
        return batch['die_face']
    letter = _lead_change_lines(batch)[0] or batch['die_face']
    ring_side = _ring_counter_used(batch)
    if (letter == batch['die_face'] and ring_side
            and ('SHADOW' if ring_side == 'sh' else 'FREE_PEOPLES') == batch['die_side']
            and not _ring_is_tablekill_payment(batch)):
        letter = _ring_face_analysis(batch).get('mirror') or letter
    return letter


def _set_effective_lead_face(state, side, face):
    """Change the staged die in both the action menu and its observation.

    Pool annotations are shared with the pre-ring observation, so copy them
    before replacing one unused face. Previously spent dice stay unchanged.
    """
    dice = state.dice.get(side) or []
    if not dice or dice[0].current == face:
        return
    pools = getattr(state, '_rx_full_pools', None)
    if pools is not None:
        from copy import deepcopy
        letter_of = {value: key for key, value in _letter2face().items()}
        pools = deepcopy(pools)
        unused = pools[side.name]['unused']
        old_letter = letter_of[dice[0].current]
        if old_letter not in unused:
            raise ValueError('effective die missing from unused observation pool')
        unused[unused.index(old_letter)] = letter_of[face]
        state._rx_full_pools = pools
    dice[0].current = face


def _annotate_remaining_pools(batches, pools, eye_counts=None):
    """Reconstruct both sides' dice from ordered rolls and zone gestures.

    Each batch gets its pre-action pool in `_rx_pools`. Selected records
    also get their exact prefix pools, currently staged faces, and a
    pre-Hunt-allocation flag. A staged die is booked as used for later
    card/discard observations; the ring observer restores only dice still
    staged when that free action is considered. Earlier spent dice stay used.

    Rolls reset pools; staging, retraction, removal, parking on a card and
    explicit face changes update them. Eyes are excluded from usable dice.
    Before Hunt allocation, record observations contain no future roll/Eyes.
    Legacy no-roll reconstruction estimates initial supply from the turn's
    observed deficits; it is a fallback, not proof of unobserved die faces.
    """
    from collections import Counter as _C
    eye_counts = eye_counts or {}
    import rendered_replay as _rr
    u2side = {}
    for b in batches:
        for r in b['records']:
            if r.get('op') == 'move' and r.get('die_stage'):
                u2side[r.get('player')] = r['die_stage'][0]
    LEAVE_PLAY = ('reinforcements', 'hunt pool')

    def _dz(z):
        return str(z or '').strip().lower() in ('sa dice', 'fp dice')

    def _setup_zone(z):
        z = str(z or '').strip().lower()
        return ('casualties' in z or 'reinforcements' in z
                or 'hunt for the ring' in z or z == 'hunt pool')

    def _cz(z):
        z = str(z or '').strip().lower()
        return z.startswith('current') and z.endswith('card')

    def _ca(z):
        return str(z or '').strip().lower() == 'current action'

    # ---- pass 1: per turn, the ordered die-gesture event stream ---------
    # ev = (kind, side, face, extra); kinds: roll / leave / enter / change /
    # snap (batch index)
    turns = {}
    hunt_started = {}
    order = []
    for i, b in enumerate(batches):
        t = b['turn']
        if t not in turns:
            turns[t] = []
            hunt_started[t] = False
            order.append(t)
        ev = turns[t]
        lead_lines = None
        for r in b['records']:
            if isinstance(r.get('decision_truth'), dict):
                r['_rx_before_hunt_allocation'] = not hunt_started[t]
                ev.append(('snap_record', None, None, (i, r)))
            if r.get('_rx_shadow_hunt_roundtrip'):
                # Same physical die returned before a new action or Hunt
                # outcome: neither leg spends or restores an action die.
                continue
            op = r.get('op')
            if op in ('alloc_eye', 'add_eye', 'roll_dice') or r.get('die_stage'):
                hunt_started[t] = True
            if op == 'roll_dice':
                sd = u2side.get(r.get('player'))
                if sd is not None:
                    partial = r.get('_rx_partial_roll')
                    if partial is not None:
                        zone = r.get('_rx_partial_roll_zone')
                        if zone in ('Game.2035', 'Game.2036', 'Game.2037'):
                            ev.append(('partial_roll_staged', sd, partial[0], (partial[1], zone)))
                        else:
                            ev.append(('partial_roll', sd, partial[0], partial[1]))
                    else:
                        ev.append(('roll', sd,
                                   [str(x).upper() for x in r.get('faces', [])],
                                   None))
                continue
            if op == 'move':
                if r.get('die_stage') and int(r.get('line', -1)) == \
                        int(b.get('start_line') or -2):
                    ev.append(('snap', None, None, i))
                if r.get('_rx_dwarven_selection'):
                    continue
                toks = _die_tokens(r.get('obj', ''))
                if r.get('_rx_lidless_original_faces'):
                    toks = [('SHADOW', face) for face in r['_rx_lidless_original_faces']]
                if not toks:
                    continue
                frm, to = r.get('from', ''), r.get('to', '')
                if _dz(frm) and not _dz(to):
                    lo = str(to or '').strip().lower()
                    if _cz(to):
                        # parked on a card zone: neither unused nor
                        # spent yet (via-card staging or a card
                        # removing it - resolved by its next gesture)
                        for sd, f in toks:
                            ev.append(('park', sd, f, None))
                        continue
                    book = not any(x in lo for x in LEAVE_PLAY)
                    for sd, f in toks:
                        ev.append(('leave', sd, f, (book, _ca(to))))
                elif _dz(to) and not _dz(frm):
                    if _setup_zone(frm) and not r.get('_rx_physical_die_restore'):
                        continue        # turn-setup return: the roll resets
                    for sd, f in toks:
                        ev.append(('enter', sd, f, _ca(frm)))
                elif r.get('die_stage') and _cz(frm):
                    # via-card staging (card zone -> Current Action):
                    # the parked die is spent by THIS batch
                    for sd, f in toks:
                        ev.append(('unpark', sd, f, True))
                elif _ca(frm) and 'hunt for the ring' in str(to).lower():
                    # A paid Fellowship die stays spent, but is no longer
                    # staged for a later free Ring conversion.
                    for sd, f in toks:
                        ev.append(('close_stage', sd, f, None))
                continue
            if op == 'kill':
                zn = r.get('zone', '')
                for p in r.get('pieces', []) or []:
                    for sd, f in _die_tokens(str(p)):
                        if _dz(zn):
                            # killed in its pool (The Day Without Dawn):
                            # removed from play, never spent
                            ev.append(('leave', sd, f, (False, False)))
                        else:
                            # a parked die killed: spent when it is the
                            # acting side's own die, removed when the
                            # opponent's card took it
                            ev.append(('unpark', sd, f,
                                       sd == b.get('die_side')))
                continue
            if op == 'other':
                mc = _DIE_CHANGE_RE.match(str(r.get('raw', '')))
                if mc:
                    sd = 'FREE_PEOPLES' if mc.group(1).lower() == 'free' \
                        else 'SHADOW'
                    if lead_lines is None:
                        lead_lines = _lead_change_lines(b)[1]
                    to_staged = None
                    if sd == b.get('die_side'):
                        to_staged = int(r.get('line', -1)) in lead_lines
                    ev.append(('change', sd, mc.group(2).upper(),
                               (mc.group(3).upper(), to_staged)))
    # ---- pass 2: simulate the zones per turn --------------------------
    SIDES = ('FREE_PEOPLES', 'SHADOW')
    for t in order:
        pool = {s: _C() for s in SIDES}
        out = {s: [] for s in SIDES}
        parked = {s: _C() for s in SIDES}     # on a card zone, fate open
        staged = {s: _C() for s in SIDES}     # in Current Action now
        rolled = {s: False for s in SIDES}
        lowest = {s: _C() for s in SIDES}     # running minimum per face
        snaps = []                # (batch idx, {side: (pool, out, parked)})

        def _track(s):
            for f, n in pool[s].items():
                if n < lowest[s][f]:
                    lowest[s][f] = n

        for kind, sd, f, ex in turns[t]:
            if kind in ('snap', 'snap_record'):
                snaps.append((kind, ex, {s: (_C(pool[s]), list(out[s]),
                                       _C(parked[s]), _C(staged[s])) for s in SIDES}))
                continue
            if kind == 'partial_roll_staged':
                new, zone = ex
                bucket = staged[sd] if zone == 'Game.2035' else parked[sd]
                if bucket[f] <= 0:
                    raise _Quarantine('coverage_dwarven_ring_unproven',
                                      {'detail': 'reroll_original_not_in_proven_zone'})
                bucket[f] -= 1
                bucket[new] += 1
                if zone == 'Game.2035':
                    if f not in out[sd]:
                        raise _Quarantine('coverage_dwarven_ring_unproven',
                                          {'detail': 'reroll_staged_spend_missing'})
                    out[sd][len(out[sd]) - 1 - out[sd][::-1].index(f)] = new
                continue
            if kind == 'partial_roll':
                if pool[sd][f] <= 0:
                    raise _Quarantine('coverage_dwarven_ring_unproven',
                                      {'detail': 'reroll_original_not_unused'})
                pool[sd][f] -= 1
                pool[sd][ex] += 1
                continue
            if kind == 'roll':
                pool[sd] = _C(f)
                out[sd] = []
                parked[sd] = _C()
                staged[sd] = _C()
                rolled[sd] = True
                lowest[sd] = _C()
                continue
            if kind == 'leave':
                book, to_ca = ex
                pool[sd][f] -= 1
                if book and f != 'E':
                    out[sd].append(f)
                if to_ca:
                    staged[sd][f] += 1
                _track(sd)
                continue
            if kind == 'park':
                pool[sd][f] -= 1
                parked[sd][f] += 1
                _track(sd)
                continue
            if kind == 'close_stage':
                if staged[sd][f] > 0:
                    staged[sd][f] -= 1
                continue
            if kind == 'unpark':
                if parked[sd][f] > 0:
                    parked[sd][f] -= 1
                    if ex and f != 'E':
                        out[sd].append(f)
                        staged[sd][f] += 1
                elif staged[sd][f] > 0:
                    staged[sd][f] -= 1     # the staged die spent (killed)
                continue
            if kind == 'enter':
                pool[sd][f] += 1
                if parked[sd][f] > 0:
                    parked[sd][f] -= 1
                elif f in out[sd]:
                    del out[sd][len(out[sd]) - 1 - out[sd][::-1].index(f)]
                if ex and staged[sd][f] > 0:
                    staged[sd][f] -= 1
                continue
            if kind == 'change':
                ex, to_staged = ex
                if staged[sd][f] > 0 and to_staged is not False:
                    # the die in Current Action is the one re-faced
                    # (elven ring / Mirror on the staged die); a change
                    # the batch's own spend gesture proves was a POOL
                    # die's (to_staged False) falls through
                    staged[sd][f] -= 1
                    if ex != 'E':
                        staged[sd][ex] += 1
                    if f in out[sd]:
                        j = len(out[sd]) - 1 - out[sd][::-1].index(f)
                        if ex == 'E':
                            del out[sd][j]
                        else:
                            out[sd][j] = ex
                elif pool[sd][f] > 0:
                    pool[sd][f] -= 1
                    pool[sd][ex] += 1
                elif f in out[sd]:
                    j = len(out[sd]) - 1 - out[sd][::-1].index(f)
                    if ex == 'E':
                        del out[sd][j]
                    else:
                        out[sd][j] = ex
                else:
                    if not rolled[sd]:
                        pool[sd][f] -= 1
                        _track(sd)
                    pool[sd][ex] += 1
                continue
        # finalise: without a roll the initial pool is what the turn's own
        # gestures required (deficit), so every snapshot is lifted by it
        for snap_kind, target, sides in snaps:
            bi, record = target if snap_kind == 'snap_record' else (target, None)
            b = batches[bi]
            rem = {}
            for s in SIDES:
                if record is not None and record.get('_rx_before_hunt_allocation'):
                    rem[s] = {'unused': [], 'used': []}
                    continue
                pc, oc, pk, staged_at_record = sides[s]
                if not rolled[s]:
                    for ff, n in lowest[s].items():
                        if n < 0:
                            pc[ff] -= n
                unused = sorted(x for x in pc.elements() if x != 'E')
                used = list(oc)
                if record is None and s == b.get('die_side') and b.get('die_face'):
                    bl = str(b['die_face']).upper()
                    if bl not in unused and pk.get(bl, 0) > 0:
                        # staged through a card zone: the die is the
                        # batch's own, still unused at the decision
                        unused = sorted(unused + [bl])
                    elif bl not in unused and bl in used:
                        used.remove(bl)
                        unused = sorted(unused + [bl])
                used = sorted(used)
                if s == 'SHADOW':
                    try:
                        used = used + ['E'] * int(eye_counts.get(t, 0) or 0)
                    except Exception:
                        pass
                rem[s] = {'unused': unused, 'used': used}
            (record if record is not None else b)['_rx_pools'] = rem
            if record is not None:
                record['_rx_staged'] = {s: sorted(sides[s][3].elements()) for s in SIDES}
    # any batch never snapshotted (no staging record at its start line)
    # keeps the empty-pool shape so consumers stay well-formed
    for b in batches:
        if '_rx_pools' not in b:
            b['_rx_pools'] = {s: {'unused': [], 'used': []} for s in SIDES}


_DIE_TOKEN_RE = re.compile(
    r'(?:\[(\d+)\])?(Free|Shadow)\s+Action\s+Die\s+\(([A-Z])\)', re.I)
_DIE_CHANGE_RE = re.compile(
    r'^\S+\s+changes\s+(Free|Shadow)\s+Action\s+Die\s+\(([A-Z])\)\s+to\s+'
    r'([A-Z])\b', re.I)


def _die_tokens(text):
    """[(side_str, face_letter)] for every action die named in a rendered
    piece/group string, group multipliers honoured ('[2]Free Action Die
    (W)' = two W dice).  W4-018 zone simulation input."""
    toks = []
    for m in _DIE_TOKEN_RE.finditer(str(text or '')):
        n = int(m.group(1) or 1)
        sd = 'FREE_PEOPLES' if m.group(2).lower() == 'free' else 'SHADOW'
        toks.extend([(sd, m.group(3).upper())] * n)
    return toks


def _lead_change_lines(batch):
    """W4-018: which in-batch 'changes X to Y' lines (batch side) re-face
    the batch's STAGED (lead) die.  The rendered line names no zone, so a
    change whose X equals the running lead face is ambiguous when the pool
    holds a second X (Mirror of Galadriel P->W while P is staged,
    2023-02-10 L386-390).  Disambiguated by the batch's own spend
    gesture: the kill (or retraction to the pool) of the batch side's die
    in Current Action names the lead's FINAL face.  Returns
    (final_letter, set of line numbers applied to the lead); a change
    naming another face than the running lead face is a pool die's."""
    side = batch.get('die_side')
    start = str(batch.get('die_face') or '').upper()
    changes = []
    kill_face = None

    def _lo(z):
        return str(z or '').strip().lower()

    for r in batch.get('records', []) or []:
        op = r.get('op')
        if op == 'other':
            m = _DIECHANGE_PARSE_RE.search(str(r.get('raw', '')))
            if m and (('SHADOW' if m.group(1).lower() == 'shadow'
                       else 'FREE_PEOPLES') == side):
                changes.append((int(r.get('line', -1)),
                                m.group(2).upper(), m.group(3).upper()))
        elif op == 'kill' and _lo(r.get('zone')) == 'current action':
            for sd, f in _die_tokens(' '.join(
                    str(x) for x in (r.get('pieces', []) or []))):
                if sd == side:
                    kill_face = f
        elif op == 'move' and _lo(r.get('from')) == 'current action' \
                and _lo(r.get('to')) in ('sa dice', 'fp dice'):
            for sd, f in _die_tokens(r.get('obj', '')):
                if sd == side:
                    kill_face = f
    if not changes:
        return start, set()
    n = len(changes)
    for mask in range((1 << n) - 1, -1, -1):
        # all-applied first (the default chain), then fewer
        face, applied, ok = start, set(), True
        for k, (ln, x, y) in enumerate(changes):
            if (mask >> k) & 1:
                if x != face:
                    ok = False
                    break
                face = y
                applied.add(ln)
        if ok and (kill_face is None or face == kill_face):
            return face, applied
    face, applied = start, set()
    for ln, x, y in changes:
        if x == face:
            face = y
            applied.add(ln)
    return face, applied


def _annotate_ring_ledger(batches):
    """True elven-ring inventories through the game (rulebook p.20: FP
    starts with Narya/Nenya/Vilya; an FP use flips the counter to the
    Shadow box, a Shadow use removes it from play; one use per side per
    turn, 6 lifetime uses max).  Identity-set ledger (a duplicate render
    of the same counter move cannot double-decrement).  Stamps
    batch['_rx_rings'] = {'pre','post'} and writes 'post' into both
    truth snapshots â€” the primary die-spend decision follows any
    in-batch ring gesture; the elven_ring emitter overrides with 'pre'
    (the use decision itself still holds the ring).  Returns audit
    counters for the per-game manifest."""
    ring_st = {nm: 'FP' for nm in _RING_NAMES}
    # per-ring states: FP (Elven Rings box) / SH (Dark Ring box) /
    # LIMBO (FP-use staged through a Current/Board slot, pass to the
    # Shadow box not yet rendered) / GONE (removed from play).
    # Dialects: FP use = Elven Ring slot -> anything (classic Dark Ring
    # drop, or Current-slot staging later swept by a kill and completed
    # by a Casualties -> Dark Ring drag); SH use = Dark Ring slot ->
    # anything non-ring (classic Casualties drop, or Current-slot
    # staging + kill); backward moves are take-backs and uncount.
    fp_used = sh_used = False
    prev_turn = None
    audit = {'fp_uses': 0, 'sh_uses': 0, 'anom_fp': 0, 'anom_sh': 0,
             'anom_turn_dup': 0}

    def _snap9():
        vals = list(ring_st.values())
        return {'fp': vals.count('FP'), 'sh': vals.count('SH'),
                'fp_used': fp_used, 'sh_used': sh_used}

    for b in batches:
        if b['turn'] != prev_turn:
            prev_turn = b['turn']
            fp_used = sh_used = False
        pre = _snap9()
        for r in b['records']:
            if isinstance(r.get('decision_truth'), dict):
                r['decision_truth']['_rx_rings'] = _snap9()
            if r.get('op') == 'kill' and \
                    'dark ring' in str(r.get('zone', '')).lower():
                # W3-016: 'kills Narya in Dark Ring N' = a Shadow use
                # rendered as a kill instead of a Casualties drop
                for nm in [str(x).strip().lower()
                           for x in (r.get('pieces') or [])]:
                    if nm not in _RING_NAMES:
                        continue
                    if ring_st[nm] == 'SH':
                        audit['sh_uses'] += 1
                    else:
                        audit['anom_sh'] += 1
                    ring_st[nm] = 'GONE'
                    if sh_used:
                        audit['anom_turn_dup'] += 1
                    sh_used = True
                continue
            if r.get('op') != 'move':
                continue
            nm = str(r.get('obj', '')).strip().lower()
            if nm not in _RING_NAMES:
                continue
            frm = str(r.get('from', '')).lower()
            to = str(r.get('to', '')).lower()
            f_er, f_dr = 'elven ring' in frm, 'dark ring' in frm
            t_er, t_dr = 'elven ring' in to, 'dark ring' in to
            s9 = ring_st[nm]
            if f_er and not t_er:                    # FP use
                if s9 == 'FP':
                    audit['fp_uses'] += 1
                else:
                    audit['anom_fp'] += 1
                ring_st[nm] = 'SH' if t_dr else 'LIMBO'
                if fp_used:
                    audit['anom_turn_dup'] += 1
                fp_used = True
            elif f_dr and t_er:                      # FP-use take-back
                if s9 == 'SH':
                    audit['fp_uses'] = max(0, audit['fp_uses'] - 1)
                    ring_st[nm] = 'FP'
                    fp_used = False
            elif f_dr and not t_dr:                  # SH use
                if s9 == 'SH':
                    audit['sh_uses'] += 1
                else:
                    audit['anom_sh'] += 1
                ring_st[nm] = 'GONE'
                if sh_used:
                    audit['anom_turn_dup'] += 1
                sh_used = True
            elif t_dr and not f_dr and not f_er:     # completion/take-back
                if s9 == 'LIMBO':
                    ring_st[nm] = 'SH'               # FP pass completes
                elif s9 == 'GONE':
                    audit['sh_uses'] = max(0, audit['sh_uses'] - 1)
                    ring_st[nm] = 'SH'
                    sh_used = False
            elif t_er and not f_er and not f_dr:     # back to the FP box
                if s9 == 'LIMBO':
                    audit['fp_uses'] = max(0, audit['fp_uses'] - 1)
                    ring_st[nm] = 'FP'
                    fp_used = False
        post = _snap9()
        b['_rx_rings'] = {'pre': pre, 'post': post}
        for kk in ('truth_before', 'truth'):
            sn = b.get(kk)
            if isinstance(sn, dict):
                sn['_rx_rings'] = post
    return audit


def _annotate_ring_carry(batches):
    """W3-005 (2026-09-02): PRE-STAGING RING USE.  Batches are cut at
    the die-stage line, so a ring counter dragged BEFORE its die is
    staged is homed in the OPPONENT's batch, after that batch's own die
    kill (2024-05-20 L1001 Narya after the Shadow M die's kill L1000,
    the FP M die staged at L1002 - a bare Mordor hide; 2024-08-11 L65
    and L950; 2023-01-24 L332).  There W4-018 rightly refuses it (the
    other side's counter never re-faces the Shadow lead) and the
    owner's own batch shows no ring at all, so the ring row AND the
    converted action were lost.  For every batch whose immediate
    predecessor is the other side's, holds no ring gesture of the
    batch's own side, and whose predecessor's records AFTER its last
    action-die kill carry the batch side's ring USE (not taken back in
    that tail), stamp batch['_rx_ring_carry'] = those records and
    restore the decision-time inventory ('pre') of the carried side to
    the predecessor's (the ring still in hand when the use was
    decided).  _ring_counter_used reads the carry; the predecessor
    keeps its records (its own die side still wins there).  Returns
    the number of carries."""
    n = 0
    for i in range(1, len(batches)):
        b, p = batches[i], batches[i - 1]
        if p.get('die_side') == b.get('die_side'):
            continue
        own = 'sh' if b.get('die_side') == 'SHADOW' else 'fp'
        if any((_ring_rec_kind(r) or '').startswith(own)
               for r in b['records']):
            continue
        kill_ln = None
        for r in p['records']:
            if r.get('op') == 'kill' and any(
                    'Action Die' in str(x) for x in (r.get('pieces') or ())):
                kill_ln = r.get('line')
        if kill_ln is None:
            continue
        carry, used = [], False
        for r in p['records']:
            if (r.get('line') or 0) <= kill_ln:
                continue
            k = _ring_rec_kind(r)
            if k == own + '_use':
                used = True
                carry.append(r)
            elif k == own + '_back':
                used = False
                carry = []
        if not (used and carry):
            continue
        b['_rx_ring_carry'] = carry
        # Keep intervening card gestures too: an explicit change in the next
        # batch must not jump backward across a card to attribute its payment.
        b['_rx_ring_carry_context'] = [r for r in p['records']
                                       if (r.get('line') or 0) > kill_ln]
        p['_rx_ring_carried_out'] = True
        rg_b, rg_p = b.get('_rx_rings'), p.get('_rx_rings')
        if isinstance(rg_b, dict) and isinstance(rg_p, dict):
            pre = dict(rg_b['pre'])
            pre[own] = rg_p['pre'].get(own, pre.get(own))
            pre[own + '_used'] = rg_p['pre'].get(own + '_used', False)
            rg_b['pre'] = pre
        n += 1
    return n


def _stamp_rings(st, rg):
    """Write a ring-ledger snapshot onto a reconstructed state."""
    st.fp_elven_rings = int(rg['fp'])
    st.sh_elven_rings = int(rg['sh'])
    st.fp_ring_used_this_turn = bool(rg['fp_used'])
    st.sh_ring_used_this_turn = bool(rg['sh_used'])
    st._rx_rings_stamped = True


# W1-025: the two FP hero flags the engine never clears.
_HERO_FLAG_NAMES = (('gtw', 'Gandalf the White'), ('crown', 'Aragorn'))


def _hero_rec_names(rec):
    """Canonical named-figure names (rendered_replay.COMP_ALIAS values)
    touched by one move / kill / flip record: the flip object, a single
    move object, every token of a group move, every killed piece."""
    import rendered_replay as _rr9h
    out = set()
    toks = []
    obj = str(rec.get('obj') or '')
    mg = re.match(r'^group\s*\((.*)\)\s*$', obj.strip(), re.I)
    if mg:
        toks.extend(_rr9h.split_group(mg.group(1)))
    elif obj:
        toks.append(obj)
    toks.extend(rec.get('pieces') or [])
    for t in toks:
        p = _rr9h.parse_piece(t)
        if p and p[0] == 'named':
            out.add(p[1])
    return out


def _annotate_hero_flags(batches):
    """Stamp every batch with the FP hero flags at its decision point:
    batch['_rx_heroes'] = {'pre': {'gtw', 'crown'}, 'post': {...}}, 'pre'
    written into truth_before and 'post' into truth.  A flag is True once
    the figure has appeared in the log (a 'flips Gandalf the White' /
    'flips Aragorn' gesture in ANY zone - Fangorn, a stronghold, the
    Current FP Card slot or FP Casualties - or, as a backstop, the first
    move / kill naming the figure) at a line BEFORE the snapshot point;
    the transformation / crowning batch itself therefore still sees the
    flag False at truth_before (the transform / crown menu stays legal
    for the gesture that IS the decision).  Mirrors the engine, which
    sets each flag once and never clears it (game_logic 4166 / 4249).
    Returns {'gtw_line', 'crown_line'} for the per-game manifest."""
    first = {k: None for k, _ in _HERO_FLAG_NAMES}
    for b in batches:
        for rec in b.get('records') or []:
            if rec.get('op') not in ('flip', 'move', 'kill'):
                continue
            try:
                ln = int(rec.get('line') or 0)
            except (TypeError, ValueError):
                continue
            if ln <= 0:
                continue
            names = _hero_rec_names(rec)
            for key, nm in _HERO_FLAG_NAMES:
                if nm in names and (first[key] is None or ln < first[key]):
                    first[key] = ln
    for b in batches:
        s0 = int(b.get('start_line') or 0)
        e0 = int(b.get('end_line') or 0)
        pre = {k: bool(first[k] is not None and first[k] < s0)
               for k, _ in _HERO_FLAG_NAMES}
        post = {k: bool(first[k] is not None and first[k] < e0)
                for k, _ in _HERO_FLAG_NAMES}
        b['_rx_heroes'] = {'pre': pre, 'post': post}
        sn = b.get('truth_before')
        if isinstance(sn, dict):
            sn['_rx_heroes'] = pre
        sn = b.get('truth')
        if isinstance(sn, dict):
            sn['_rx_heroes'] = post
        for rec in b.get('records') or []:
            sn = rec.get('decision_truth')
            if isinstance(sn, dict):
                line = int(rec.get('line') or 0)
                sn['_rx_heroes'] = {
                    k: bool(first[k] is not None and first[k] < line)
                    for k, _ in _HERO_FLAG_NAMES}
    return {'gtw_line': first['gtw'], 'crown_line': first['crown']}


def _stamp_hero_flags(st, flags, names=None):
    """Set gandalf_the_white.in_play / aragorn_crowned.crowned on a
    reconstructed state: the figure is on the board now (a named leader
    of any army) OR the ledger stamp says it appeared earlier in the log
    (`flags` = the batch's {'gtw','crown'} dict; None when the snapshot
    was not produced by extract_one -> board presence only).  The engine
    never clears either flag: a fallen Gandalf the White keeps in_play
    (only the bonus die is gated by _fp_alive), so the stamp is the only
    witness for the FP-casualties case."""
    if names is None:
        names = set()
        for _a in getattr(st, 'armies', {}).values():
            for _l in getattr(_a, 'leaders', ()) or ():
                _n = getattr(_l, 'name', None)
                if _n:
                    names.add(str(_n).strip().lower())
    flags = flags if isinstance(flags, dict) else {}
    gw = getattr(st, 'gandalf_the_white', None)
    if gw is not None and ('gandalf the white' in names or flags.get('gtw')):
        gw.in_play = True
    ac = getattr(st, 'aragorn_crowned', None)
    if ac is not None and ('aragorn' in names or flags.get('crown')):
        ac.crowned = True
    st._rx_heroes_stamped = True


# W1-011: FP bid-compensation action tokens.  The module renders them as
# die-like pieces 'Hunt for the ring (Political)' / '(Card)' granted to
# the FP Dice box at setup and USED by staging one into Current Action
# (or 'Current FP Card'), taking the effect (a political advance / a
# deck draw) and killing the piece.  The census snapshot carries no
# token model, so until now every reconstructed state read 'no tokens':
# the use_token options were absent from every menu while a token was
# held (a 100% prior on the rare use rows) and the token emitter had
# to fake the holding of the one token it was emitting.
_TOKEN_PIECE_RE = re.compile(
    r'hunt for the ring \((political|card(?: draw)?)\)', re.I)


def _token_kinds(text):
    """['political' | 'card', ...] for every token piece named in a
    record field (a single piece, a 'group (...)' move or a kill list)."""
    if isinstance(text, (list, tuple)):
        text = ' '.join(str(x) for x in text)
    return ['political' if m.lower().startswith('pol') else 'card'
            for m in _TOKEN_PIECE_RE.findall(str(text or ''))]


def _annotate_token_ledger(batches):
    """Stamp every batch with the FP token holdings at its decision point:
    batch['_rx_tokens'] = {'pre': {'political': n, 'card': n, 'used':
    bool}, 'post': {...}, 'grants': [(line, kind)], 'uses': [(stage_line,
    turn, kind)]}; 'pre' is written into truth_before and 'post' into
    truth.  A grant = a token piece moved INTO the FP Dice box (setup, or
    a mid-game correction); a use = the piece staged from FP Dice into a
    Current slot and killed there (a stage returned to FP Dice is a
    retraction, not a use).  The use is dated at its STAGE line (the
    decision), so the batch that hosts the use still sees the token held
    at truth_before.  'used' = a use earlier in the same turn (engine:
    max one token action per side per turn).  Returns the audit dict."""
    grants, uses = [], []
    staged = {}                               # kind -> stage line
    recs = []
    for b in batches:
        for rec in b.get('records') or []:
            try:
                ln = int(rec.get('line') or 0)
            except (TypeError, ValueError):
                continue
            if ln > 0:
                recs.append((ln, b.get('turn') or 1, rec))
    recs.sort(key=lambda x: x[0])
    for ln, turn, rec in recs:
        op = rec.get('op')
        if op == 'move':
            kinds = _token_kinds(rec.get('obj'))
            if not kinds:
                continue
            frm = str(rec.get('from', '')).lower()
            to = str(rec.get('to', '')).lower()
            if 'fp dice' in to and 'current' not in frm:
                for k in kinds:
                    grants.append((ln, k))
            elif 'current' in to and 'fp dice' in frm:
                for k in kinds:
                    staged[k] = (ln, turn)
            elif 'fp dice' in to and 'current' in frm:
                for k in kinds:                   # retraction
                    staged.pop(k, None)
        elif op == 'kill':
            kinds = _token_kinds(rec.get('pieces') or rec.get('body'))
            if not kinds or 'current' not in str(rec.get('zone', '')).lower():
                continue
            for k in kinds:
                sl, st9 = staged.pop(k, (ln, turn))
                uses.append((sl, st9, k))
    uses.sort()

    def _at(line, turn):
        held = {'political': 0, 'card': 0}
        for gl, k in grants:
            if gl < line:
                held[k] += 1
        used = False
        for ul, ut, k in uses:
            if ul < line:
                held[k] -= 1
                if ut == turn:
                    used = True
        return {'political': max(0, held['political']),
                'card': max(0, held['card']), 'used': used}

    for b in batches:
        s0 = int(b.get('start_line') or 0)
        e0 = int(b.get('end_line') or 0)
        t0 = b.get('turn') or 1
        pre = _at(s0, t0)
        post = _at(e0, t0)
        b['_rx_tokens'] = {'pre': pre, 'post': post,
                           'grants': list(grants), 'uses': list(uses)}
        sn = b.get('truth_before')
        if isinstance(sn, dict):
            sn['_rx_tokens'] = pre
        sn = b.get('truth')
        if isinstance(sn, dict):
            sn['_rx_tokens'] = post
        for rec in b.get('records') or []:
            sn = rec.get('decision_truth')
            if isinstance(sn, dict):
                sn['_rx_tokens'] = _at(int(rec.get('line') or 0), t0)
    return {'token_grants': len(grants), 'token_uses': len(uses),
            'token_use_lines': [u[0] for u in uses]}


def _token_holdings_at(batch, line):
    """Holdings at an arbitrary line of an annotated batch (the token
    emitter's stage line): None when the ledger did not run."""
    tk = batch.get('_rx_tokens')
    if not isinstance(tk, dict) or 'grants' not in tk:
        return None
    turn = batch.get('turn') or 1
    held = {'political': 0, 'card': 0}
    for gl, k in tk['grants']:
        if gl < line:
            held[k] += 1
    used = False
    for ul, ut, k in tk['uses']:
        if ul < line:
            held[k] -= 1
            if ut == turn:
                used = True
    return {'political': max(0, held['political']),
            'card': max(0, held['card']), 'used': used}


def _stamp_tokens(st, tk):
    """Write a ledger holdings dict onto a reconstructed state (engine
    model: two FP booleans + the once-per-turn latch)."""
    tk = tk if isinstance(tk, dict) else {}
    st.fp_political_token = bool(int(tk.get('political') or 0) > 0)
    st.fp_card_token = bool(int(tk.get('card') or 0) > 0)
    st.fp_token_used_this_turn = bool(tk.get('used'))
    st._rx_tokens_stamped = True


def _install_full_pools(pools_letters):
    """Build {Side: [ActionDie]} from the pool annotation.  v31 shape:
    {side: {'unused': [letters], 'used': [letters]}} â€” used faces
    install as used=True dice (Eyes included: hunt-box evidence);
    legacy flat-list shape still accepted."""
    import game_state as gs
    l2f = _letter2face()
    out = {gs.Side.FREE_PEOPLES: [], gs.Side.SHADOW: []}
    for side, key in ((gs.Side.FREE_PEOPLES, 'FREE_PEOPLES'),
                      (gs.Side.SHADOW, 'SHADOW')):
        info = (pools_letters or {}).get(key)
        if isinstance(info, dict):
            groups = ((info.get('unused') or (), False),
                      (info.get('used') or (), True))
        elif info is None:
            groups = (((), False),)
        else:
            # only the v31 {'unused','used'} dict shape exists â€”
            # anything else is corrupt input (loud, never a guess)
            raise ValueError('non-v31 pool shape: %r' % (info,))
        for letters, used in groups:
            for L in letters:
                f = l2f.get(L)
                if f is None:
                    continue
                if f is gs.ActionDieFace.EYE and not used:
                    continue
                d = gs.ActionDie(gs.SH_FACES if side is gs.Side.SHADOW
                                 else gs.FP_FACES, side)
                d.current = f
                d.used = used
                out[side].append(d)
    return out


def _pend_note(stats, fam, kind):
    """EMISSION COMPLETENESS instrument (DET census 2026-08-30): every
    pending the replay OPENS with a real (>=2-option) menu is tallied
    per family - kind in ('open', 'emit', 'auto').  auto = resolved by
    a standing ruling (never a decision).  extract_one publishes the
    per-game table (line['pend_table']); _print_agg prints the corpus
    table at extraction end: opened / emitted / auto / dropped."""
    stats.setdefault('pend_' + kind, Counter())[fam] += 1


def _emit_observed_combat_sample(st, side, legal, hidx, turn, evidence):
    """Bind validated chronological evidence for the central v52 admission gate."""
    import copy
    missing = object()
    previous = getattr(st, '_rx_combat_choice_evidence', missing)
    st._rx_combat_choice_evidence = copy.deepcopy(evidence)
    try:
        return _emit_sample(st, side, legal, hidx, 'combat_effect_choice', turn)
    finally:
        if previous is missing:
            del st._rx_combat_choice_evidence
        else:
            st._rx_combat_choice_evidence = previous


def _emit_sample(st, side, legal, hidx, phase, turn):
    global _RX_ALT_HIDX, _RX_ALIAS_V41
    _alias9 = bool(_RX_ALIAS_V41)      # W4-032: consume + clear FIRST
    _RX_ALIAS_V41 = None
    _alt9 = sorted(set(int(i) for i in (_RX_ALT_HIDX or ())
                       if 0 <= int(i) < len(legal) and int(i) != int(hidx)))
    _RX_ALT_HIDX = None
    from rendered_casualty_groups import indices as grouped_indices
    _alt9 = sorted(set(_alt9 + grouped_indices(st, legal, hidx, phase)) - {int(hidx)})
    if _RING_PROBE is not None:
        # W3-016 face probe: record what the matcher bound (phase, action,
        # die index) and return a stub - no encoding, no census counters
        try:
            _a9 = legal[hidx]
            _RING_PROBE[1].append((phase, _a9.action_type, _a9.die_idx))
        except Exception:
            _RING_PROBE[1].append((phase, None, None))
        return {'side': side, 'x': None, 'encs': None, 'regs': None,
                'human': hidx, 'phase': phase, 'turn': turn, 'z': None,
                'phi': None, '_rx_probe': True}
    import ai
    from game_state import Side
    from rendered_history import HistoryUnproven
    deck_refusal = None
    try:
        _ensure_decks(st)
    except HistoryUnproven as error:
        deck_refusal = str(error)
    # Project only AI training candidates; the legal replay above still
    # follows the actual human's actions, including legal policy deviations.
    from ai_policies import POLICY_VERSION, project_ai_actions, project_recruitment_target
    projection = project_ai_actions(st, side, legal)
    identities = {id(action) for action in projection.actions}
    kept_indices = [i for i, action in enumerate(legal) if id(action) in identities]
    policy_exclusion = projection.forced_reason
    if deck_refusal is not None:
        policy_exclusion = deck_refusal
    target_projection = None
    combat_refusal = None
    combat_evidence = getattr(st, '_rx_combat_choice_evidence', None)
    shine_evidence = getattr(st, '_rx_shine_evidence', None)
    shine_refusal = None
    if phase == 'fp_shine':
        shine_refusal = ('shine_wrong_actor' if side != Side.FREE_PEOPLES else
                         _shine_evidence_refusal(st, shine_evidence, legal[hidx]))
        if shine_refusal is not None and policy_exclusion is None:
            policy_exclusion = shine_refusal
    if phase == 'combat_effect_choice' and getattr(ai, '_COMBAT_ROUND_CONTEXT_ON', False):
        from combat_choice_evidence import admission_refusal
        combat_refusal = admission_refusal(st, combat_evidence, legal[hidx])
        if combat_refusal is not None and policy_exclusion is None:
            policy_exclusion = combat_refusal
    if policy_exclusion is None and hidx not in kept_indices:
        projected = project_recruitment_target(legal, hidx, projection)
        if projected is not None:
            target_index, target_projection = projected
            target_projection['source_menu_index'] = int(hidx)
            target_projection['target_menu_index'] = int(target_index)
            hidx = target_index
            _alt9 = []
        else:
            policy_exclusion = 'expert_action_outside_fixed_policy'
    if policy_exclusion is None:
        remap = {old: new for new, old in enumerate(kept_indices)}
        hidx = remap[hidx]
        _alt9 = [remap[i] for i in _alt9 if i in remap]
        legal = list(projection.actions)
    # PRIOR CENSUS (USER 2026-08-20): how many emitted decisions had a
    # 1.149 removal available but did something else.  Without this the
    # yield number alone cannot say whether the prior is honest.
    try:
        if _RX_STATS is not None and any(
                a.action_type == 'discard_table_card' for a in legal):
            _RX_STATS['detected']['tablekill_in_menu'] += 1
    except Exception:
        pass
    # skew-gate P0 backstop: the batch turn is authoritative for EVERY
    # emitted state - combat-family emitters ride census-driver-internal
    # states that never see _recon_state's threading.  No menu
    # generator reads state.turn (verified), so stamping here cannot
    # skew the already-enumerated legal list.
    try:
        if turn:
            st.turn = max(1, int(turn))
    except (TypeError, ValueError):
        pass
    # ring-inventory backstop (encoding-only; the menu was already
    # enumerated) for states that saw neither _recon_state nor
    # _install_ledger_hands
    if _RX_RINGS_CUR is not None and \
            not getattr(st, '_rx_rings_stamped', False):
        _stamp_rings(st, _RX_RINGS_CUR)
    # P0-a backstop: ledger discard piles for states that never saw
    # _install_ledger_hands (resolver-path drive states).  Encoding-only
    # - the menu was already enumerated, and no menu generator reads
    # discard_face_up (see _install_ledger_discards).
    if _RX_LINE_CUR is not None and \
            not getattr(st, '_rx_dfu_stamped', False):
        _install_ledger_discards(st, _RX_LINE_CUR)
    # These ledger entries describe THIS turn's later allocation and roll.
    # Rulebook p.18: declarations and turn-start guide choices happen before
    # both. Do not give an earlier training decision future information.
    before_hunt = (phase in ('eye_allocate', 'fellowship_declare')
                   or getattr(st, '_declare_phase', False)
                   or getattr(st, '_eye_allocation_phase', False)
                   or getattr(st, '_rx_before_hunt_allocation', False))
    if _RX_EYEALLOC is not None and not before_hunt and \
            not getattr(st, '_rx_eyes_stamped', False):
        import game_state as _gse
        _t9 = 1
        try:
            _t9 = max(1, int(turn or getattr(st, 'turn', 1) or 1))
        except (TypeError, ValueError):
            pass
        _alloc9 = int(_RX_EYEALLOC.get(_t9, 0) or 0)
        _roll9 = int((_RX_EYEROLL or {}).get(_t9, 0) or 0)
        if (_alloc9 or _roll9) and not getattr(st, 'hunt_box', None):
            _box9 = []
            for _ in range(_alloc9 + _roll9):
                _d9 = _gse.ActionDie(_gse.SH_FACES, _gse.Side.SHADOW)
                _d9.current = _gse.ActionDieFace.EYE
                _box9.append(_d9)
            st.hunt_box = _box9
        st.eyes_allocated = _alloc9
        st._rx_eyes_stamped = True
    # v30: encode the STATE under the reconstructed full dice pools
    # (realistic play-time context); the menu was already enumerated
    # (v31: under the full marginal pool â€” every unspent face), so
    # emission stays byte-identical.
    from rendered_hunt_observation import encoding_pool
    hunt_evidence = hunt_refusal = None
    hunt_pool = hunt_discard = None
    if _RX_HUNT_HISTORY is not None:
        hunt_pool, hunt_discard, hunt_evidence, hunt_refusal = \
            _RX_HUNT_HISTORY.observation(st, phase, legal[hidx])
    with encoding_pool(st, hunt_pool, hunt_discard):
        _fullp = getattr(st, '_rx_full_pools', None)
        if (_fullp is not None
                and getattr(st, '_fellowship_move_die_pending', None) is None):
            _saved_dice = st.dice
            st.dice = _install_full_pools(_fullp)
            try:
                x = ai.encode_state(st, side)
            finally:
                st.dice = _saved_dice
        else:
            x = ai.encode_state(st, side)
    encs = [ai.encode_action(a, st, side) for a in legal]
    regs = [ai._get_action_region_index(a) for a in legal]
    # CURRENCY COHERENCE (USER root-cause order 2026-08-10): every
    # sample carries its side-signed profile-shaping phi so the BC /
    # value-td stages can train the RESIDUAL value (z - phi) â€” the
    # currency search consumes (net+phi) and the TD loop trains
    # (G - phi).  Sign conventions derive from ai's OWN functions
    # (one-implementation doctrine): phi_sample = sgn * _profile_
    # shaping(state), sgn=+1 fp / -1 sh â€” exactly _replay_reward_
    # stream / _mcts_eval_state.  Shipping profile pair pinned
    # (fp=ring, sh=military) for the phi evaluation only; encoding
    # and menus never read profiles.
    _old_prof9 = dict(ai.PLAYER_PROFILES)
    try:
        ai.PLAYER_PROFILES['fp'] = 'ring'
        ai.PLAYER_PROFILES['sh'] = 'military'
        _sgn9 = 1.0 if side == Side.FREE_PEOPLES else -1.0
        _phi9 = _sgn9 * float(ai._profile_shaping(st))
    except Exception as _e9:
        # LOUD by doctrine: a silent phi=0 sample is wrong-currency
        # data â€” the exact incoherence the residual order abolishes
        raise ValueError('phi computation failed (currency '
                         'coherence): %r' % _e9)
    finally:
        ai.PLAYER_PROFILES.update(_old_prof9)
    if not (-0.5 <= _phi9 <= 0.5):        # sanity: |phi| bound
        raise ValueError('phi out of bound: %r' % _phi9)
    _smp9 = {'side': 'fp' if side == Side.FREE_PEOPLES else 'sh',
             'x': np.asarray(x, dtype=np.float32),
             'encs': np.asarray(encs, dtype=np.float32),
             'regs': np.asarray(regs, dtype=np.int64),
             'human': int(hidx), 'phase': phase, 'turn': int(turn),
             'z': 0, 'phi': float(_phi9), 'alt': _alt9}
    _smp9['ai_policy_version'] = POLICY_VERSION
    _smp9['observation_contract'] = ai.OBSERVATION_CONTRACT
    from trace_decisions import ACTION_ENCODING_CONTRACT
    _smp9['action_encoding_contract'] = ACTION_ENCODING_CONTRACT
    from rendered_history import CHRONOLOGY_CONTRACT
    _smp9['chronology_contract'] = CHRONOLOGY_CONTRACT
    # A batch-start snapshot proves the first action boundary only. It does
    # not timestamp simulated followups, combat rounds, or later phases.
    # Those rows need their emitter's explicit prefix; unresolved rows stay
    # valid policy examples but cannot silently enter TD trajectories.
    _batch_phases = {'draw_card', 'army_move', 'army_attack', 'muster_minion',
                     'muster_elite', 'muster_recruit', 'advance_political',
                     'companion_separate', 'fellowship_move', 'fellowship_hide',
                     'play_event', 'nazgul_move', 'mouth_move'}
    _scope = getattr(st, '_rx_decision_scope', None)
    _time = getattr(st, '_rx_decision_time', None)
    _smp9['decision_time'] = _time if _scope == 'explicit' or phase in _batch_phases else None
    # Reusing this state after a transition needs a fresh boundary stamp.
    st._rx_decision_time = None
    _smp9['deck_prefix'] = getattr(st, '_rx_deck_prefix', None)
    if target_projection is not None:
        _smp9['policy_target_projection'] = target_projection
    if combat_refusal is not None:
        _smp9['combat_choice_refusal'] = combat_refusal
    elif combat_evidence is not None:
        import copy
        _smp9['combat_choice_evidence'] = copy.deepcopy(combat_evidence)
    if phase == 'fp_shine' and shine_refusal is None:
        _smp9['shine_choice_evidence'] = dict(shine_evidence)
    if hunt_evidence is not None:
        _smp9['hunt_pool_provenance'] = hunt_evidence
    if hunt_refusal is not None:
        _smp9['hunt_pool_refusal'] = hunt_refusal
    if policy_exclusion is not None:
        _smp9['ai_policy_exclusion'] = policy_exclusion
    if _alias9:
        # W4-032 sponsor condition: the row trains an action the v41
        # encoder aliases with a plain twin - stamped, never hidden
        _smp9['alias_v41'] = True
    return _smp9


# Ring menus and observations must agree on inventory and turn availability.
# Missing Ring arms are diagnostic counts: pending decisions may correctly
# exclude them. No menu arm or positive target is invented by this guard.
_RINGX_COL_CAT = 9           # action-encoding: special_character cat
# ai.encode_action puts the NEW face in col 23 and the OLD face in col
# 24 (HUNT-A F9).
_RINGX_COL_NEWFACE = 23      # elven ring NEW face
_RINGX_COL_OLDFACE = 24      # elven ring OLD face
_RINGX_COL_OWNRINGS = 3333   # count/3; -(count+1)/3 once used this turn


def _ring_cand_mask(encs):
    e = np.asarray(encs)
    return ((e[:, _RINGX_COL_CAT] > 0.5)
            & (e[:, _RINGX_COL_NEWFACE] > 1e-3)
            & (e[:, _RINGX_COL_OLDFACE] > 1e-3))


def _filter_ai_policy_samples(samples, stats):
    """Retain human history, excluding fixed/off-policy training targets."""
    kept = []
    for sample in samples:
        reason = sample.get('ai_policy_exclusion')
        if reason:
            stats['drop_reasons']['fixed_policy:' + reason] += 1
        else:
            kept.append(sample)
    stats['stored_phases'] = Counter(str(s.get('phase', '')) for s in kept)
    return kept


def _ring_truth_guard(samples, stats):
    """Reject stored menus offering unavailable ring actions.

    The current own-ring slot stores count/3 before use and -(count+1)/3
    after use this turn. Historical nonnegative counts remain readable.
    Menus with a ring arm while count is zero or the ring is already used
    are dropped. Available-ring observations without a ring arm are counted
    for inspection; some are legitimate pending decisions.
    """
    kept = []
    for s in samples:
        encs = s.get('encs')
        x9 = s.get('x')
        if encs is None or x9 is None or np.asarray(encs).ndim != 2 \
                or not len(encs):
            kept.append(s)
            continue
        has_ring_arm = bool(_ring_cand_mask(encs).any())
        encoded = float(x9[_RINGX_COL_OWNRINGS])
        used = encoded < 0
        count = -3 * encoded - 1 if used else 3 * encoded
        holds = count > 1e-6
        if has_ring_arm and (not holds or used):
            reason = 'ring_truth_already_used' if used else 'ring_truth_zero_hold'
            stats['drop_reasons'][reason] += 1
            continue
        if holds and not used and not has_ring_arm:
            stats['detected']['ring_opt_gap'] += 1
        kept.append(s)
    return kept


def _match_index(legal, pred):
    """Unique params-level match; on a cross-face multi-match (the v31
    marginal menus offer the same action under several die faces),
    disambiguate by the face the human actually spent (_MM_* context
    set by _mk_state_with_die).  Still returns None on genuine
    ambiguity â€” purity preserved."""
    idxs = [i for i, a in enumerate(legal) if pred(a)]
    if len(idxs) > 1 and _MM_FACE is not None and _MM_STATE is not None:
        dl = _MM_STATE.dice.get(_MM_SIDE, [])

        def _face_of(a):
            if a.die_idx is None or a.die_idx < 0:
                return None
            if a.die_idx < len(dl):
                return dl[a.die_idx].current
            return None

        fidxs = [i for i in idxs if _face_of(legal[i]) is _MM_FACE]
        if fidxs:
            idxs = fidxs
    return idxs[0] if len(idxs) == 1 else None


def _combat_target_sample(batch, stats):
    """BC label for the attacker's TARGET decision in a cleanly-driven combat.
    Rebuilds a FRESH pre-combat state (the drive mutated its own copy), plans
    the attacker/defender split, and matches the human's target against the
    legal army_attack candidate set."""
    import ai
    import rendered_census as rc
    import rendered_replay as rr
    st, side = _mk_state_with_die(batch, install_hands=True)
    if st is None:
        stats['drop_reasons']['attack_recon'] += 1
        return None
    try:
        _ch0 = [r for r in batch['rng'] if r.get('kind') == 'roll_num']
        if _card_strike(batch):
            # a CARD STRIKE (Dreadful Spells), not a battle - no attack
            # decision exists (W1-001: the strike test is the shared
            # predicate; an attacker-skipped-roll battle is NOT a strike)
            stats['drop_reasons']['attack_card_strike'] += 1
            return None
        _ed0 = len(_ch0[0]['ints']) if _ch0 else None
        plan = rc._plan_combat(st, batch, expected_dice=_ed0,
                               chats=_batch_chats(batch))
    except Exception:
        plan = None
    if not plan:
        stats['drop_reasons']['attack_no_plan'] += 1
        return None
    atk, dfn, _is_siege = plan
    try:
        legal = ai.get_legal_actions(st, side)
    except Exception:
        stats['drop_reasons']['attack_legal_exc'] += 1
        return None
    # W4-018: when the batch's die BOUGHT a card (a play_event row was
    # emitted for this batch - the resolver path marks it) the battle
    # is the card's interior (The Ringwraiths Are Abroad, 2023-03-16
    # L1036-1043: P die -> card -> Nazgul flights -> Iron Hills attack)
    # - a spare A/H die in the now-true pool must not turn it into a
    # second die spend, and the die the card consumed cannot be claimed
    # twice. A plain combat
    # card commit from the hand is NOT a purchase (no play_event row).
    _cardint9 = bool(batch.get('_rx_die_bought_card'))
    if _cardint9:
        stats['detected']['attack_card_interior'] += 1
    hidx = None if _cardint9 else _match_index(
        legal, lambda a: a.action_type == 'army_attack' and
        rr.sanitize(str(a.params.get('from', ''))) == atk and
        rr.sanitize(str(a.params.get('to', a.params.get('to_region', '')))) ==
        dfn)
    if hidx is None and not _cardint9:
        hidx = _match_index(
            legal, lambda a: a.action_type == 'army_attack' and
            rr.sanitize(str(a.params.get('to',
                            a.params.get('to_region', '')))) == dfn)
    if hidx is None:
        # A battle gesture proves an outcome, not an executable root action
        # or its payment. Card interiors need their actual continuation menu.
        stats['drop_reasons'][('attack_card_interior_unmatched' if _cardint9
                               else 'attack_recovery_not_in_public_menu')] += 1
        return None
    stats['kinds_emit']['army_attack'] += 1
    return _emit_sample(st, side, legal, hidx, 'army_attack', batch['turn'])


def _emit_chat_battle(batch, stats):
    """Zero-kill battle family (truth rows T4-8..T4-11, Pelargir): a battle
    whose rounds produced NO casualties never classifies 'combat'
    (classify_batch keys on kill records), so the whole family fell through
    to bookkeeping - where the FP retreat drag even mis-emitted as a SHADOW
    army_move.  Detection is deliberately tight (low-false-positive): a
    battle-capable staged die, roll chunks from BOTH players, no hunt tile,
    no unit kills, and a defender region PROVEN by an advance-in or
    retreat-out gesture.  Emissions, each gesture-anchored: the attack
    (advance source = attacker region), per-side round-1 combat-card
    commit/decline, the attacker continue ('press' chat / the retreat
    proves the battle went past the round), the defender retreat (net
    drag destination), and the victor advance (the advance gesture)."""
    import ai
    import rendered_census as rc
    import rendered_replay as rr
    from game_state import Side
    from ai import Action as _Act
    if batch.get('die_side') not in ('SHADOW', 'FREE_PEOPLES'):
        return None
    if batch.get('die_face') not in ('A', 'C', 'H', 'M', 'W'):
        return None
    if any(it.get('kind') == 'draw_tile' for it in batch['rng']):
        return None
    chunks = [it for it in batch['rng'] if it.get('kind') == 'roll_num']
    # W2-029 (P, guard i): the acting side's EVENT PLAY lands in the same
    # Current-card zone as a combat-card commit and every event card
    # carries a '/' combat face (bowzo 2025-05-20 L313 "Corsairs of
    # Umbar/Deadly Strife" Shadow Hand -> Current FP Card), so the
    # batch's own play_event is never commit proof.  W1-001's shared
    # predicate decides: the actor's hand exit is a COMMIT only when
    # _commit_in_missed_battle says so (opponent's commit / rolls);
    # otherwise the exit is the play and drops out of the proof.
    _own_plays9 = set()
    if not _commit_in_missed_battle(batch):
        _own_plays9 = {str(t9) for (_l9, t9) in _card_play_titles(batch)}
    _cc_recs9 = [r for r in batch['records']
                 if r.get('op') == 'move' and '/' in str(r.get('obj', ''))
                 and str(r.get('obj', '')) not in _own_plays9]
    _own_in_cc9 = any(
        r.get('op') == 'move' and str(r.get('obj', '')) in _own_plays9
        and ('current fp card' in str(r.get('to', '')).lower()
             or 'current sa card' in str(r.get('to', '')).lower())
        for r in batch['records'])
    _cc_pair = (
        any('current fp card' in str(r.get('to', '')).lower()
            for r in _cc_recs9)
        and any('current sa card' in str(r.get('to', '')).lower()
                for r in _cc_recs9))
    _one_cc = (
        any('current fp card' in str(r.get('to', '')).lower()
            for r in _cc_recs9)
        or any('current sa card' in str(r.get('to', '')).lower()
               for r in _cc_recs9))
    if len(chunks) < 2 or len({c.get('player') for c in chunks}) < 2:
        # ROLL-LESS battle (GAME02 T4 Dale): a Scouts-class retreat
        # pre-empts every roll - the pre-roll combat-card COMMIT PAIR
        # is the battle's proof.  A SINGLE commit also qualifies when
        # the other side declined (GAME19 T6 Westemnet: SH 'nc', FP
        # Scouts) - the advance-in/retreat-out gestures confirm below.
        if not (not chunks and (_cc_pair or _one_cc)):
            if not chunks and _own_in_cc9:
                # the only "commit" was the batch's own event play:
                # not a battle - the card / bookkeeping path owns it.
                # C2-006 (COORD #2 2026-09-03, USER order 22:30): this is
                # a ROUTING tick - the caller falls through to
                # _emit_bookkeeping, which emits the row or ticks its own
                # NAMED drop - so it is metered in `detected`, never in
                # drop_reasons (where it inflated the 24-game drop total
                # by 150 and hid real reasons under the per-game
                # most_common(15) cap).
                stats['detected']['zk_own_play_not_commit'] += 1
            return None
    if rr._battle_regions(batch):
        return None                      # casualties -> combat family owns it
    side = (Side.SHADOW if batch['die_side'] == 'SHADOW'
            else Side.FREE_PEOPLES)
    dside = Side.FREE_PEOPLES if side == Side.SHADOW else Side.SHADOW

    def _rec_side(rec):
        import game_state as gs
        sides = set()
        for tok in _group_tokens(str(rec.get('obj', ''))):
            p = rr.parse_piece(tok)
            if p and p[0] == 'unit':
                try:
                    sides.add(ai._side_of_nation(gs.Nation[p[1]]))
                except Exception:
                    pass
        return sides

    # defender region + the gestures: own units IN (advance), enemy
    # units OUT (retreat)
    adv_srcs, dfn_cands, ret_moves = {}, set(), []
    adv_sources = {}
    reg_moves = []
    box_moves9 = []     # W2-029 (guard ii): every leg, box-sourced too
    for r in batch['records']:
        if r.get('op') != 'move':
            continue
        fk = rr.zone_kind(str(r.get('from', '')))
        tk = rr.zone_kind(str(r.get('to', '')))
        if fk[0] in ('region', 'stronghold') \
                and tk[0] in ('region', 'stronghold') and fk[1] != tk[1]:
            box_moves9.append((fk[1], tk[1], _rec_side(r),
                               r.get('line', 0)))
        if fk[0] != 'region' or tk[0] not in ('region', 'stronghold') \
                or fk[1] == tk[1]:
            continue
        ss = _rec_side(r)
        reg_moves.append((fk[1], tk[1], ss, r.get('line', 0)))
    tb = batch['truth_before']
    _ds_str = ('FREE_PEOPLES' if batch['die_side'] == 'SHADOW'
               else 'SHADOW')                # _snap_sides_at speaks strings
    for (frm, to, ss, ln) in reg_moves:
        if side in ss and _ds_str in (
                rc._snap_sides_at(tb, to)
                if hasattr(rc, '_snap_sides_at') else set()):
            # An army can be dragged through empty staging regions after
            # advancing. Such later legs do not change its attack origin.
            if batch['die_side'] in rc._snap_sides_at(tb, frm):
                adv_sources.setdefault(to, set()).add(frm)
                if len(adv_sources[to]) == 1:
                    adv_srcs[to] = frm
                else:
                    adv_srcs.pop(to, None)
            dfn_cands.add(to)
        if dside in ss and _ds_str in (
                rc._snap_sides_at(tb, frm)
                if hasattr(rc, '_snap_sides_at') else set()):
            dfn_cands.add(frm)
            ret_moves.append((frm, to, ln))
    # W2-029 (P, guard ii): a retreat-out whose units come BACK to the
    # defender region through LATER drags is a corrected mis-drag, not
    # a retreat (bowzo L315 3 Gondor Regular Dol Amroth -> Minas Tirith
    # Stronghold box, L316 Minas Tirith Stronghold -> Dol Amroth
    # Stronghold = the withdrawal INTO the stronghold).  The correction
    # leg leaves a BOX, so reg_moves never saw it and the "card play
    # whose effect moved units" guard below could not fire.
    _box_sorted9 = sorted(box_moves9, key=lambda x: x[3])

    def _nets_home9(frm, to, ln):
        dest, seen = to, {frm}
        while dest not in seen:
            seen.add(dest)
            nxt = next((t for (f, t, ss, l) in _box_sorted9
                        if l > ln and dside in ss and f == dest), None)
            if nxt is None:
                return False
            dest = nxt
            if dest == frm:
                return True
        return False
    _kept9 = [m for m in ret_moves if not _nets_home9(*m)]
    if len(_kept9) != len(ret_moves):
        stats['drop_reasons']['zk_retreat_netted_home'] += 1
        ret_moves = _kept9
    if len(dfn_cands) != 1:
        # prefer the candidate with BOTH signals (own units IN and
        # enemy units OUT) - a retreat PATH region with a garrison
        # (GAME02 T4 Iron Hills) only ever shows the out-signal
        _both9 = [d for d in dfn_cands if d in adv_srcs]
        if len(_both9) == 1:
            dfn_cands = set(_both9)
        else:
            return None                 # no unambiguous battle -> not ours
    dfn = next(iter(dfn_cands))
    if not chunks and not _cc_pair and not ret_moves:
        # single-commit roll-less: without a retreat-out the "commit"
        # is just a card play whose effect moved units (GAME19 T5 TSL
        # into besieged Lorien) - no battle
        return None
    atk = adv_srcs.get(dfn)
    if atk is None:
        stats['drop_reasons']['zk_no_attacker'] += 1
        return []
    st, _sd = _mk_state_with_die(batch, install_hands=True)
    if st is None:
        stats['drop_reasons']['zk_recon'] += 1
        return []
    out = []
    # 1) the attack ------------------------------------------------------
    try:
        legal = ai.get_legal_actions(st, side)
    except Exception:
        stats['drop_reasons']['zk_legal_exc'] += 1
        return out
    hidx = _match_index(
        legal, lambda a: a.action_type == 'army_attack' and
        rr.sanitize(str(a.params.get('from', ''))) == atk and
        rr.sanitize(str(a.params.get('to',
                        a.params.get('to_region', '')))) == dfn)
    if hidx is None:
        stats['drop_reasons']['attack_recovery_not_in_public_menu'] += 1
    else:
        stats['kinds_emit']['army_attack'] += 1
        s0 = _emit_sample(st, side, legal, hidx, 'army_attack', batch['turn'])
        if s0 is not None:
            out.append(s0)
    if not chunks:
        if hidx is None:
            stats['drop_reasons']['zk_rollless_attack_unproven'] += 1
            return out
        out.extend(_emit_rollless_public_retreat(
            batch, st, legal[hidx], side, dside, atk, dfn,
            ret_moves, reg_moves, stats))
        return out
    # 2) round-1 combat-card commit/decline, attacker then defender ------
    try:
        cc_atk, cc_dfn = rc._combat_cards(batch)
    except Exception:
        cc_atk = cc_dfn = None
    for (stage, s_en, cid) in (('attacker_combat_card', side, cc_atk),
                               ('defender_combat_card', dside, cc_dfn)):
        st.pending_combat = {
            'phase': stage, 'attacker_region': atk,
            'defender_region': dfn, 'attacker_side': side,
            'defender_side': dside, 'deciding_side': s_en,
            'is_siege': False}
        try:
            la = [a for a in ai.get_legal_actions(st, s_en)
                  if a.action_type.startswith('combat_card')]
        except Exception:
            la = []
        if cid is not None:
            h = _match_index(la, lambda a:
                             a.action_type == 'combat_card_play' and
                             a.params.get('card_id') == cid)
        else:
            h = _match_index(la, lambda a:
                             a.action_type == 'combat_card_none')
        if h is not None and len(la) >= 2:
            stats['kinds_emit']['combat_card'] += 1
            sc = _emit_sample(st, s_en, la, h, 'combat_card',
                              batch['turn'])
            if sc is not None:
                out.append(sc)
        elif cid is not None:
            stats['drop_reasons']['zk_cc_unmatched'] += 1
    # 3) attacker continue / cease (only after a FOUGHT round - a
    # Scouts-class pre-combat retreat leaves no continue decision) ----
    wins_n = len(_combat_round_windows(batch))
    if wins_n >= 1 and (ret_moves or wins_n >= 2):
        st.pending_combat = {
            'phase': 'attacker_continue', 'attacker_region': atk,
            'defender_region': dfn, 'attacker_side': side,
            'defender_side': dside, 'deciding_side': side,
            'is_siege': False}
        try:
            la = ai.get_legal_actions(st, side)
        except Exception:
            la = []
        h = _match_index(la, lambda a:
                         a.action_type == 'combat_continue')
        if h is not None and len(la) >= 2:
            stats['kinds_emit']['attacker_continue'] += 1
            sc = _emit_sample(st, side, la, h, 'attacker_continue',
                              batch['turn'])
            if sc is not None:
                out.append(sc)
        else:
            stats['drop_reasons']['zk_continue_unmatched'] += 1
    # 4) defender retreat (net drag destination) --------------------------
    if ret_moves:
        ret_moves.sort(key=lambda x: x[2])
        dest = ret_moves[0][1]
        # net drag destination: follow the retreating units through EVERY
        # later enemy-composition move (the leg out of an empty via-region
        # is not in ret_moves - T4-10 Pelargir->Lamedon->LOSSARNACH)
        _chain = sorted([(f, t, ln) for (f, t, ss, ln) in reg_moves
                         if dside in ss], key=lambda x: x[2])
        _seen9 = {dfn}
        moved_on = True
        while moved_on and dest not in _seen9:
            _seen9.add(dest)
            moved_on = False
            for (frm, to, ln) in _chain:
                if frm == dest and to != dest and to not in _seen9:
                    dest = to
                    moved_on = True
                    break
        st.pending_combat = {
            'phase': 'defender_retreat', 'attacker_region': atk,
            'defender_region': dfn, 'attacker_side': side,
            'defender_side': dside, 'deciding_side': dside,
            'is_siege': False}
        try:
            la = ai.get_legal_actions(st, dside)
        except Exception:
            la = []
        h = _match_index(la, lambda a:
                         a.action_type == 'combat_retreat' and
                         rr.sanitize(str(a.params.get('to_region', '')))
                         == dest)
        if h is None:
            stats['drop_reasons']['zk_retreat_not_in_public_menu'] += 1
        else:
            stats['kinds_emit']['defender_retreat'] += 1
            sc = _emit_sample(st, dside, la, h, 'defender_retreat',
                              batch['turn'])
            if sc is not None:
                out.append(sc)
        # strip the retreated defenders for the advance menu state
        try:
            import combat as _cmb
            _k9, _a9 = _cmb._army_entry(st, dfn)
            if _a9 is not None:
                _a9.units = [u for u in _a9.units
                             if ai._side_of_nation(u.nation) != dside]
        except Exception:
            pass
    # 5) victor advance ----------------------------------------------------
    comp = _moved_group_composition(batch, atk, dfn)
    # net out a split-back (GAME02 T4: the WK group advances into Dale,
    # then 1 regular is moved BACK to garrison Woodland Realm)
    _rev9 = _moved_group_composition(batch, dfn, atk)
    for _k9 in ('reg', 'elite', 'leaders', 'nazgul'):
        comp[_k9] = max(0, comp[_k9] - _rev9[_k9])
    if comp['reg'] + comp['elite'] + comp['leaders'] + comp['nazgul'] > 0:
        st.pending_combat = {
            'phase': 'victor_advance', 'attacker_region': atk,
            'defender_region': dfn, 'attacker_side': side,
            'defender_side': dside, 'deciding_side': side,
            'is_siege': False}
        try:
            la = ai.get_legal_actions(st, side)
        except Exception:
            la = []
        h = _match_army_move(la, st, side, atk, dfn, comp,
                             atype='victor_advance')
        if h is not None and len(la) >= 2:
            stats['kinds_emit']['victor_advance'] += 1
            sc = _emit_sample(st, side, la, h, 'victor_advance',
                              batch['turn'])
            if sc is not None:
                out.append(sc)
        else:
            stats['drop_reasons']['zk_advance_unmatched'] += 1
    st.pending_combat = None
    return out


def _emit_rollless_public_retreat(batch, state, attack, side, dside,
                                 atk, dfn, ret_moves, reg_moves, stats):
    """Reach a card-caused retreat via the actual attack and commitments.

    The log contains no rolls. Any engine request for randomness invalidates
    this continuation instead of manufacturing a retreat phase.
    """
    import ai
    import random
    import rendered_replay as rr
    out = []
    if not ret_moves:
        stats['drop_reasons']['zk_rollless_retreat_unproven'] += 1
        return out
    first_retreat = min(int(m[2]) for m in ret_moves)
    cards, _, ambiguous = _round_cards(batch, [
        {'first': first_retreat, 'last': first_retreat, 'next_first': 10 ** 9}])
    if ambiguous:
        stats['drop_reasons']['zk_rollless_commit_ambiguous'] += 1
        return out
    slots = cards[0]

    def apply_without_rng(st, action):
        def unexpected(*args, **kwargs):
            raise RuntimeError('rollless_unrecorded_rng')
        names = ('randint', 'choice', 'choices', 'random', 'shuffle')
        saved = {name: getattr(random, name) for name in names}
        for name in names:
            setattr(random, name, unexpected)
        try:
            return ai.apply_action(st, action)
        finally:
            for name, function in saved.items():
                setattr(random, name, function)

    try:
        st = apply_without_rng(state, attack)
        for owner in (side, dside):
            key = 'fp' if owner.name == 'FREE_PEOPLES' else 'sh'
            cid = slots.get(key)
            if cid is not None and not _combat_commit_board_matches(st, batch, slots.get(key + '_ln')):
                stats['drop_reasons']['zk_rollless_commit_board_unproven'] += 1
                return out
            menu = ai.get_legal_actions(st, owner)
            if cid is None and len(menu) > 1:
                # The existing chat index drops speaker identity. Silence or
                # another player's "nc" cannot prove this owner's decline.
                stats['drop_reasons']['zk_rollless_decline_unproven'] += 1
                return out
            index = _match_index(menu, lambda a:
                (a.action_type == 'combat_card_play' and a.params.get('card_id') == cid)
                if cid is not None else a.action_type == 'combat_card_none')
            if index is None:
                stats['drop_reasons']['zk_rollless_commit_not_in_menu'] += 1
                return out
            if len(menu) >= 2:
                out.append(_emit_sample(st, owner, menu, index, 'combat_card', batch['turn']))
                stats['kinds_emit']['combat_card'] += 1
            st = apply_without_rng(st, menu[index])
        pc = st.pending_combat or {}
        if pc.get('phase') != 'defender_retreat' or not pc.get('_card_retreat_offered'):
            stats['drop_reasons']['zk_rollless_card_retreat_unproven'] += 1
            return out
        ordered = sorted(ret_moves, key=lambda m: m[2])
        dest, last_line = ordered[0][1], ordered[0][2]
        seen = {dfn}
        for frm, to, owners, line in sorted(reg_moves, key=lambda m: m[3]):
            if line > last_line and dside in owners and frm == dest and to not in seen:
                seen.add(dest)
                dest, last_line = to, line
        menu = ai.get_legal_actions(st, dside)
        index = _match_index(menu, lambda a: a.action_type == 'combat_retreat'
                             and rr.sanitize(str(a.params.get('to_region', ''))) == dest)
        if index is None:
            stats['drop_reasons']['zk_rollless_retreat_not_in_menu'] += 1
            return out
        if len(menu) >= 2:
            out.append(_emit_sample(st, dside, menu, index, 'defender_retreat', batch['turn']))
            stats['kinds_emit']['defender_retreat'] += 1
        st = apply_without_rng(st, menu[index])
        comp = _moved_group_composition(batch, atk, dfn)
        reverse = _moved_group_composition(batch, dfn, atk)
        for key in ('reg', 'elite', 'leaders', 'nazgul'):
            comp[key] = max(0, comp[key] - reverse[key])
        menu = ai.get_legal_actions(st, side)
        index = _match_army_move(menu, st, side, atk, dfn, comp, atype='victor_advance')
        if index is not None and len(menu) >= 2:
            out.append(_emit_sample(st, side, menu, index, 'victor_advance', batch['turn']))
            stats['kinds_emit']['victor_advance'] += 1
        elif comp['reg'] + comp['elite']:
            stats['drop_reasons']['zk_rollless_advance_unproven'] += 1
    except Exception as exc:
        stats['drop_reasons']['zk_rollless_apply:' + str(exc)[:32]] += 1
    return out


def _fellowship_move_sample(batch, stats):
    """BC label for the FREE-PEOPLES fellowship-MOVE decision (advance the
    hidden pawn / step in Mordor).  This is a pure die-policy choice â€” the
    Fellowship pawn advance â€” that is INDEPENDENT of the RNG hunt outcome (the
    tile damage/reveal is forced randomness, not a decision), so it is labelled
    whether or not the hunt itself was engine-drivable.  Emitted only when the
    batch physically advances the pawn (an FSP feltrack move, or a Mordor step)
    and fellowship_move is uniquely legal on the reconstructed truth-before."""
    import ai
    from game_state import Side
    if batch.get('die_side') != 'FREE_PEOPLES':
        return None
    # a Fellowship pawn advance is always a Character (or Will) die; other
    # faces that touch the FSP marker are the reveal marker-reset (which the
    # VASSAL Fellowship track zeroes on every reveal â€” NOT a move).
    _ring_c = False
    if batch['die_face'] not in ('C', 'W'):
        # ring-made C without a 'changes' line (GAME02 dialect): the
        # counter move + the pawn advance itself pin the conversion
        if _ring_counter_used(batch) and _ring_implied_face(batch) == 'C':
            _ring_c = True
        else:
            # gate-(f): a no-sample return must be COUNTED (purity
            # drop) - FP hunt batches under M/P/H faces with no ring
            # evidence (an unpinned ring/card-driven advance dialect).
            stats['drop_reasons']['felmove_face_skip'] += 1
            return None
    tb = batch['truth_before']['fel']
    ta = batch['truth']['fel']
    moved = _is_paid_fellowship_move(batch)
    if not moved:
        # gate-(f): counted - C-face hunt batches with no visible pawn
        # advance
        stats['drop_reasons']['felmove_nomove_skip'] += 1
        return None
    st, side = _mk_state_with_die(batch, install_hands=True)
    if st is None:
        stats['drop_reasons']['felmove_recon'] += 1
        return None
    if _ring_c:
        import game_state as _gsF
        # convert only the LEAD (staged) die â€” rewriting the whole pool
        # misstated the v31 marginal menu for the batch
        if st.dice.get(side):
            st.dice[side][0].current = _gsF.ActionDieFace.CHARACTER
    _prepare_fellowship_dice(st, batch)
    try:
        legal = ai.get_legal_actions(st, side)
    except Exception as e:
        stats['drop_reasons']['felmove_legal_exc:%s' % str(e)[:14]] += 1
        return None
    hidx = _match_index(legal,
                        lambda a: a.action_type == 'fellowship_move')
    if hidx is not None:
        stats['kinds_emit']['fellowship_move'] += 1
        return _emit_sample(st, side, legal, hidx, 'fellowship_move',
                            batch['turn'])
    stats['drop_reasons']['fellowship_move_unmatched'] += 1
    return None


# ==========================================================================
# rendered-gesture extractors (what the human spent the die on) + exact,
# unambiguous legality matchers.  EVERY sample is matched to a UNIQUE legal
# action (the composition/among-variants IS the decision) â€” an ambiguous or
# absent match DROPS the sample (purity, never a guess).  All menus are
# board-only (the per-batch reconstruction carries EMPTY hands, so no
# fabricated event-card alternatives ever enter the candidate set).
# ==========================================================================
def _group_tokens(obj):
    import rendered_replay as rr
    mg = re.match(r'^group\s*\((.*)\)$', obj, re.I)
    return rr.split_group(mg.group(1)) if mg else [obj]


def _moved_group_composition(batch, frm, to, lo=0, hi=None):
    """Sum the figures the human moved frm->to across EVERY move record of the
    batch (one army move renders as several unit/leader/split lines that all
    share one (from,to)).  Returns dict(reg, elite, leaders, nazgul).
    Optional [lo, hi) line window restricts to a combat-round segment
    (victor-advance evidence must not count the pre-battle approach)."""
    import rendered_replay as rr
    comp = {'reg': 0, 'elite': 0, 'leaders': 0, 'nazgul': 0}
    for r in batch['records']:
        if r.get('op') != 'move':
            continue
        ln = r.get('line', 0)
        if ln < lo or (hi is not None and ln >= hi):
            continue
        fk = rr.zone_kind(r.get('from', ''))
        tk = rr.zone_kind(r.get('to', ''))
        if fk[0] not in ('region', 'stronghold') or \
                tk[0] not in ('region', 'stronghold'):
            continue
        if fk[1] != frm or tk[1] != to:
            continue
        for t in _group_tokens(r.get('obj', '')):
            if t.strip().lower() == 'the fellowship':
                continue
            p = rr.parse_piece(t)
            if not p:
                continue
            if p[0] == 'unit':
                if p[2] == 'REGULAR':
                    comp['reg'] += p[3]
                elif p[2] == 'ELITE':
                    comp['elite'] += p[3]
            elif p[0] == 'nazgul':
                comp['nazgul'] += p[3]
            elif p[0] in ('leader', 'named'):
                # W2-033 (COORD #2 2026-09-03): Saruman never leaves
                # Orthanc (game_logic._is_saruman excludes him from every
                # transport) - his token in a drag is a client gesture,
                # not a moved leader (2023-12-25 L768 dragged him with the
                # Fords of Isen advance; the opponent put him back L770).
                if p[0] == 'named' and 'saruman' in str(p[1]).lower():
                    continue
                comp['leaders'] += p[3]
    return comp


def _piece_key(tok):
    """Figure identity of one rendered piece token for the netting pass:
    ('unit', NATION, KIND) / ('nazgul', 'SAURON', None) / ('leader',
    NATION, None) / ('named', NAME, None) plus its count, or None for a
    non-figure token (the fellowship, markers, dice, cards, tiles)."""
    import rendered_replay as rr
    if tok.strip().lower() == 'the fellowship':
        return None, 0
    p = rr.parse_piece(tok)
    if not p or p[0] not in ('unit', 'nazgul', 'leader', 'named'):
        return None, 0
    return (p[0], p[1], p[2]), p[3]


def _comp_of_keys(keys):
    """dict(reg, elite, leaders, nazgul) of a piece-key Counter (same
    shape as _moved_group_composition; named figures count as leaders)."""
    comp = {'reg': 0, 'elite': 0, 'leaders': 0, 'nazgul': 0}
    for (kind, _nat, sub), n in keys.items():
        if kind == 'unit':
            if sub == 'REGULAR':
                comp['reg'] += n
            elif sub == 'ELITE':
                comp['elite'] += n
        elif kind == 'nazgul':
            comp['nazgul'] += n
        else:
            if kind == 'named' and 'saruman' in str(_nat).lower():
                continue                # W2-033: Saruman never moves
            comp['leaders'] += n
    return comp


_ADJ_CACHE = {}


def _adjacent_san(a, b):
    """W4-003: board adjacency of two SANITIZED region names (cached)."""
    key = (a, b)
    if key not in _ADJ_CACHE:
        import board
        import rendered_replay as rr
        try:
            nb = board.load_board().adjacent(a)
            _ADJ_CACHE[key] = any(rr.sanitize(str(x)) == b for x in nb)
        except Exception:
            _ADJ_CACHE[key] = False
    return _ADJ_CACHE[key]


def _own_figures_before(batch, region):
    """W4-003: Counter(piece key -> n) of the figures standing in
    `region` at the batch's truth-before (the mirror snapshot), same
    keys as _piece_key."""
    from collections import Counter
    c = Counter()
    try:
        ents = (batch.get('truth_before') or {}).get('armies') or {}
        ent = ents.get(region)
        if ent is None:
            return c
        units, leaders = ent
        for (nat, kind) in units:
            c[('unit', str(nat), str(kind))] += 1
        for ld in leaders:
            ld = tuple(ld)
            if ld[0] == 'NAZGUL':
                c[('nazgul', 'SAURON', None)] += 1
            elif ld[0] == 'LEADER':
                c[('leader', str(ld[1]), None)] += 1
            elif ld[0] == 'NAMED':
                c[('named', str(ld[1]), None)] += 1
    except Exception:
        return Counter()
    return c


def _companion_flows(batch, lo=0, hi=None):
    """W4-003: the batch's COMPANION flights as ORIGIN -> FINAL region
    per companion, grouped by (origin, final) in first-seen order:
    [(origin, final, (names...)), ...].

    A multi-region companion move is rendered as one drag per region
    (Legolas Moria->Dimrill Dale, Dimrill Dale->Lorien = ONE move
    Moria->Lorien); a drag back is a retraction (Aragorn Dol Amroth->
    Anfalas->Dol Amroth->Osgiliath = Dol Amroth->Osgiliath); a group
    that splits on the way (Strider+Boromir Moria->Dimrill Dale, then
    Legolas on to Lorien) is two groups.  Only FP companions (COMP_ALIAS
    names) count - minions belong to the Shadow branch.  A companion
    whose final region is its origin made no move."""
    import rendered_replay as rr
    pos, origin, order = {}, {}, []
    for r in batch['records']:
        if r.get('op') != 'move':
            continue
        ln = r.get('line', 0)
        if ln < lo or (hi is not None and ln >= hi):
            continue
        fk = rr.zone_kind(r.get('from', ''))
        tk = rr.zone_kind(r.get('to', ''))
        if fk[0] not in ('region', 'stronghold') or \
                tk[0] not in ('region', 'stronghold') or fk[1] == tk[1]:
            continue
        for t in _group_tokens(r.get('obj', '')):
            p = rr.parse_piece(t)
            if not p or p[0] != 'named':
                continue
            nm = rr.COMP_ALIAS.get(str(p[1]).lower())
            if nm is None or nm == 'Gollum':
                continue
            if nm not in origin:
                origin[nm] = fk[1]
                order.append(nm)
            pos[nm] = tk[1]
    groups, gorder = {}, []
    for nm in order:
        if origin[nm] == pos[nm]:
            continue
        key = (origin[nm], pos[nm])
        if key not in groups:
            groups[key] = []
            gorder.append(key)
        groups[key].append(nm)
    return [(f, t, tuple(groups[(f, t)])) for (f, t) in gorder]


def _net_batch_flows(batch, lo=0, hi=None, keep_empty=False, through=False):
    """W4-001 / W4-002 (shared with the combat side, W2-003 / W2-004):
    the batch's region->region figure moves NETTED per drag chain, in
    first-seen order.

    VASSAL has no 'move fewer' gesture.  A human who dragged N figures
    A->B and wants N-k there drags k back (partial return, W4-001: the
    gross count trained and the return broke the batch's second army
    move); a drag back of the whole group is a retracted plan (full
    reversal, W4-002: a phantom army_move row for a move the human did
    not make); two drags of the same (A,B) sum (re-drag).  The human's
    decision is the NET flow, so every move record in [lo, hi) is
    replayed as a token flow: a figure keeps the region it started the
    batch in as its ORIGIN, and a record X->Y first takes the figures at
    X that came from Y (they go home and cancel), the rest are X's own
    figures (a new flow X->Y).  Flows whose figures all went home emit
    nothing (unless keep_empty, for telemetry).

    Returns a list of dicts, same-(from,to) flows merged in first-seen
    order: {'from', 'to', 'keys': Counter(piece key -> n), 'comp':
    dict(reg, elite, leaders, nazgul), 'has_unit', 'line' (first record),
    'returned': Counter(piece key -> n dragged back)}.  Composition and
    order equal _army_move_records + _moved_group_composition whenever
    the batch has no return drag.

    through=True (W4-003, the caller proves ONE die was spent): figures
    that ARRIVED at B from A earlier in the batch and are dragged on to
    C are one move A->C when A is adjacent to C - VASSAL renders a
    drag-through of B as two drags, but one die never moves the same
    army twice (2023-01-24 L329-330 Parth Celebrant->South Anduin Vale
    ->Western Brown Lands; 2025-05-20 L545-546 Dagorlad->Morannon->
    North Ithilien; 2023-09-07 L635-636 Lorien->Dimrill Dale->Parth
    Celebrant with Dimrill Dale's own nazgul dragged along; 2025-01-18
    L650-652 Lorien->Old Ford->Parth Celebrant then two Sauron regulars
    back home; 2023-03-24 L777-778 Orthanc->Helm's Deep->Fords of
    Isen).  The through reading is taken when the drag out of B is
    EXACTLY the stack that arrived (the human picked up what he just
    dropped) or when B's own figures at the batch start cannot supply
    the drag (a mix of B's own and arrived figures would move the
    arrived army twice); otherwise B's own figures move (literal
    reading, counted on batch['_rx_through_ambiguous']).  The output
    flow carries 'via': [B, ...] and A's remainder stays a flow A->B."""
    from collections import Counter
    import rendered_replay as rr
    open_ = []          # drag chains still on the board, in record order
    batch['_rx_through_ambiguous'] = 0
    # W4-026: figures this batch PLACED from Reinforcements (directly or
    # parked on a card zone first) are not board figures at the batch
    # start - a later drag of such a figure out of its landing region
    # is the muster-then-drag placement gesture (_muster_placements),
    # never an army move (2023-03-06 L128-130: 3 Isengard regulars
    # placed at North Dunland, one dragged on to South Dunland and one
    # to Orthanc, read as army_move north dunland->south dunland of the
    # regular that stood there).  Placed tokens travel with the drag
    # (capped at the placed count: an old figure dragged along is still
    # a flow) and a put-back un-places.
    placed = Counter()  # (region | '__card__', piece key) -> n
    for r in batch['records']:
        if r.get('op') != 'move':
            continue
        ln = r.get('line', 0)
        if ln < lo or (hi is not None and ln >= hi):
            continue
        obj = str(r.get('obj', ''))
        ol = obj.strip().lower()
        if ol in ('the fellowship', 'fsp', 'corruption', 'nothing') or \
                ol.endswith('control marker') or \
                ol.startswith('victory points marker'):
            continue
        fk = rr.zone_kind(r.get('from', ''))
        tk = rr.zone_kind(r.get('to', ''))
        f_reg = fk[0] in ('region', 'stronghold')
        t_reg = tk[0] in ('region', 'stronghold')
        if not (f_reg and t_reg):
            src = ('__card__' if fk[0] == 'card' else
                   fk[1] if f_reg else None)
            dst = ('__card__' if tk[0] == 'card' else
                   tk[1] if t_reg else None)
            if fk[0] == 'reinf' and dst is not None:
                for t in _group_tokens(obj):
                    k, n = _piece_key(t)
                    if k is not None and n > 0:
                        placed[(dst, k)] += n
            elif src is not None and (dst is not None or
                                      tk[0] == 'reinf'):
                for t in _group_tokens(obj):
                    k, n = _piece_key(t)
                    if k is None or n <= 0:
                        continue
                    take = min(n, placed[(src, k)])
                    if take > 0:
                        placed[(src, k)] -= take
                        if dst is not None:
                            placed[(dst, k)] += take
            continue
        frm, to = fk[1], tk[1]
        if frm == to:
            continue
        rem = Counter()
        for t in _group_tokens(obj):
            k, n = _piece_key(t)
            if k is not None and n > 0:
                rem[k] += n
        if not rem:
            continue
        # 0) figures placed here by this batch travel on as placements
        for k in list(rem):
            take = min(rem[k], placed[(frm, k)])
            if take > 0:
                placed[(frm, k)] -= take
                placed[(to, k)] += take
                rem[k] -= take
        rem = +rem
        if not rem:
            continue
        # 1) figures that came from `to` earlier in the batch go home
        for o in reversed(open_):
            if not rem:
                break
            if o['to'] != frm or o['from'] != to:
                continue
            for k in list(rem):
                take = min(rem[k], o['keys'].get(k, 0))
                if take > 0:
                    o['keys'][k] -= take
                    o['returned'][k] += take
                    rem[k] -= take
            rem = +rem
        # 1b) W4-003 drag-through: figures that arrived at `frm` from A
        #     and are dragged on to `to` (A adjacent to `to`, one die)
        #     continue their ONE move A->`to`
        if through and rem:
            own = None
            for o in reversed(open_):
                if not rem:
                    break
                if o['to'] != frm or o['from'] == to or not +o['keys']:
                    continue
                if not _adjacent_san(o['from'], to):
                    continue
                shared = Counter()
                for k in list(rem):
                    n = min(rem[k], o['keys'].get(k, 0))
                    if n > 0:
                        shared[k] = n
                if not shared:
                    continue
                if own is None:
                    own = _own_figures_before(batch, frm)
                exact = (+Counter(rem)) == (+Counter(o['keys']))
                forced = any(own.get(k, 0) < rem[k] for k in shared)
                if not (exact or forced):
                    batch['_rx_through_ambiguous'] += 1
                    continue
                for k, n in shared.items():
                    o['keys'][k] -= n
                    rem[k] -= n
                open_.append({'from': o['from'], 'to': to,
                              'keys': shared, 'returned': Counter(),
                              'line': o['line'],
                              'via': list(o.get('via', [])) + [frm]})
            rem = +rem
        # 2) the rest is a drag of frm's own figures
        if rem:
            open_.append({'from': frm, 'to': to, 'keys': rem,
                          'returned': Counter(), 'line': ln})
    merged = {}
    order = []
    for o in open_:
        key = (o['from'], o['to'])
        if key not in merged:
            merged[key] = {'from': o['from'], 'to': o['to'],
                           'keys': Counter(), 'returned': Counter(),
                           'line': o['line'], 'via': []}
            order.append(key)
        merged[key]['keys'] += o['keys']
        merged[key]['returned'] += o['returned']
        for v in o.get('via', []):
            if v not in merged[key]['via']:
                merged[key]['via'].append(v)
    out = []
    for key in order:
        fl = merged[key]
        fl['keys'] = +fl['keys']
        fl['comp'] = _comp_of_keys(fl['keys'])
        fl['has_unit'] = (fl['comp']['reg'] + fl['comp']['elite']) > 0
        if fl['keys'] or keep_empty:
            out.append(fl)
    return out


def _match_army_move(legal, st, side, frm, to, comp, atype='army_move'):
    """Return the UNIQUE legal `atype` (army_move / victor_advance) index
    whose EFFECTIVE moved figures == the human's moved composition, or None.

    The engine's split grid keys on the boolean ``take_leaders`` (all plain
    leaders ride) + ``n_nazgul``, NOT on a moved-leader COUNT, and it only
    offers ``take_leaders=True`` for a full-stack move (a proper subset that
    leaves units behind may set it False).  So a bare 3-Regular move out of a
    LEADERLESS stack matches the action (3R, take_leaders=True) â€” take_leaders
    is moot with no leaders present.  We therefore resolve each candidate to
    the figures it would ACTUALLY move â€” eff_leaders = (all source plain
    leaders if take_leaders else 0); eff_nazgul = n_nazgul, or (all source
    nazgul if take_leaders else 0) for the base variant â€” and match those
    against the human's moved counts.  Ambiguity (two candidates that move the
    identical figure set) tie-breaks to take_leaders == (a plain leader
    actually moved); anything still ambiguous DROPS (purity)."""
    import rendered_replay as rr
    from game_state import Side, Nazgul, Leader
    src_naz = src_ldr = 0
    for k, a in st.armies.items():
        if rr.sanitize(k) == frm:
            if atype == 'victor_advance':
                # RB28/31: only the declared attacking force can advance.
                # A peaceful rear Leader remains physically in this Army
                # entry, even when the destination is its own homeland.
                from army_participants import battle_army_view
                a = battle_army_view(st, a, k, side)
            src_naz = sum(1 for l in a.leaders if isinstance(l, Nazgul))
            # W4-037: ONE Army entry per region holds BOTH sides' leader
            # figures (game_logic.get_or_create_army) - a lone Companion
            # standing where a Shadow army leaves, a Nazgul / Minion on
            # an FP field army (1.251).  Only the MOVER's own figures
            # ride (engine grid _n_pl9: game_logic._leader_side(l) ==
            # side), so the enemy's figures never count toward the
            # effective plain leaders of a take_leaders=True candidate.
            import game_logic as _gl
            _own_l = [l for l in a.leaders if _gl._leader_side(l) == side]
            # Saruman (immobile) NEVER rides - the executor excludes
            # him from every transport, so take_leaders is moot for
            # him and he must not count toward the effective leaders
            # (T5 FoI: 4-elite advance out of Orthanc left him behind)
            src_ldr = sum(1 for l in _own_l
                          if not isinstance(l, Nazgul)
                          and not getattr(l, 'immobile', False)
                          and 'saruman' not in
                          str(getattr(l, 'name', '')).lower())
            # W4-038b: on a VOLUNTARY crossing into another Nation's
            # borders the engine's take_leaders=True carries only the
            # ELIGIBLE plain Leaders - Companions / Minions always,
            # plain Leaders of at-war or destination nations; the
            # non-belligerent Leader stays behind (rulebook L3093-3096 /
            # L3108-3109 / L4189-4191; 2024-05-21 SlickRick L842-844:
            # the North Leader is put back at Dale).  Resolve eff_ldr
            # with the engine's OWN predicate body (ai._border_pred9,
            # shared by the menu and the executor) - never a second
            # copy of the rule; None = no restriction on this move.
            import ai as _ai
            _bp = getattr(_ai, '_border_pred9', None)
            if _bp is not None:
                _own = [u for u in a.units
                        if _ai._side_of_nation(u.nation) == side]
                _nok = _bp(st, _own, a.leaders, side, to)
                if _nok is not None:
                    src_ldr = sum(1 for l in _own_l
                                  if not isinstance(l, Nazgul)
                                  and not getattr(l, 'immobile', False)
                                  and 'saruman' not in
                                  str(getattr(l, 'name', '')).lower()
                                  and (not _gl.leader_must_follow(l)
                                       or _nok(l.nation)))
            break
    want_ldr = comp['leaders']
    want_naz = comp['nazgul']
    matches = []
    for i, a in enumerate(legal):
        if a.action_type != atype:
            continue
        p = a.params
        if rr.sanitize(str(p.get('from', ''))) != frm or \
                rr.sanitize(str(p.get('to', ''))) != to:
            continue
        if int(p.get('n_regular', 0)) != comp['reg'] or \
                int(p.get('n_elite', 0)) != comp['elite']:
            continue
        take = bool(p.get('take_leaders', False))
        if p.get('n_leaders') is not None:
            # LA-8 per-count leader split variants (DET census P3): the
            # explicit count IS the effective moved-leader number - the
            # src_ldr fallback computed eff_ldr=src for every variant,
            # multi-matched every full-leader partial-unit move out of a
            # multi-leader stack and ambiguity-dropped it.
            eff_ldr = int(p.get('n_leaders') or 0)
        else:
            eff_ldr = src_ldr if take else 0
        if 'n_nazgul' in p:
            eff_naz = int(p['n_nazgul'])
        else:
            eff_naz = src_naz if take else 0
        if side != Side.SHADOW:
            eff_naz = 0
        if eff_ldr != want_ldr or eff_naz != want_naz:
            continue
        matches.append((i, take))
    # v31 marginal menus: the identical move rides several die faces
    # (Army / Character / Muster-Army / Will) â€” disambiguate by the
    # face the human actually spent before the purity checks.
    if len(matches) > 1 and _MM_FACE is not None:
        dl = st.dice.get(side, [])

        def _face_i(i):
            di = legal[i].die_idx
            if di is None or di < 0 or di >= len(dl):
                return None
            return dl[di].current

        fm = [(i, t) for (i, t) in matches if _face_i(i) is _MM_FACE]
        if fm:
            matches = fm
    if len(matches) == 1:
        return matches[0][0]
    if len(matches) > 1:
        pref = [i for (i, take) in matches if take == (want_ldr > 0)]
        if len(pref) == 1:
            return pref[0]
    return None


def _reinf_recruits(batch):
    """(nation, region_san, kind, count) for every Reinforcements->region
    figure placement, IN ORDER (kind in REGULAR/ELITE/LEADER/NAZGUL/MINION).

    A minion muster reaches the board in TWO VASSAL legs â€” Reinforcements ->
    'Current FP/SA Card' (staging) -> region â€” so the reinf->card leg lands in
    no region and the card->region leg has a non-reinf source.  We recover the
    card->region leg for NAMED MINIONS ONLY (Saruman / Mouth / Witch-King),
    gated on the minion name so a companion/leader returning from a card slot
    is never mis-read as a muster.  (An event-card PLAY batch returns at the
    play_event branch before muster, so this only ever fires on a die-muster.)
    """
    import rendered_replay as rr
    out = []
    for r in batch['records']:
        if r.get('op') != 'move':
            continue
        fk = rr.zone_kind(r.get('from', ''))
        tk = rr.zone_kind(r.get('to', ''))
        if tk[0] not in ('region', 'stronghold'):
            continue
        if fk[0] == 'reinf':
            for t in _group_tokens(r.get('obj', '')):
                p = rr.parse_piece(t)
                if not p:
                    continue
                if p[0] == 'unit':
                    out.append((p[1], tk[1], p[2], p[3]))
                elif p[0] == 'leader':
                    out.append((p[1], tk[1], 'LEADER', p[3]))
                elif p[0] == 'nazgul':
                    out.append(('SAURON', tk[1], 'NAZGUL', p[3]))
                elif p[0] == 'named':
                    out.append((p[1], tk[1], 'MINION', 1))
        elif fk[0] == 'card':
            for t in _group_tokens(r.get('obj', '')):
                p = rr.parse_piece(t)
                if p and p[0] == 'named' and _is_minion_name(p[1]):
                    out.append((p[1], tk[1], 'MINION', 1))
    return out


def _is_minion_name(name):
    n = str(name).lower()
    return ('saruman' in n or 'mouth' in n or 'witch' in n)


def _muster_placements(batch):
    """FINAL (nation, region_san, kind, count) placements of a die-muster
    batch, one entry per (nation, region, kind), in order of first arrival
    at the final region (kind in REGULAR/ELITE/LEADER/NAZGUL/MINION).

    W4-007/W4-008: the reinf->region placements of _reinf_recruits are
    followed through the batch's LATER gestures of the same (nation, kind):
    a figure dragged on from its landing region ends where it stopped
    (staging drag - 2023-03-24 L506 Nazgul+Regular parked at E. Emyn Muil
    then split to Minas Morgul / Dol Guldur; 2026-01-08 L940 2 Nazgul to
    Morannon, one on to Minas Morgul; GAME19 T6 2 regulars into Minas
    Tirith, one on to Pelargir) and a figure put back into Reinforcements
    is un-placed (2024-08-11 L308-310: 2 Nazgul to Minas Morgul, one back
    to the pool, one to Dol Guldur).  Only figures placed by THIS batch
    move (the moved count is capped at what the batch put there), so an
    old unit dragged in the same batch is never re-read as a muster.
    Minions pass through as _reinf_recruits reports them."""
    import rendered_replay as rr
    placed = {}          # (nat, kind) -> {region: count}
    order = []           # (nat, region, kind) first-arrival order
    out_minions = []

    def _add(key, reg, n):
        d = placed.setdefault(key, {})
        if d.get(reg, 0) <= 0 and (key[0], reg, key[1]) not in order:
            order.append((key[0], reg, key[1]))
        d[reg] = d.get(reg, 0) + n

    for r in batch['records']:
        if r.get('op') != 'move':
            continue
        fk = rr.zone_kind(r.get('from', ''))
        tk = rr.zone_kind(r.get('to', ''))
        f_reg = fk[0] in ('region', 'stronghold')
        t_reg = tk[0] in ('region', 'stronghold')
        for t in _group_tokens(r.get('obj', '')):
            p = rr.parse_piece(t)
            if not p:
                continue
            if p[0] == 'unit':
                key, n = (p[1], p[2]), p[3]
            elif p[0] == 'leader':
                key, n = (p[1], 'LEADER'), p[3]
            elif p[0] == 'nazgul':
                key, n = ('SAURON', 'NAZGUL'), p[3]
            elif p[0] == 'named':
                if t_reg and (fk[0] == 'reinf' or
                              (fk[0] == 'card' and _is_minion_name(p[1]))):
                    out_minions.append((len(order), (p[1], tk[1],
                                                     'MINION', 1)))
                    order.append((p[1], tk[1], 'MINION'))
                continue
            else:
                continue
            if fk[0] == 'reinf' and t_reg:
                _add(key, tk[1], n)
            elif fk[0] == 'reinf' and tk[0] == 'card':
                # W4-026: parked on a card zone first (2023-03-03 L468
                # 3 Isengard regulars via Current SA Card, 2023-08-15
                # L273 via Current FP Card) - they arrive at their
                # region on the card->region leg below
                d = placed.setdefault(key, {})
                d['__card__'] = d.get('__card__', 0) + n
            elif fk[0] == 'card' and t_reg and \
                    placed.get(key, {}).get('__card__', 0) > 0:
                m = min(n, placed[key]['__card__'])
                placed[key]['__card__'] -= m
                _add(key, tk[1], m)
            elif f_reg and placed.get(key, {}).get(fk[1], 0) > 0:
                m = min(n, placed[key][fk[1]])
                placed[key][fk[1]] -= m
                if t_reg:
                    _add(key, tk[1], m)
                elif tk[0] == 'card':
                    d = placed[key]
                    d['__card__'] = d.get('__card__', 0) + m
                # back to Reinforcements: un-placed
    out = []
    minions = dict(out_minions)
    for i, (nat, reg, kind) in enumerate(order):
        if kind == 'MINION':
            out.append(minions[i])
            continue
        cnt = placed.get((nat, kind), {}).get(reg, 0)
        if cnt > 0:
            out.append((nat, reg, kind, cnt))
    return out


# --- W4-004: nazgul-class flights (plain Nazgul / Witch-king / Mouth) -------
def _naz_class(key):
    """Flight class of a _piece_key: 'NZ' plain Nazgul, 'WK' the Witch-
    king, 'MO' the Mouth of Sauron, None for any other figure."""
    if key[0] == 'nazgul':
        return 'NZ'
    if key[0] == 'named':
        n = str(key[1]).lower()
        if 'witch' in n:
            return 'WK'
        if 'mouth' in n:
            return 'MO'
    return None


def _naz_flight_hops(batch):
    """W4-004: the batch's literal leader-only region->region drags that
    carry nazgul-class figures, in record order: [(line, from, to,
    Counter{cls: n})].  A record with an army unit is an army move (#1);
    a record whose figures are all non-nazgul (companion, Saruman) is
    not a flight."""
    import rendered_replay as rr
    from collections import Counter
    hops = []
    for r in batch['records']:
        if r.get('op') != 'move':
            continue
        fk = rr.zone_kind(r.get('from', ''))
        tk = rr.zone_kind(r.get('to', ''))
        if fk[0] not in ('region', 'stronghold') or \
                tk[0] not in ('region', 'stronghold') or fk[1] == tk[1]:
            continue
        cls, has_unit = Counter(), False
        for t in _group_tokens(r.get('obj', '')):
            k, n = _piece_key(t)
            if k is None:
                continue
            if k[0] == 'unit':
                has_unit = True
                break
            c = _naz_class(k)
            if c is not None:
                cls[c] += n
        if has_unit or not cls:
            continue
        hops.append((r.get('line'), fk[1], tk[1], cls))
    return hops


def _nazgul_flows(batch, pre_nz):
    """W4-004: the human's nazgul-class flights of one batch as ordered
    flows [from, to, Counter{cls: n}], from the literal drags:

    (1) drag-and-return netting: a later B->A drag of a class cancels an
        earlier A->B drag of the same class (LIFO, anywhere in the batch,
        partial counts) - that figure never moved;
    (2) through-fold: a drag B->C of a figure that ARRIVED at B in this
        batch is the tail of ONE flight A->C (VASSAL takes one drag per
        region).  The Witch-king and the Mouth are unique, so their
        drag out of an arrival region always folds; plain Nazgul are
        fungible, so the literal reading (a nazgul that already stood at
        B flies on) is kept while B's pre-batch engine count pre_nz[B]
        covers it, and only the surplus folds - LIFO onto the latest
        arrivals (the top of the VASSAL stack).  That is exactly the
        engine's chain rule: a nazgul that flew this die may not fly
        again, every other one may;
    (3) same-(from, to) drags aggregate, first-seen order.
    Returns (flows, info) - info counts returned / folded / overdrawn
    figures (overdrawn = more figures dragged out than the batch or the
    state holds; the literal drag is kept and the matcher decides)."""
    from collections import Counter
    live = []
    info = Counter()
    # (1) returns
    for (ln, f, t, c0) in _naz_flight_hops(batch):
        h = [ln, f, t, Counter(c0)]
        for c in list(h[3]):
            n = h[3][c]
            for g in reversed(live):
                if n <= 0:
                    break
                if g[1] == t and g[2] == f and g[3][c] > 0:
                    m = min(n, g[3][c])
                    g[3][c] -= m
                    n -= m
                    info['returned'] += m
            h[3][c] = n
        if any(n > 0 for n in h[3].values()):
            live.append(h)
    live = [h for h in live if any(n > 0 for n in h[3].values())]
    # (2) through-fold
    pre = dict(pre_nz)
    arr = {}    # (region, cls) -> LIFO [[arrival hop, n]]
    i = 0
    while i < len(live):
        h = live[i]
        f, t, cls = h[1], h[2], h[3]
        for c in list(cls):
            n = cls[c]
            if n <= 0:
                continue
            stack = arr.setdefault((f, c), [])
            if c == 'NZ':
                lit = min(n, max(0, pre.get(f, 0)))
                pre[f] = pre.get(f, 0) - lit
            else:
                lit = 0 if stack else n
            forced = n - lit
            while forced > 0 and stack:
                src, m0 = stack[-1]
                m = min(m0, forced)
                src[3][c] -= m
                stack[-1][1] -= m
                if stack[-1][1] <= 0:
                    stack.pop()
                forced -= m
                info['folded'] += m
                if src[1] == t:
                    # A -> B -> A: the figure is back where it started
                    info['returned'] += m
                    continue
                fold = [src[0], src[1], t, Counter({c: m})]
                j = live.index(src)
                live.insert(j + 1, fold)
                i += 1
                arr.setdefault((t, c), []).append([fold, m])
            if forced > 0:
                info['overdrawn'] += forced
                lit += forced
            cls[c] = lit
            if lit > 0:
                arr.setdefault((t, c), []).append([h, lit])
        i += 1
    # (3) aggregate
    flows, idx = [], {}
    for h in live:
        cls = Counter({c: n for c, n in h[3].items() if n > 0})
        if not cls:
            continue
        key = (h[1], h[2])
        if key in idx:
            flows[idx[key]][2].update(cls)
        else:
            idx[key] = len(flows)
            flows.append([h[1], h[2], cls])
    return flows, info


def _emit_nazgul_flights(batch, st, side, legal, stats, turn):
    """W4-004: Shadow Character-die flights of nazgul-class figures.

    The die's action is the FIRST flow: nazgul_move from -> to with the
    exact count (no 'count' param = every plain nazgul of the source;
    include_wk when the Witch-king is in the flow; count 0 + include_wk
    = the Witch-king alone).  Every further flow is one leg of the
    engine's '__nazgul_moves__' chain PER FIGURE (the chain moves one
    nazgul per decision; the Witch-king's leg is the include_wk
    option), matched on the live option list after the previous leg
    is applied to a clone.  A leg the chain does not offer is dropped
    with a reason - never a synthetic non-menu action (the baseline's
    naz_hop_recovered die=-1 rows).  The Mouth flying in the same die
    as nazgul is one more leg of the chain (W4-027, USER ruling 1.262:
    the die-seeded chain offers the Mouth's include_mouth legs; before
    W4-027 the chain enumerated Nazgul sources only and the batch was
    dropped as nazchain_mouth_unsupported).  A Mouth-only batch is ONE
    mouth_move origin -> final region."""
    import rendered_replay as rr
    from game_state import Nazgul as _Nz
    pre = {}
    for k1, a1 in st.armies.items():
        n1 = sum(1 for l in a1.leaders if isinstance(l, _Nz))
        if n1:
            k9 = rr.sanitize(k1)
            pre[k9] = pre.get(k9, 0) + n1
    flows, info = _nazgul_flows(batch, pre)
    for k9 in ('returned', 'folded', 'overdrawn'):
        if info.get(k9):
            stats['detected']['naz_%s' % k9] += info[k9]
    naz = [f for f in flows if f[2].get('NZ') or f[2].get('WK')]
    mouth = [f for f in flows if f[2].get('MO')]
    if not naz:
        if not mouth:
            stats['drop_reasons'][
                'nazgul_flight_retracted' if info.get('returned')
                else 'leader_flight_unclassified'] += 1
            return None
        f0, t0 = mouth[0][0], mouth[0][1]
        hidx = _match_index(
            legal, lambda a: a.action_type == 'mouth_move' and
            rr.sanitize(str(a.params.get('from', ''))) == f0 and
            rr.sanitize(str(a.params.get('to', ''))) == t0)
        if hidx is None:
            stats['drop_reasons']['mouth_move_unmatched'] += 1
            return None
        stats['kinds_emit']['mouth_move'] += 1
        return _emit_sample(st, side, legal, hidx, 'mouth_move', turn)
    if mouth:
        # W4-027: the Mouth's flight becomes a chain leg (below); a leg
        # the engine's path rules refuse is naz_leg_unmatched
        stats['detected']['nazchain_mouth_leg'] += len(mouth)
    frm, to, cls = naz[0]
    k0 = int(cls.get('NZ', 0))
    wk0 = bool(cls.get('WK'))
    n_src = pre.get(frm, 0)
    if k0 > n_src:
        stats['drop_reasons']['nazgul_move_overdraw'] += 1
        return None

    def _first_ok(a):
        if a.action_type != 'nazgul_move':
            return False
        if rr.sanitize(str(a.params.get('from', ''))) != frm or \
                rr.sanitize(str(a.params.get('to', ''))) != to or \
                bool(a.params.get('include_wk')) != wk0:
            return False
        cnt = a.params.get('count')
        if k0 == n_src:
            return cnt is None
        return cnt is not None and int(cnt) == k0
    hidx = _match_index(legal, _first_ok)
    if hidx is None:
        stats['drop_reasons']['nazgul_move_unmatched'] += 1
        return None
    stats['kinds_emit']['nazgul_move'] += 1
    s1 = _emit_sample(st, side, legal, hidx, 'nazgul_move', turn)
    out1 = [s1] if s1 is not None else []
    legs = []
    for (f2, t2, c2) in naz[1:]:
        if c2.get('WK'):
            legs.append((f2, t2, 'WK'))
        for _ in range(int(c2.get('NZ', 0))):
            legs.append((f2, t2, 'NZ'))
    for (f3, t3, c3) in mouth:
        # W4-027: the Mouth's own leg (include_mouth), matched after the
        # Nazgul legs - the chain re-enumerates every leg from the live
        # state, so the order does not change what is offered
        legs.append((f3, t3, 'MO'))
    if legs:
        stats['detected']['naz_chain_enter'] += 1
        try:
            import ai as _ai
            st2 = _ai.clone_state(st)
            la1 = _ai.get_legal_actions(st2, side)
            h1 = _match_index(la1, _first_ok)
            if h1 is None:
                stats['drop_reasons']['naz_leg1_reclone'] += 1
                stats['drop_reasons']['naz_leg_unmatched'] += len(legs)
                stats['detected']['naz_leg_unmatched'] += len(legs)
            else:
                st2 = _ai.apply_action(st2, la1[h1])
                st2.pending_discard = None  # ledger-owned
                for li, (f2, t2, w2) in enumerate(legs):
                    la2 = _ai.get_legal_actions(st2, side)

                    def _hop_ok(a, _f=f2, _t=t2, _w=w2):
                        if a.action_type != 'card_option':
                            return False
                        ch = a.params.get('choices', {}) or {}
                        if ch.get('_naz_done'):
                            return False
                        return (rr.sanitize(str(ch.get(
                            'from_region', ''))) == _f and
                            rr.sanitize(str(ch.get(
                                'to_region', ''))) == _t and
                            bool(ch.get('include_wk')) == (_w == 'WK') and
                            bool(ch.get('include_mouth')) == (_w == 'MO'))
                    h2 = _match_index(la2, _hop_ok)
                    if h2 is None:
                        # uncapped twin of the drop reason (the top-15
                        # drop table hides a count of 1): the legs the
                        # engine's chain does not offer (W4-028)
                        stats['drop_reasons']['naz_leg_unmatched'] += \
                            len(legs) - li
                        stats['detected']['naz_leg_unmatched'] += \
                            len(legs) - li
                        break
                    stats['kinds_emit']['naz_leg2'] += 1
                    s2s = _emit_sample(st2, side, la2, h2, 'card_option',
                                       turn)
                    if s2s is not None:
                        out1.append(s2s)
                    st2 = _ai.apply_action(st2, la2[h2])
                    st2.pending_discard = None
        except Exception as _en2:
            stats['drop_reasons']['naz_leg2_exc:%s' % str(_en2)[:14]] += 1
    return out1 if len(out1) != 1 else (out1[0] if out1 else None)


def _leader_only_moves(batch):
    """DISTINCT (from,to) region->region moves that carry ONLY leader figures
    (named companion/minion/nazgul/plain leader, NO plain army unit), in first-
    seen order.  Each entry: (from, to, kinds_frozenset, named_list)."""
    import rendered_replay as rr
    seen, order = {}, []
    for r in batch['records']:
        if r.get('op') != 'move':
            continue
        fk = rr.zone_kind(r.get('from', ''))
        tk = rr.zone_kind(r.get('to', ''))
        if fk[0] not in ('region', 'stronghold') or \
                tk[0] not in ('region', 'stronghold') or fk[1] == tk[1]:
            continue
        kinds, named, has_unit = set(), [], False
        for t in _group_tokens(r.get('obj', '')):
            if t.strip().lower() == 'the fellowship':
                continue
            p = rr.parse_piece(t)
            if not p:
                continue
            if p[0] == 'unit':
                has_unit = True
            # only figure-leaders count; a VP / control-marker / die / card /
            # tile token is off-board bookkeeping (zone_kind maps the bare
            # 'victory points N' box to a pseudo-region, so guard here).
            if p[0] in ('leader', 'named', 'nazgul'):
                kinds.add(p[0])
                if p[0] == 'named':
                    named.append(p[1])
        if has_unit or not kinds:
            continue
        key = (fk[1], tk[1])
        if key not in seen:
            seen[key] = (frozenset(kinds), tuple(named))
            order.append(key)
    return [(f, t, seen[(f, t)][0], seen[(f, t)][1]) for (f, t) in order]


def _sep_moves(batch):
    """(companion_name, to_region_san) for each fellowship->region separation."""
    import rendered_replay as rr
    out = []
    for r in batch['records']:
        if r.get('op') != 'move':
            continue
        fk = rr.zone_kind(r.get('from', ''))
        tk = rr.zone_kind(r.get('to', ''))
        if fk[0] == 'fellowship' and tk[0] in ('region', 'stronghold'):
            p = rr.parse_piece(r.get('obj', ''))
            if p and p[0] == 'named':
                out.append((p[1], tk[1]))
    return out


_COMPANION_NAMES = None


def _companion_card_gesture(batch, after_line=0):
    """W1-008: the companion gesture of a Gwaihir the Windlord / We Prove
    Swifter play ("Move or separate from the Fellowship one Companion or
    one group of Companions"), read BY NAME over the batch's move records
    after `after_line` (the play line):

      * a fellowship -> region exit, as a single named token OR a group
        token ('group (Legolas, Gimli, Boromir, Merry) from Fellowship Of
        The Ring to Minas Tirith Stronghold', 2023-11-26 L469) = the
        SEPARATE arm; a later region -> region hop of the same name is its
        placement drag and moves the destination on;
      * region -> region hops of companions already on the board (single
        or a co-located group token) = the MOVE arm; a multi-hop chain is
        followed to the LAST hop's destination (2023-03-06 L990-994 five
        hops Orthanc -> Dol Amroth).  The chain is followed by NAME, not
        by from/to continuity: the client renders a STALE 'from' on rapid
        successive drags (2025-04-19 L558-560, three 'from Fangorn' hops
        of one Gandalf the White figure).

    Records carrying anything but companion figures (a unit, a leader
    token, a minion, a nazgul) are not the card's gesture and are skipped.
    Returns (arm, {name: dest_san}) with arm in ('sep', 'move', 'mixed')
    or (None, {}) when no companion moved."""
    global _COMPANION_NAMES
    import rendered_replay as rr
    if _COMPANION_NAMES is None:
        _COMPANION_NAMES = frozenset(rr.COMP_ALIAS.values())
    sep, mov = {}, {}
    for r in batch['records']:
        if r.get('op') != 'move':
            continue
        if int(r.get('line') or 0) <= int(after_line or 0):
            continue
        fk = rr.zone_kind(r.get('from', ''))
        tk = rr.zone_kind(r.get('to', ''))
        if tk[0] not in ('region', 'stronghold'):
            continue
        if fk[0] not in ('fellowship', 'region', 'stronghold'):
            continue
        names, other = [], False
        for t in _group_tokens(r.get('obj', '')):
            if t.strip().lower() == 'the fellowship':
                continue
            p = rr.parse_piece(t)
            if not p:
                continue
            if p[0] == 'named' and p[1] in _COMPANION_NAMES:
                names.append(rr.COMP_ALIAS.get(p[1].lower(), p[1]))
            else:
                other = True
        if other or not names:
            continue
        dest = rr.sanitize(str(tk[1]))
        for nm in names:
            if fk[0] == 'fellowship':
                sep[nm] = dest
            elif nm in sep:
                sep[nm] = dest         # placement drag of a separation
            else:
                mov[nm] = dest
    if sep and mov:
        return 'mixed', dict(sep, **mov)
    if sep:
        return 'sep', sep
    if mov:
        return 'move', mov
    return None, {}


def _crown_transform_flip(batch):
    """('crown', region) / ('transform', region) from a Will-of-West flip
    gesture ('flips Aragorn in R' = crowning; 'flips Gandalf the White in R' =
    transformation), or None."""
    import rendered_replay as rr
    for r in batch['records']:
        if r.get('op') != 'flip':
            continue
        ol = str(r.get('obj', '')).strip().lower()
        reg = r.get('region')
        if not reg or rr.zone_kind(reg)[0] not in ('region', 'stronghold'):
            continue
        if ol == 'aragorn':
            return ('crown', rr.sanitize(reg))
        if ol.startswith('gandalf the white'):
            return ('transform', rr.sanitize(reg))
    return None


def _nation_name(action):
    n = action.params.get('nation', '')
    return n.name if hasattr(n, 'name') else str(n).upper()


def _card_play_titles(batch):
    """EVERY event/table card TITLE the acting side played this batch, in
    line order â€” moves FROM the acting side's Hand to a PLAY AREA (Current
    FP/SA Card or BoardN) as (line, title) pairs.  The play-area gate
    excludes end-of-turn excess discards (which go to a discard / casualties
    zone, not a play slot) and combat-card discards.  The FIRST entry is the
    die-bought play; later entries are free chained plays (Ents Awake)."""
    want = (_HAND_ZONE_FP if batch['die_side'] == 'FREE_PEOPLES'
            else _HAND_ZONE_SH)
    out = []
    previous_move = None
    for r in batch['records']:
        move = ((str(r.get('player', '')), str(r.get('obj', '')).strip().casefold(),
                 _zone_norm(r.get('from', '')), _zone_norm(r.get('to', '')))
                if r.get('op') == 'move' else None)
        repeated = move is not None and move == previous_move
        previous_move = move
        if r.get('op') != 'move':
            continue
        # W1-001 (audit 2026-09-02): a play made through VASSAL's
        # right-click 'chooses a card' menu renders the from-zone as
        # '<Side> Hand (chosen)'; the exact compare never claimed the
        # batch, so the card's EFFECT trained as a plain die spend or the
        # batch fell to book_other.  Normalise the zone the way the hand
        # ledger's _norm_zone already does (strip a trailing parenthetical).
        if _zone_norm(r.get('from', '')) != want:
            continue
        to = _zone_norm(r.get('to', ''))
        if to == want:
            continue
        if to in _PLAY_AREA_DEST or _BOARD_DEST_RE.match(to):
            # One physical card cannot leave the same hand twice with no
            # intervening operation. Some client feeds repeat this exact
            # move notification; it is one settled play. A return/restage,
            # another card, or any intervening record remains distinct.
            if repeated:
                continue
            out.append((r.get('line', 0), r.get('obj')))
    return out


def _card_play_title(batch):
    """The FIRST (die-bought) card TITLE played this batch, or None."""
    plays = _card_play_titles(batch)
    return plays[0][1] if plays else None


# Event cards whose EVENT effect rolls dice for hits against an army (the
# card_list.csv event_effect 'roll ... die ... hit' family): the only event
# plays that legitimately render as a one-sided roll plus kills and so
# classify 'combat' without any battle (a CARD STRIKE).
# Event-text classes of the dual-titled cards, keyed by CARD ID through the
# CardDecoder route (_title_to_cid) so the rendered title's punctuation /
# accent variants ('Grond, Hammer of the Underworld', 'Nazgul Strike!')
# resolve to the one identity (W1-001, audit 2026-09-02).
# strike: the event rolls dice AGAINST an army (hits are its effect)
_STRIKE_EVENT_NAMES = (
    'The Ents Awake: Treebeard', 'The Ents Awake: Huorns',
    'The Ents Awake: Entmoot', 'The Spirit of Mordor', "Faramir's Rangers",
    'Dead Men of Dunharrow', 'Dreadful Spells', 'Return to Valinor',
    'The Eagles are Coming')
# roll: the event rolls dice WITHOUT scoring hits (Athelas heals, Candles
# corrupts, Nazgul Strike! rolls for the Hunt) - a one-sided roll by the
# actor is the card's own effect, never a combat roll
_ROLL_EVENT_NAMES = ('Athelas', 'Candles of Corpses', 'The Nazgul Strike!')
# open: the event OPENS a battle ('Attack ...' / 'a battle starts' / 'move
# or attack with an Army containing a Nazgul'): the play is real although
# a battle - with the actor's own commit and rolls - follows in the batch
_BATTLE_OPEN_NAMES = (
    'Help Unlooked For', 'The Fighting Uruk-hai',
    'Grond, Hammer of the Underworld', 'Corsairs of Umbar',
    'The Black Captain Commands', 'The Ringwraiths Are Abroad')
_TITLE_CID_CLASSES = None


def _title_class(title):
    """'strike' | 'roll' | 'open' | None for a rendered play title, by card
    id (the sanctioned CardDecoder identity import; rendered text stays the
    truth of WHAT was played)."""
    global _TITLE_CID_CLASSES
    if _TITLE_CID_CLASSES is None:
        def _cids(names):
            return frozenset(c for c in (_title_to_cid(n) for n in names)
                             if c)
        _TITLE_CID_CLASSES = {}
        for cls, names in (('strike', _STRIKE_EVENT_NAMES),
                           ('roll', _ROLL_EVENT_NAMES),
                           ('open', _BATTLE_OPEN_NAMES)):
            for c in _cids(names):
                _TITLE_CID_CLASSES[c] = cls
    cid = _title_to_cid(title)
    return _TITLE_CID_CLASSES.get(cid) if cid else None


# 1.267 (USER RULING 2026-09-03): "grond / fighting uruk hai are mandatory
# 3 rounds with no elite cost for those 3 rounds. it can only stop with
# elimination of either party. shadow can still press beyond 3 rounds
# though, with the standard cost".  The ENGINE already runs them so
# (event_effects mandatory_rounds 3; ai._set_post_round_phase never opens
# combat_round_result before round 3): NO press decision exists after
# rounds 1 and 2 of such a siege.
_MANDATORY_ROUND_NAMES = ('Grond, Hammer of the Underworld',
                          'The Fighting Uruk-hai')
_MANDATORY_ROUND_CIDS = None


def _mandatory_siege_rounds(batch, cards):
    """3 when the batch's Shadow actor PLAYED Grond / The Fighting Uruk-hai
    as its EVENT this batch - a hand exit into a play area whose card is
    NOT one of the rounds' committed combat cards (`cards` from
    _round_cards: the same title committed as Onslaught / Dread and
    Despair is that round's combat card, never the event) - else 0.
    Drop beats wrong: an ambiguous exit (the only Shadow card in its
    round window) keeps today's combat-card reading."""
    global _MANDATORY_ROUND_CIDS
    if _MANDATORY_ROUND_CIDS is None:
        _MANDATORY_ROUND_CIDS = frozenset(
            str(c) for c in (_title_to_cid(n) for n in _MANDATORY_ROUND_NAMES)
            if c)
    if not _MANDATORY_ROUND_CIDS or batch.get('die_side') != 'SHADOW':
        return 0
    committed = set()
    for cr in (cards or []):
        for slot in ('fp', 'sh'):
            if cr.get(slot) is not None:
                committed.add(str(cr.get(slot)))
    for _ln, title in _card_play_titles(batch):
        cid = _title_to_cid(title)
        if cid and str(cid) in _MANDATORY_ROUND_CIDS \
                and str(cid) not in committed:
            return 3
    return 0


def _batch_actor(batch):
    """User name of the player who staged this batch's die (the ACTING
    player), None when no staging record exists."""
    recs = batch.get('records') or []
    for r in recs:
        if r.get('op') == 'move' and r.get('die_stage'):
            return r.get('player')
    return recs[0].get('player') if recs else None


def _card_strike(batch):
    """True when the batch is a CARD STRIKE: the acting side's event play
    whose effect rolls dice against an army, so the batch classifies
    'combat' although no battle was fought.  ALL of:
      (1) every logged roll is by ONE player;
      (2) that player is the ACTOR (die stager) - in a battle where only
          the DEFENDER rolled (attacker took an automatic hit) the sole
          roller is the opponent;
      (3) the played title's event effect is a strike;
      (4) the opponent committed no combat card (a hand exit of the OTHER
          side into a Current-card zone proves a battle: Dead Men committed
          as Sudden Strike against a Breaking of the Fellowship commit).
    W1-001 (audit 2026-09-02): the former one-sided-roll + any-title test
    misread attacker-skipped-roll battles as strikes once the VASSAL
    '(chosen)' combat-card commits became visible to _card_play_titles."""
    chunks = [c for c in (batch.get('rng') or [])
              if c.get('kind') == 'roll_num']
    if not chunks:
        return False
    rollers = {c.get('player') for c in chunks}
    if len(rollers) != 1:
        return False
    actor = _batch_actor(batch)
    if actor is None or next(iter(rollers)) != actor:
        return False
    title = _card_play_title(batch)
    if title is None:
        return False
    if _title_class(title) != 'strike':
        return False
    other = (_HAND_ZONE_SH if batch.get('die_side') == 'FREE_PEOPLES'
             else _HAND_ZONE_FP)
    for r in batch.get('records') or []:
        if r.get('op') == 'move' \
                and _zone_norm(r.get('from', '')) == other \
                and _zone_norm(r.get('to', '')) in _PLAY_AREA_DEST:
            return False
    return True


def _commit_in_missed_battle(batch):
    """True when the acting side's hand exit this batch is a COMBAT-CARD
    COMMIT inside a battle the batch classifier missed (no figure kill in
    a region: the defender conceded, the attacker skipped his roll, the
    hits were all absorbed) - such a batch classifies 'other' and its
    hand exit would otherwise train as an event play (W1-001, audit
    2026-09-02: 2023-01-20 L1066 The Day Without Dawn/Relentless Assault
    with an M die, 2023-03-16 L1014 Stormcrow/Great Host with a C die).
    Evidence of the battle, any one of:
      E1 the OPPONENT's hand exits into a Current-card zone (a commit pair);
      E2 the OPPONENT rolled (no event effect makes the opponent roll);
      E3 the ACTOR rolled alone although the played title's event text
         rolls no dice (the roll can only be a combat roll).
    Never on combat/hunt kinds (their own paths), never with a hunt tile.
    A title whose event OPENS a battle (Help Unlooked For, Corsairs, Grond,
    The Ringwraiths Are Abroad: the play is real, the battle with its
    rolls and commits is the play's consequence) is a commit only by E1
    with the opponent's commit rendered BEFORE the title's own hand exit -
    an event-opened battle can never render the opponent's commit first.
    Drop beats wrong: the row lost is the missed battle's commit, which is
    the combat family's to recover."""
    if batch.get('kind') in ('combat', 'combat?', 'hunt'):
        return False
    rng = batch.get('rng') or []
    if any(c.get('kind') == 'draw_tile' for c in rng):
        return False
    title = _card_play_title(batch)
    if title is None:
        return False
    cls = _title_class(title)
    side = batch.get('die_side')
    own = _HAND_ZONE_FP if side == 'FREE_PEOPLES' else _HAND_ZONE_SH
    other = _HAND_ZONE_SH if side == 'FREE_PEOPLES' else _HAND_ZONE_FP
    own_line = None
    for r in batch.get('records') or []:
        if r.get('op') != 'move':
            continue
        f = _zone_norm(r.get('from', ''))
        t = _zone_norm(r.get('to', ''))
        if f == own and (t in _PLAY_AREA_DEST or _BOARD_DEST_RE.match(t)):
            if own_line is None:
                own_line = int(r.get('line') or 0)
        elif f == other and t in _PLAY_AREA_DEST:
            if cls != 'open' or own_line is None:
                return True                               # E1
    if cls == 'open':
        return False
    chunks = [c for c in rng if c.get('kind') == 'roll_num']
    if not chunks:
        return False
    actor = _batch_actor(batch)
    if actor is None or {c.get('player') for c in chunks} - {actor}:
        return True                                       # E2
    return cls not in ('strike', 'roll')                  # E3


# --------------------------------------------------------------------------
# per-game CHAT index (fuzzy-region battle-card evidence, spec 1.216)
# --------------------------------------------------------------------------
_CHAT_INDEX = None          # per-game [(line_no, lowercase_text)]
_COMBAT_CHAT_INDEX = []     # same source, retaining speaker for ff ownership
_CHAT_TAG_SKIP = {'game', 'replay', 'address', 'observer'}
_CHAT_BODY_RE = re.compile(r'^<([^>]+)>\s*(.+?)\s*$')
# region words too generic to identify a region on their own
_REGION_STOPWORDS = {'north', 'south', 'east', 'west', 'the', 'of', 'old',
                     'far', 'near'}


def _build_chat_index(rendered_path):
    global _COMBAT_CHAT_INDEX
    _COMBAT_CHAT_INDEX = []
    out = []
    with open(rendered_path, encoding='utf-8', errors='ignore') as f:
        for i, raw in enumerate(f, 1):
            s = raw.strip()
            if not s.startswith('%'):
                continue
            m = _CHAT_BODY_RE.match(s[1:].strip())
            if m and m.group(1).strip().lower() not in _CHAT_TAG_SKIP:
                out.append((i, m.group(2).lower()))
                _COMBAT_CHAT_INDEX.append((i, m.group(2).lower(), m.group(1).strip()))
    return out


def _batch_chats(batch):
    """Chat texts inside the batch's line window."""
    if not _CHAT_INDEX:
        return []
    # W2-031: a folded hesitation opens the window at ITS stage line
    lo = batch.get('_chat_lo') or batch.get('start_line') or 0
    hi = batch.get('end_line') or 10 ** 9
    return [t for (ln, t) in _CHAT_INDEX if lo <= ln < hi]


def _chat_names_region(reg, chats):
    """True when any chat line plausibly names sanitized region `reg`:
    full name, word-initial abbreviation ('NI' = north ithilien), or a
    >=3-char prefix of a DISTINCTIVE (non-stopword) region word."""
    words = [w for w in str(reg).split() if w]
    if not words:
        return False
    abbr = ''.join(w[0] for w in words)
    dwords = [w for w in words if w not in _REGION_STOPWORDS and len(w) >= 4]
    for t in chats:
        if reg in t:
            return True
        toks = re.findall(r'[a-z]+', t)
        if len(words) > 1 and abbr in toks:
            return True
        for w in dwords:
            if any(tok == w or (len(tok) >= 3 and w.startswith(tok))
                   for tok in toks):
                return True
    return False


def _table_kill_title(batch):
    """The OPPONENT table card this batch deliberately removed (ruling
    1.149), or None.

    The gesture (corpus 2023-01-20 CircleDakkon/GenericPie L583-591) is
    a play-on-table card dragged OFF its Board slot into the ACTOR's
    Current-Card zone, paid for with a die (+ two hand discards for the
    Shadow arm, a Will die / Elven Ring for the FP arm).  The payment
    is not tested here: the ENGINE's menu already encodes who may pay
    what, and _match_index only matches when the engine offered the
    action - so an unpaid drag simply finds no match and is dropped,
    which is the low-false-positive doctrine.

    Distinguished from the OWNER shuffling his own card between slots
    (destination must NOT be another Board slot) and from the seat that
    PLAYS a card (that moves hand -> slot, the opposite direction, and
    _card_play_title claims it first)."""
    import rendered_replay as rr
    for r in batch['records']:
        if r.get('op') != 'move':
            continue
        frm = str(r.get('from', '')).strip().lower()
        if not _BOARD_DEST_RE.match(frm):
            continue
        to = str(r.get('to', '')).strip().lower()
        if _BOARD_DEST_RE.match(to):
            continue                      # slot -> slot: not a kill
        if rr.zone_kind(r.get('to', ''))[0] not in ('card',):
            continue                      # must land in a card area
        obj = str(r.get('obj', ''))
        if '/' in obj:
            return obj
    return None


def _emit_table_kill(batch, st, side, legal, title, stats):
    """ONE `discard_table_card` sample (USER 2026-08-20: "give it its
    own category and extract the training signal").

    Until now this family was DETECTION-ONLY - the extractor's own
    _catalog_unemitted docstring lists `discard_table_card` among the
    classes it counts but never emits - so the corpus carried zero
    examples and the net scored the act through its play-event head."""
    # ---- AUTO-DISCARDS ARE NOT DELIBERATE REMOVALS --------------------
    # A play-on-table card leaving its slot looks identical whether the
    # opponent PAID to remove it (1.149) or the card simply expired.
    # Emitting an expiry as a paid action would teach the net that a
    # free discard was a deliberate one, so each exclusion below is a
    # label-purity gate, not an optimisation.
    _pt = _title_to_ptype(title)
    if _pt in _TABLEKILL_EXCLUDED:
        # USER 2026-08-20: "ignore denethor's folly from the discard
        # extraction. it doesn't happen enough to matter. most of the
        # time it gets discarded due to minas tirith no longer being
        # under siege" - i.e. game_logic.cleanup_sieges, an automatic
        # removal wearing the same gesture.
        stats['drop_reasons']['tablekill_excluded_type:%s' % _pt] += 1
        return None
    if _pt == 'PALANTIR_OF_ORTHANC' and _saruman_eliminated(batch):
        # USER 2026-08-20: "for palantir of orthanc, make sure we are
        # filtering the discards that happen due to saruman's death".
        # persistent_effects.check_saruman_eliminated_discards drops it
        # for free; no die is paid and no choice is made.
        stats['drop_reasons']['tablekill_saruman_autodiscard'] += 1
        return None
    if not _table_kill_die_spend(batch):
        # THE GENERAL DISCRIMINATOR, and the one log_replay already
        # trusts for this act ("the removal is provably die-PAID"):
        # every deliberate 1.149 removal spends an action die; every
        # automatic one (siege end, Saruman elimination, effect expiry,
        # turn-boundary tidy) spends nothing.
        stats['drop_reasons']['tablekill_not_die_paid'] += 1
        return None
    cid = _title_to_cid(title)
    if cid is None:
        stats['drop_reasons']['tablekill_title_unresolved'] += 1
        return None
    # The victim is already on the table: _mk_state_with_die installs
    # the persistent cards for EVERY emitted sample now, which is what
    # gives the un-chosen removals their negative evidence.  So this
    # matches against the SAME state and menu the sample is encoded
    # with - a private clone here would have matched one menu and
    # emitted another, and the label index is only meaningful against
    # the menu it was chosen from.
    hidx = _match_index(
        legal,
        lambda a: a.action_type == 'discard_table_card'
        and str(a.params.get('card_id')) == str(cid))
    if hidx is None:
        # the engine did not offer this removal in the reconstructed
        # state (payment not reconstructible, card not modelled on the
        # table, ...) - drop, never guess an index
        stats['drop_reasons']['tablekill_no_menu_match'] += 1
        return None
    # NAME THE PAYMENT (USER 2026-08-20: "I want it to log exactly
    # which cards were discarded. cross reference the log that
    # contains card info").  The rendered kill lines are MASKED
    # ('Shadow Character Card'), so the identities come from the hand
    # ledger - the same CardDecoder commitment reconstruction the rest
    # of the extractor trusts - by diffing the hand across the discard
    # lines.  Reported, never guessed: an unresolvable diff is logged
    # as such and the sample is still emitted (the LABEL is the chosen
    # action; the payment is provenance).
    paid = _table_kill_payment(batch, side, stats)
    stats['tablekill_paid'][
        '%s|%s' % (str(cid), '+'.join(paid) if paid else '?')] += 1
    stats['kinds_emit']['discard_table_card'] += 1
    kill_smp = _emit_sample(st, side, legal, hidx, 'discard_table_card',
                            batch['turn'])
    # ---- EX-8 PAYMENT ROW (DET census P0-b, 2026-08-30) ---------------
    # The Shadow arm pays one Army-type + one Character card - a learned
    # pending_card_option (ai._open_table_kill_payment) whose human
    # answer the ledger diff above already NAMED; until now it was
    # stored only as the provenance counter, never as a training row.
    pay_smp = None
    from game_state import Side as _S149
    if side == _S149.SHADOW and len(paid) == 2:
        try:
            pay_smp = _emit_table_kill_payment_row(
                batch, st, legal, hidx, paid, stats)
        except Exception as e:
            stats['drop_reasons']['tkpay_exc:%s' % str(e)[:16]] += 1
    if pay_smp is not None:
        return [kill_smp, pay_smp]
    return kill_smp


def _emit_table_kill_payment_row(batch, st, legal, hidx, paid, stats):
    """EX-8 (P0-b): apply the matched kill on a clone - the learned-path
    executor opens the payment pending - then match the ledger-named
    pair ((Army-type, Character) resolved by the deck classifier) and
    emit ONE card_option sample.  Every gap drops with its own counter,
    never a guess."""
    import ai
    from cards import get_card as _gc
    from game_state import Side
    army = [c for c in paid
            if getattr(_gc(c), 'card_type', None) == 'Army']
    chars = [c for c in paid if '_Character_' in str(c)]
    if len(army) != 1 or len(chars) != 1 or army[0] == chars[0]:
        stats['drop_reasons']['tkpay_pair_unclassified'] += 1
        return None
    s3 = ai.apply_action(st, legal[hidx])
    s3._rx_full_pools = getattr(st, '_rx_full_pools', None)
    pco = getattr(s3, 'pending_card_option', None)
    if not (isinstance(pco, dict) and pco.get('table_kill_payment')):
        stats['drop_reasons']['tkpay_no_pending'] += 1
        return None
    la = ai.get_legal_actions(s3, Side.SHADOW)
    copts = [a for a in la if a.action_type == 'card_option']
    if len(copts) >= 2:
        _pend_note(stats, 'table_kill_payment', 'open')
    h4 = _match_index(
        la, lambda a: a.action_type == 'card_option'
        and (a.params.get('choices') or {}).get('army_card') == army[0]
        and (a.params.get('choices') or {}).get('character_card')
        == chars[0])
    if h4 is None:
        stats['drop_reasons']['tkpay_no_menu_match'] += 1
        return None
    if len(copts) < 2:
        # single legal pair = forced, a correct non-decision
        stats['detected']['tkpay_forced'] += 1
        return None
    stats['kinds_emit']['table_kill_payment'] += 1
    _pend_note(stats, 'table_kill_payment', 'emit')
    return _emit_sample(s3, Side.SHADOW, la, h4, 'table_kill_payment',
                        batch['turn'])


# Table cards whose removal is NOT extracted as a deliberate 1.149
# kill (USER 2026-08-20).  Denethor's Folly is dominated by the
# automatic siege-end discard and is too rare to be worth the
# false-positive risk.
_TABLEKILL_EXCLUDED = ('DENETHORS_FOLLY',)


def _saruman_eliminated(batch):
    """Does this batch eliminate Saruman?  His death auto-discards the
    Palantir (persistent_effects.check_saruman_eliminated_discards),
    which is not a paid removal.  Rendered as 'kills Saruman in
    Orthanc' (the only elimination form in the corpus)."""
    for r in batch['records']:
        if r.get('op') == 'kill' and any(
                'saruman' in str(p).lower()
                for p in (r.get('pieces') or ())):
            return True
        if r.get('op') == 'move' \
                and 'saruman' in str(r.get('obj', '')).lower() \
                and 'casualt' in str(r.get('to', '')).lower():
            return True
    return False


def _batch_dice_spent(batch, faces):
    """W4-026: how many of the batch side's action dice showing one of
    `faces` were killed in Current Action by this batch (min 1).  Two
    Muster dice staged together and killed as '[2]Shadow Action Die
    (H)' (2023-01-05 L204-208: Saruman AND the Witch-king; 2025-05-20
    L1792-1795: two Sauron Elites at Mount Gundabad) are two dice'
    worth of muster, not an over-capacity single die."""
    side = batch.get('die_side')
    want = set(str(f or '').upper() for f in faces if f)
    n = 0
    for r in batch.get('records', []) or []:
        if r.get('op') != 'kill':
            continue
        if 'current action' not in str(r.get('zone', '')).strip().lower():
            continue
        for (sd, face) in _die_tokens(' , '.join(r.get('pieces', []) or [])):
            if sd == side and face in want:
                n += 1
    return max(1, n)


def _table_kill_die_spend(batch):
    """True when the acting side put an action die into Current Action
    this batch - the mark of a PAID removal.  Automatic discards
    (siege end, Saruman death, effect expiry) spend no die."""
    for r in batch['records']:
        if r.get('op') != 'move':
            continue
        if 'Action Die' not in str(r.get('obj', '')):
            continue
        if 'current action' in str(r.get('to', '')).strip().lower():
            return True
    return False


# Table cards whose paid removal is the SHADOW arm of 1.149 - paid with
# hand cards (APTG / Tom Bombadil text: "using any one Action Die and
# discarding one Army [Event] card and one Character card from his
# hand").  The FP arm (Palantir / Denethor's Folly) pays with a Will die
# or a die + Elven Ring and never touches the hand.
_TABLEKILL_SHADOW_PAID = ('POWER_TOO_GREAT', 'TOM_BOMBADIL')


def _table_kill_shown_payments(batch, side):
    """W1-010: the payment cards the Shadow SHOWED face-up instead of
    killing them hidden in hand - rendered as the same hand -> Current
    card-zone move a play makes (and then killed in Current Action
    together with the die and the table card) - as (line, title,
    card_id) triples in line order; card_id is None when the rendered
    title does not resolve.  Empty for the FP arm (no cards paid)."""
    from game_state import Side
    if side != Side.SHADOW:
        return []
    return [(int(ln or 0), title, _title_to_cid(title))
            for ln, title in _card_play_titles(batch)]


def _table_kill_payment(batch, side, stats):
    """The card ids the actor DISCARDED to pay for a 1.149 removal.

    The rendered lines name only the deck ('Shadow Strategy
    Card(Army)'), so identity comes from the hand ledger: the hand just
    BEFORE the first hand-kill line minus the hand just AFTER the last
    one.  Returns [] when the ledger cannot answer (no raw log /
    passwords), which is reported rather than filled in.

    W1-010: a payment card SHOWN face-up (hand -> Current card zone,
    see _table_kill_shown_payments) counts like a hidden kill; its
    rendered title names it and the ledger diff must agree."""
    from game_state import Side
    kills = [r for r in batch['records']
             if r.get('op') == 'kill'
             and 'hand' in str(r.get('zone', '')).lower()
             and r.get('line')]
    shown = _table_kill_shown_payments(batch, side)
    if any(cid is None for _ln, _t, cid in shown):
        stats['drop_reasons']['tablekill_payment_shown_unresolved'] += 1
        return []
    if not kills and not shown:
        # SIDE-AWARE, because the two arms of 1.149 pay differently:
        #   FP     removes the Palantir / Denethor's Folly with a
        #          Will-of-the-West die ALONE, or any die + an Elven
        #          Ring - NO cards are discarded, so there is nothing
        #          to name and 'unresolved' would be a lie.
        #   SHADOW removes A Power too Great / Tom Bombadil by
        #          discarding one Army-type + one Character card, so
        #          finding no hand kills IS a gap and must say so.
        if side == Side.FREE_PEOPLES:
            return ['(die/ring - no cards discarded)']
        stats['drop_reasons']['tablekill_payment_no_discards_seen'] += 1
        return []
    if _LEDGER is None:
        stats['drop_reasons']['tablekill_payment_unledgered'] += 1
        return []
    tag = 'fp' if side == Side.FREE_PEOPLES else 'sh'
    _lines = ([int(r['line']) for r in kills]
              + [ln for ln, _t, _c in shown])
    lo = min(_lines)
    hi = max(_lines)
    try:
        before = list(_LEDGER.hand_at(tag, lo))
        after = list(_LEDGER.hand_at(tag, hi + 1))
    except Exception:
        stats['drop_reasons']['tablekill_payment_ledger_exc'] += 1
        return []
    gone = list(before)
    for c in after:
        if c in gone:
            gone.remove(c)
    if len(gone) != len(kills) + len(shown):
        # the diff must account for EXACTLY the kills we saw, or it is
        # not evidence - say so instead of reporting a partial payment
        stats['drop_reasons'][
            'tablekill_payment_diff:%d/%d' % (len(gone),
                                              len(kills) + len(shown))] += 1
        return []
    for _ln, _t, cid in shown:
        if str(cid) not in [str(c) for c in gone]:
            # the shown title and the ledger disagree on what left the
            # hand - not evidence either
            stats['drop_reasons']['tablekill_payment_shown_unledgered'] += 1
            return []
    return sorted(str(c) for c in gone)


def _emit_play_event(batch, st, side, legal, title, stats):
    """Legality-match the human's EVENT-CARD PLAY against the (hand-installed)
    legal set and emit ONE BC sample.  Emit-or-DROP: a card play never falls
    through to a board matcher (its effects must not be mislabelled)."""
    cid = _title_to_cid(title)
    if cid is None:
        stats['drop_reasons']['play_title_unresolved'] += 1
        return None
    side_tag = 'fp' if batch['die_side'] == 'FREE_PEOPLES' else 'sh'
    # a unique card is played at most once per game â€” dedupe undo/redo phantoms
    if cid in stats['played_cids']:
        stats['drop_reasons']['play_dupe_undo'] += 1
        return None
    # an unresolved generic discard of this card's DECK would leave a phantom
    # (already-gone) card in the reconstructed hand, polluting the menu -> drop
    # (purity, never a guess).  Only the ambiguous deck is affected.
    # W1-002: only while the phantom can actually sit in the hand - the
    # window from the unresolved removal to the named play that resolves
    # it.  The old game-wide flag dropped every play of that deck in the
    # whole game (28 rows over 4 games) - plays BEFORE the removal and
    # after the resolution are from a clean hand.
    deck_letter = 'C' if 'Character' in str(cid) else 'S'
    if _LEDGER is not None and _LEDGER.is_deck_ambiguous_at(
            side_tag, deck_letter, batch['start_line'],
            batch.get('end_line') or batch['start_line']):
        stats['drop_reasons']['play_ambiguous_hand'] += 1
        return None
    hidx = _match_index(legal, lambda a: a.action_type == 'play_event' and
                        str(a.params.get('card_id')) == str(cid))
    # W1-021 (2026-09-02): _match_index's single-candidate short cut binds
    # the play to whichever die can legally play the card - when that
    # die's face is not the face the human SPENT, the row would carry the
    # wrong die (2024-05-21 L149 Faramir's Rangers: C die + Narya, no
    # 'changes' line, the ring probe is A/P-ambiguous; the engine offers
    # the Army card only on the spare W die at idx 2). Refuse unproven
    # payment. Only the state the face context was set
    # for (_mk_state_with_die) is judged; a lead re-faced by the ring
    # inference (W3-016) carries _MM_FACE itself and passes.
    if hidx is not None and _MM_FACE is not None and _MM_STATE is st:
        _a21 = legal[hidx]
        _dl21 = _MM_STATE.dice.get(_MM_SIDE, [])
        _f21 = None
        if _a21.die_idx is not None and 0 <= _a21.die_idx < len(_dl21):
            _f21 = _dl21[_a21.die_idx].current
        if _f21 is not None and _f21 is not _MM_FACE:
            stats['detected']['play_event_face_mismatch'] += 1
            hidx = None
    if hidx is not None:
        stats['played_cids'].add(cid)
        stats['kinds_emit']['play_event'] += 1
        return _emit_sample(st, side, legal, hidx, 'play_event', batch['turn'])
    # A named card gesture does not establish this reconstructed menu's
    # requirements, timing or payment. Preserve history in replay, but never
    # manufacture a free card play as an expert policy target.
    stats['drop_reasons']['play_event_recovery_not_in_public_menu'] += 1
    return None


_FP_HUNTBOX_IDX = []
_FP_HUNTBOX_FACES = []
_FP_HUNTBOX_BATCHES = {}


def _paid_hunt_restage_return(batch):
    """Prove that a paid move's second Hunt deposit is the same die.

    Preserve the actual Fellowship movement and draw records. This handles
    one unique, newly spent natural movement die taken briefly from Hunt
    to Current Action and returned before any RNG or other game effect.
    Unproven histories retain all entries and the physical-pool refusal.
    """
    if not _is_paid_fellowship_move(batch):
        return None
    records = batch.get('records') or []
    line = int(batch.get('start_line') or 0)
    stage_index = next((i for i, record in enumerate(records)
                        if int(record.get('line') or 0) == line
                        and record.get('die_stage')), None)
    if stage_index is None:
        return None
    stage = records[stage_index]
    zone = lambda value: str(value or '').strip().lower()
    box = 'the hunt for the ring'
    tokens = _die_tokens(stage.get('obj', ''))
    if (stage.get('op') != 'move' or zone(stage.get('from')) != 'fp dice'
            or zone(stage.get('to')) != box or len(tokens) != 1
            or tokens[0] not in (('FREE_PEOPLES', 'C'), ('FREE_PEOPLES', 'W'))):
        return None
    face = tokens[0][1]
    remainder = _DIE_TOKEN_RE.sub('', str(stage.get('obj', '')))
    if re.sub(r'group|[\s,()]', '', remainder, flags=re.I):
        return None
    pool = (batch.get('_rx_pools') or {}).get('FREE_PEOPLES') or {}
    if (list(pool.get('unused') or []).count(face) != 1
            or face in (pool.get('used') or [])):
        return None
    tail = [record for record in records[stage_index + 1:]
            if record.get('op') != 'chat']
    if len(tail) < 3:
        return None
    outgoing, movement, returned = tail[:3]
    if not (line < int(outgoing.get('line') or 0)
            < int(movement.get('line') or 0) < int(returned.get('line') or 0)):
        return None
    for rng in batch.get('rng') or []:
        rng_line = rng.get('line')
        if not isinstance(rng_line, int) or line < rng_line <= int(returned['line']):
            return None
    if any(record.get('op') != 'move' or record.get('obj') != stage.get('obj')
           or record.get('player') != stage.get('player')
           for record in (outgoing, returned)):
        return None
    if (zone(outgoing.get('from')) != box
            or zone(outgoing.get('to')) != 'current action'
            or zone(returned.get('from')) != 'current action'
            or zone(returned.get('to')) != box):
        return None
    if (movement.get('op') != 'move'
            or movement.get('player') != stage.get('player')):
        return None
    before = batch['truth_before']['fel']
    if before.get('mordor'):
        origin = re.search(r'\bmordor\s+(\d+)\b', str(movement.get('from', '')), re.I)
        target = re.search(r'\bmordor\s+(\d+)\b', str(movement.get('to', '')), re.I)
        if (str(movement.get('obj', '')).strip().lower() != 'the fellowship'
                or not origin or not target
                or int(origin.group(1)) != int(before.get('step', 0))
                or int(target.group(1)) != int(origin.group(1)) + 1):
            return None
    else:
        origin = re.fullmatch(r'Fellowship Track (\d+)', str(movement.get('from', '')), re.I)
        target = re.fullmatch(r'Fellowship Track (\d+)', str(movement.get('to', '')), re.I)
        if (str(movement.get('obj', '')).strip().lower() != 'fsp'
                or not origin or not target
                or int(origin.group(1)) != int(before.get('progress', 0))
                or int(target.group(1)) != int(origin.group(1)) + 1):
            return None
    # Further die manipulations make the identity witness insufficient.
    if any(_die_tokens(record.get('obj', ''))
           or any(_die_tokens(str(piece)) for piece in record.get('pieces', []) or [])
           or (record.get('op') == 'other'
               and 'action die' in str(record.get('raw', '')).lower())
           for record in tail[3:]):
        return None
    return {'stage_line': stage['line'], 'out_line': outgoing['line'],
            'return_line': returned['line'], 'face': face}


def _normalize_physical_combat_corrections(batches, rendered_path, raw_path=None):
    """Prove repeated/corrected casualty drags without changing real losses.

    Raw figure IDs, identical roll prefixes and both literal move streams
    distinguish a second fungible casualty from a second drag of the same
    figures. The only cancellable interval contains casualty parking moves;
    another roll, card, action die or board move is a boundary.
    """
    import log_census as lc
    import rendered_replay as rr
    from piece_table_authoritative import AUTH_ARMY_PIECES
    from location_codes_authoritative import AUTH_LOCATION_CODES
    offboard={'cas','reinf','card','dice','hunt'}
    def cargo(record):
        result=Counter()
        for token in _group_tokens(record.get('obj','')):
            piece=rr.parse_piece(token)
            if piece is None or piece[0]!='unit':return None
            result[(piece[1],piece[2])]+=piece[3]
        return tuple(sorted(result.items())) if result else None
    def parking(record):
        return (record.get('op')=='move' and cargo(record) is not None
                and rr.zone_kind(record.get('from',''))[0] in ('region','stronghold')
                and rr.zone_kind(record.get('to',''))[0] in offboard)
    candidates=[]
    for batch in batches:
        if batch.get('kind') not in ('combat','combat?'):continue
        records=batch['records']
        for i,first in enumerate(records):
            if not parking(first):continue
            for j in range(i+1,len(records)):
                second=records[j]
                duplicate=(j==i+1 and parking(second) and cargo(first)==cargo(second)
                           and all(first.get(k)==second.get(k) for k in ('from','to')))
                reverse=(second.get('op')=='move' and cargo(first)==cargo(second)
                         and first.get('from')==second.get('to') and first.get('to')==second.get('from')
                         and all(parking(r) for r in records[i+1:j]))
                if duplicate or reverse:candidates.append((batch,first,second,'duplicate' if duplicate else 'return'))
                if not parking(second):break
    if not candidates:return
    if raw_path is None:raw_path=_raw_log_path(os.path.basename(str(rendered_path)).replace('.rendered.txt',''))
    if not raw_path or not os.path.isfile(raw_path):return
    raw=lc.undo_redo_net(read_raw_log_text(raw_path).splitlines())
    raw_moves=[];turn=1;rolls=0
    for line,text in enumerate(raw,1):
        marker=re.search(r'<~Game\.7~>(\d+)\b',text)
        if marker:turn=int(marker[1]);rolls=0
        if '<~Controls.1~>' in text:rolls+=1
        move=lc.MOVE_RE.search(text)
        if move is None:continue
        ids=move[1].split();pieces=[AUTH_ARMY_PIECES.get(pid) for pid in ids]
        if len(ids)!=len(set(ids)) or not pieces or any(p is None or p[1] not in ('REGULAR','ELITE') for p in pieces):continue
        raw_moves.append({'line':line,'ids':ids,'turn':turn,'rolls':rolls,
            'player':text.split('<~',1)[0].lstrip('$'),'from':AUTH_LOCATION_CODES.get(move[2]),
            'to':AUTH_LOCATION_CODES.get(move[3]),'cargo':tuple(sorted(Counter(pieces).items()))})
    prefixes={};turn=None;rolls=0
    for batch in batches:
        if batch['turn']!=turn:turn=batch['turn'];rolls=0
        for record in batch['records']:
            if record.get('op')=='roll_num':rolls+=1
            prefixes[id(record)]=(turn,rolls)
    def matches(record):
        turn,rolls=prefixes[id(record)]
        return [r for r in raw_moves if r['turn']==turn and r['rolls']==rolls
                and r['cargo']==cargo(record) and all(record.get(k)==r[k] for k in ('player','from','to'))]
    removed=set()
    for batch,first,second,kind in candidates:
        if id(first) in removed or id(second) in removed:continue
        starts,ends=matches(first),matches(second)
        pairs=[(a,b) for a in starts for b in ends if a['line']<b['line'] and set(a['ids'])==set(b['ids'])]
        if len(pairs)!=1:continue
        a,b=pairs[0]
        if kind=='duplicate':
            if b['line']!=a['line']+1:continue
            removed.add(id(second))
        else:
            window=[];safe=True
            for line in range(a['line'],b['line']-1):
                text=raw[line]
                if not text.strip() or (re.fullmatch(r'<([^<>~]+)> .*',text)
                        and not text.startswith(('<auto>','<game>')) and '<~' not in text):continue
                event=next((r for r in raw_moves if r['line']==line+1),None)
                if (event is None or set(event['ids']).intersection(a['ids'])
                        or rr.zone_kind(event['from'] or '')[0] not in ('region','stronghold')
                        or rr.zone_kind(event['to'] or '')[0] not in offboard):
                    safe=False;break
                window.append(event)
            public=[r for r in batch['records'] if int(first['line'])<int(r['line'])<int(second['line'])]
            if (not safe or len(public)!=len(window)
                    or any(cargo(r)!=w['cargo'] or any(r.get(k)!=w[k] for k in ('player','from','to'))
                           for r,w in zip(public,window))):continue
            removed.update((id(first),id(second)))
        batch.setdefault('_rx_physical_combat_corrections',[]).append({
            'kind':kind,'ids':a['ids'],'raw_lines':[a['line'],b['line']],
            'rendered_lines':[first['line'],second['line']]})
    for batch in batches:
        batch['records']=[r for r in batch['records'] if id(r) not in removed]


def _paired_physical_raw_view(batches, raw_text):
    """Read one line-preserving raw projection shared with public normalization.

    Original source bytes remain untouched. Only exact commands already paired
    with removed public records can disappear from this physical reader view.
    """
    import hashlib
    import log_census as lc
    raw = lc.undo_redo_net(raw_text.splitlines())
    digest = hashlib.sha256(raw_text.encode('utf8', errors='surrogateescape')).hexdigest()
    removed = set()
    for batch in batches:
        for proof in batch.get('_rx_physical_raw_projection', []):
            lines = proof.get('raw_lines', [])
            originals = proof.get('raw_commands', [])
            if (proof.get('raw_sha256') != digest or len(lines) != 2 or len(originals) != 2
                    or len(set(lines)) != 2 or any(not isinstance(n, int) or not 1 <= n <= len(raw)
                                                for n in lines)
                    or any(raw[n - 1] != command for n, command in zip(lines, originals))):
                raise _Quarantine('coverage_dwarven_ring_unproven',
                                  {'detail': 'paired_raw_projection_mismatch'})
            first, second = [lc.MOVE_RE.search(command) for command in originals]
            ids = proof.get('ids')
            if (not first or not second or not isinstance(ids, list) or len(ids) != 1
                    or ids[0] not in lc.BASE_SH_DIE_PIDS | lc.KNOWN_BONUS_SH
                    or first[1].split() != ids or second[1].split() != ids
                    or (first[2], first[3], second[2], second[3]) !=
                       ('Game.1997', 'Game.2043', 'Game.2043', 'Game.1997')):
                raise _Quarantine('coverage_dwarven_ring_unproven',
                                  {'detail': 'paired_raw_projection_identity'})
            removed.update(lines)
    # A cancelled Ring attempt has independently paired Undo and exact-ID
    # face restoration proof. Merge those two blanked raw events into this
    # same view, retaining both raw/public source positions for diagnostics.
    cancelled = batches[0].get('_rx_cancelled_ring_rolls', []) if batches else []
    if cancelled:
        normalized_digest = hashlib.sha256('\n'.join(raw_text.splitlines()).encode(
            'utf8', errors='surrogateescape')).hexdigest()
        if batches[0].get('_rx_cancelled_ring_source_sha256') != normalized_digest:
            raise _Quarantine('coverage_dwarven_ring_unproven',
                              {'detail': 'cancelled_ring_source_mismatch'})
        for proof in cancelled:
            roll_line, restore_line = proof['raw_roll_line'], proof['raw_restore_line']
            if not (1 <= roll_line < restore_line <= len(raw)):
                raise _Quarantine('coverage_dwarven_ring_unproven',
                                  {'detail': 'cancelled_ring_projection_identity'})
            roll = lc.ROLL_RE.search(raw[roll_line - 1])
            change = lc.CHANGE_RE.search(raw[restore_line - 1])
            if (not roll or not change or roll[1].split() != [proof['die_id']]
                    or change[1] != proof['die_id'] or change[2] != proof['original_face']):
                raise _Quarantine('coverage_dwarven_ring_unproven',
                                  {'detail': 'cancelled_ring_projection_identity'})
            removed.update((roll_line, restore_line))
    for number in removed:
        raw[number - 1] = ''
    return lc.dedupe_echoes(raw)


def _normalize_reversed_hunt_parking(batches, rendered_path, raw_path=None):
    """Ignore a physically proved Eye recovery immediately reversed unchanged.

    This is not a new allocation/conversion: the same literal Eye returns to
    its prior box before any surviving action, random draw, or roll.
    """
    import log_census as lc
    from location_codes_authoritative import AUTH_LOCATION_CODES
    pairs=[]
    for batch in batches:
        records=batch['records']
        for first,second in zip(records,records[1:]):
            if (first.get('op')==second.get('op')=='move'
                    and _die_tokens(first.get('obj',''))==[('SHADOW','E')]
                    and first.get('obj')==second.get('obj')
                    and first.get('from')=='The Hunt For The Ring' and first.get('to')=='SA Dice'
                    and second.get('from')==first['to'] and second.get('to')==first['from']):
                pairs.append((batch,first,second))
    if not pairs:return
    if raw_path is None:raw_path=_raw_log_path(os.path.basename(str(rendered_path)).replace('.rendered.txt',''))
    if not raw_path or not os.path.isfile(raw_path):return
    import hashlib
    raw_text=read_raw_log_text(raw_path)
    raw_digest=hashlib.sha256(raw_text.encode('utf8', errors='surrogateescape')).hexdigest()
    raw=lc.undo_redo_net(raw_text.splitlines())
    meaningful=[];turn=1;rolls=0
    for number,text in enumerate(raw,1):
        marker=re.search(r'<~Game\.7~>(\d+)\b',text)
        if marker:turn=int(marker[1]);rolls=0
        if '<~Controls.1~>' in text:rolls+=1
        if not text.strip() or (re.fullmatch(r'<([^<>~]+)> .*',text)
                and not text.startswith(('<auto>','<game>')) and '<~' not in text):continue
        meaningful.append((number,turn,rolls,text))
    prefixes={};turn=None;rolls=0
    for batch in batches:
        if batch['turn']!=turn:turn=batch['turn'];rolls=0
        for record in batch['records']:
            if record.get('op')=='roll_num':rolls+=1
            prefixes[id(record)]=(turn,rolls)
    removed=set()
    for batch,first,second in pairs:
        matches=[]
        for a,b in zip(meaningful,meaningful[1:]):
            ma,mb=lc.MOVE_RE.search(a[3]),lc.MOVE_RE.search(b[3])
            if not ma or not mb:continue
            ids=ma[1].split()
            if (len(ids)!=1 or ids!=mb[1].split() or ids[0] not in lc.BASE_SH_DIE_PIDS|lc.KNOWN_BONUS_SH
                    or a[1:3]!=prefixes[id(first)] or b[1:3]!=prefixes[id(second)]):continue
            if any(record.get('player')!=event[3].split('<~',1)[0].lstrip('$')
                    or record.get('from')!=AUTH_LOCATION_CODES.get(move[2])
                    or record.get('to')!=AUTH_LOCATION_CODES.get(move[3])
                    for record,event,move in ((first,a,ma),(second,b,mb))):continue
            matches.append((a,b,ids))
        if len(matches)!=1:continue
        a,b,ids=matches[0];removed.update((id(first),id(second)))
        batch.setdefault('_rx_reversed_hunt_parking',[]).append({'ids':ids,'raw_lines':[a[0],b[0]],
            'rendered_lines':[first['line'],second['line']]})
        batch.setdefault('_rx_physical_raw_projection',[]).append({'ids':ids,'raw_lines':[a[0],b[0]],
            'rendered_lines':[first['line'],second['line']],
            'raw_commands':[a[3],b[3]],'raw_sha256':raw_digest})
    for batch in batches:batch['records']=[r for r in batch['records'] if id(r) not in removed]


def _normalize_repeated_ring_moves(batches, rendered_path, raw_path=None):
    """Collapse consecutive moves of one named Ring only with raw-ID proof."""
    import log_census as lc
    from location_codes_authoritative import AUTH_LOCATION_CODES
    records = [r for b in batches for r in b['records']]
    suspects = [(a, b) for a, b in zip(records, records[1:])
                if a.get('op') == b.get('op') == 'move'
                and str(a.get('obj', '')).lower() in _RING_NAMES
                and all(a.get(k) == b.get(k) for k in ('player', 'obj', 'from', 'to', 'body'))
                and int(a.get('line', 0)) + 1 == int(b.get('line', 0))]
    if not suspects:
        return
    def refuse(detail):
        raise _Quarantine('coverage_fellowship_die_unproven', {'reason': detail})
    if raw_path is None:
        raw_path = _raw_log_path(os.path.basename(str(rendered_path)).replace('.rendered.txt', ''))
    if not raw_path or not os.path.isfile(raw_path):
        refuse('ring_duplicate_raw_missing')
    raw = lc.undo_redo_net(read_raw_log_text(raw_path).splitlines())
    moves = []
    for line, text in enumerate(raw, 1):
        match = lc.MOVE_RE.search(text)
        if not match:
            continue
        ids, origin, target = match.group(1).split(), match.group(2), match.group(3)
        if len(ids) == 1 and ids[0] in lc.RING_PIECES:
            moves.append({'raw_line': line, 'id': ids[0], 'obj': lc.RING_PIECES[ids[0]],
                          'from': AUTH_LOCATION_CODES.get(origin), 'to': AUTH_LOCATION_CODES.get(target),
                          'player': text.split('<~', 1)[0].lstrip('$')})
    public = [r for r in records if r.get('op') == 'move'
              and str(r.get('obj', '')).lower() in _RING_NAMES]
    if len(public) != len(moves):
        refuse('ring_duplicate_transport_count')
    aligned = {}
    for record, move in zip(public, moves):
        if any(str(record.get(k, '')).lower() != str(move[k]).lower()
               for k in ('obj', 'from', 'to', 'player')):
            refuse('ring_duplicate_transport_identity')
        aligned[id(record)] = move
    remove = set()
    for first, second in suspects:
        a, b = aligned[id(first)], aligned[id(second)]
        if b['raw_line'] != a['raw_line'] + 1 or a['id'] != b['id']:
            refuse('ring_duplicate_not_contiguous_same_id')
        first['_rx_ring_duplicate_witness'] = {'first': a, 'second': b,
                                              'rendered_lines': [first['line'], second['line']]}
        remove.add(id(second))
    for batch in batches:
        batch['records'] = [r for r in batch['records'] if id(r) not in remove]


def _fold_cancelled_preroll_move(game, rendered_path, raw_path=None):
    """Discard no decision: prove a pre-roll drag and its exact reversal."""
    import log_census as lc
    from location_codes_authoritative import AUTH_LOCATION_CODES
    batches = game['batches']
    count = 0
    for index in range(len(batches) - 2, -1, -1):
        batch, following = batches[index:index + 2]
        if batch['turn'] != following['turn'] or batch.get('die_side') != 'FREE_PEOPLES':
            continue
        records = batch['records']
        at = next((i for i, r in enumerate(records) if r.get('line') == batch['start_line']), -1)
        if at < 1 or len(records[at - 1:]) != 5:
            continue
        advance, deposit, returned, reverse, roll = records[at - 1:]
        actor = deposit.get('player')
        if (advance.get('op') != 'move' or str(advance.get('obj', '')).lower() != 'the fellowship'
                or deposit.get('op') != 'move' or deposit.get('from') != 'FP Dice'
                or deposit.get('to') != 'The Hunt For The Ring'
                or len(_die_tokens(deposit.get('obj', ''))) != 1
                or not str(deposit.get('obj', '')).startswith('Free Action Die (')
                or returned.get('op') != 'move' or reverse.get('op') != 'move'
                or any(returned.get(k) != deposit.get(k) for k in ('obj', 'player'))
                or returned.get('from') != deposit.get('to') or returned.get('to') != deposit.get('from')
                or any(reverse.get(k) != advance.get(k) for k in ('obj', 'player'))
                or reverse.get('from') != advance.get('to') or reverse.get('to') != advance.get('from')
                or roll.get('op') != 'roll_dice' or any(r.get('player') != actor for r in (advance, reverse, roll))):
            continue
        if any(r.get('op') == 'roll_dice' and r.get('player') == actor
               for b in batches[:index + 1] if b['turn'] == batch['turn']
               for r in b['records'] if int(r.get('line', 0)) < int(deposit['line'])):
            continue
        if any(not r.get('line') or int(advance['line']) <= int(r['line']) <= int(reverse['line'])
               for r in batch.get('rng', [])):
            continue
        if raw_path is None:
            raw_path = _raw_log_path(os.path.basename(str(rendered_path)).replace('.rendered.txt', ''))
        if not raw_path or not os.path.isfile(raw_path):
            continue
        raw = lc.undo_redo_net(read_raw_log_text(raw_path).splitlines())
        # Chat has no board/RNG effect. Every other raw message remains a
        # boundary; an unrecognized transport cannot disappear from proof.
        meaningful = []; turn = 1
        for line, text in enumerate(raw, 1):
            marker = re.search(r'<~Game\.7~>(\d+)\b', text)
            if marker: turn = int(marker[1])
            if re.match(r'^<[^~>]+> ', text) and not text.startswith(('<game>', '<auto>')):
                continue
            meaningful.append((line, turn, text))
        matches = []
        for pos in range(len(meaningful) - 4):
            window = meaningful[pos:pos + 5]
            if any(t != batch['turn'] for _, t, _ in window): continue
            moves = [lc.MOVE_RE.search(text) for _, _, text in window[:4]]
            mr = lc.ROLL_RE.search(window[4][2])
            if not all(moves) or mr is None: continue
            if any(text.split('<~', 1)[0].lstrip('$') != actor for _, _, text in window): continue
            ids = [m[1].split() for m in moves]
            if (ids[0] != ['247'] or ids[2] != ids[1] or ids[3] != ids[0]
                    or len(ids[1]) != 1 or ids[1][0] not in lc.BASE_FP_DIE_PIDS | lc.KNOWN_BONUS_FP): continue
            if any(AUTH_LOCATION_CODES.get(m[2]) != r['from'] or AUTH_LOCATION_CODES.get(m[3]) != r['to']
                   for m, r in zip(moves, (advance, deposit, returned, reverse))): continue
            roll_ids = mr[1].split(); numbers = mr[2].split()
            try: faces = ['WCPMH'[int(n)] for n in numbers]
            except (ValueError, IndexError): continue
            if faces != roll.get('faces') or len(roll_ids) != len(set(roll_ids)) or ids[1][0] not in roll_ids: continue
            matches.append({'raw_lines': [line for line, _, _ in window], 'die_id': ids[1][0],
                            'rendered_lines': [r['line'] for r in (advance, deposit, returned, reverse, roll)]})
        if len(matches) != 1:
            continue
        witness = matches[0]
        deposit['_rx_cancelled_preroll_deposit'] = witness
        following['_rx_cancelled_preroll_move'] = witness
        following['records'] = records + following['records']
        following['rng'] = batch['rng'] + following['rng']
        del batches[index]
        count += 1
    return count


def _annotate_delayed_fellowship_ring(batches):
    """Recover a counter dragged after movement but before any Hunt outcome.

    Only the exact stage/advance/deposit/counter gesture sequence commutes.
    The Ring observation is the literal snapshot before the advance, never
    the board after moving or learning the Hunt result.
    """
    import copy
    for batch in batches:
        if not _is_paid_fellowship_move(batch) or batch.get('die_face') in ('C', 'W'):
            continue
        start = int(batch.get('start_line') or 0)
        tail = [r for r in batch['records'] if int(r.get('line') or 0) >= start]
        if len(tail) < 4:
            continue
        stage, advance, deposit, ring = tail[:4]
        face = batch.get('die_face')
        player = stage.get('player')
        if (stage.get('die_stage') != ('FREE_PEOPLES', face)
                or str(stage.get('from', '')).lower() != 'fp dice'
                or str(stage.get('to', '')).lower() != 'current action'
                or advance.get('op') != 'move'
                or str(advance.get('obj', '')).lower() not in ('fsp', 'the fellowship')
                or deposit.get('op') != 'move'
                or _die_tokens(deposit.get('obj', '')) != [('FREE_PEOPLES', face)]
                or str(deposit.get('from', '')).lower() != 'current action'
                or str(deposit.get('to', '')).lower() != 'the hunt for the ring'
                or _ring_rec_kind(ring) != 'fp_use'
                or any(r.get('player') != player for r in (advance, deposit, ring))
                or not _ring_use_is_settled(ring, tail)):
            continue
        # No hidden RNG list entry may have occurred before the payment.
        if any(not item.get('line') or int(item['line']) <= int(ring['line'])
               for item in batch.get('rng', [])
               if item.get('kind') not in ('action_roll', 'alloc_eye', 'add_eye')):
            continue
        prefix = advance.get('decision_truth')
        if not isinstance(prefix, dict) or not isinstance(advance.get('_rx_pools'), dict):
            continue
        expected_from, expected_to = (('mordor', int(batch['truth_before']['fel'].get('step', 0)))
                                     if batch['truth_before']['fel'].get('mordor') else
                                     ('fellowship track', int(batch['truth_before']['fel'].get('progress', 0))))
        origin = re.fullmatch(r'%s (\d+)(?: - .+)?' % expected_from, str(advance.get('from', '')).lower())
        target = re.fullmatch(r'%s (\d+)(?: - .+)?' % expected_from, str(advance.get('to', '')).lower())
        if not origin or not target or int(origin[1]) != expected_to or int(target[1]) != expected_to + 1:
            continue
        ring['_rx_fellowship_payment_witness'] = {
            'stage_line': start, 'advance_line': advance['line'],
            'deposit_line': deposit['line'], 'counter_line': ring['line'], 'face': face}
        ring['_rx_fellowship_payment_line'] = float(advance['line']) - 0.5
        ring['decision_truth'] = copy.deepcopy(prefix)
        ring['_rx_pools'] = copy.deepcopy(advance['_rx_pools'])
        ring['_rx_staged'] = {'FREE_PEOPLES': [face], 'SHADOW': []}


def _shadow_hunt_conversion(batch):
    """Certify literal extra Eyes from a legal Lidless Eye or Shadow Ring."""
    import ai
    import game_state as gs
    import random
    saved = random.getstate(), np.random.get_state()
    try:
        records = batch['records']
        deposits = [r for r in records if r.get('op') == 'move'
                    and not r.get('_rx_shadow_hunt_roundtrip')
                    and str(r.get('to', '')).lower() == 'the hunt for the ring'
                    and any(s == 'SHADOW' for s, _ in _die_tokens(r.get('obj', '')))]
        if not deposits:
            return None
        support_lines = []
        title = _card_play_title(batch)
        lidless = bool(title and str(title).split('/')[0].strip().lower() == 'the lidless eye')
        faces = []
        for record in deposits:
            cargo = str(record.get('obj', ''))
            tokens = _die_tokens(cargo)
            if (_DIE_TOKEN_RE.sub('', cargo).replace('group', '').strip(' (),')
                    or any(s != 'SHADOW' for s, f in tokens)):
                return None
            faces.extend(f for _, f in tokens)
        kind = None
        if lidless:
            from rendered_event_selections import lidless_evidence, FACE_NAMES
            originals, count, why = lidless_evidence(batch)
            if why or originals is None:
                return None
            letters = {name: letter for letter, name in FACE_NAMES.items()}
            faces = [letters[name] for name in originals]
            if batch.get('die_side') != 'SHADOW' or not 1 <= len(faces) <= 3:
                return None
            play_line = _card_play_titles(batch)[0][0]
            if any(r['line'] <= play_line or str(r.get('from', '')).lower() != 'sa dice'
                   for r in deposits):
                return None
            available = Counter(batch['_rx_pools']['SHADOW']['unused'])
            available[batch['die_face']] -= 1
            if any(n > available[f] for f, n in Counter(faces).items()):
                return None
            state = _recon_state(batch['truth_before'], batch['turn'])
            state.dice = _install_full_pools(batch['_rx_pools'])
            _install_ledger_hands(state, batch)
            _ensure_decks(state)
            _install_persistent_cards(state, batch['start_line'], killable=True)
            cid = _title_to_cid(title)
            legal = ai.get_legal_actions(state, gs.Side.SHADOW)
            play = next((a for a in legal if a.action_type == 'play_event'
                         and str(a.params.get('card_id')) == str(cid)
                         and state.dice[gs.Side.SHADOW][a.die_idx].current == _letter2face()[batch['die_face']]), None)
            if play is None:
                return None
            after = ai.apply_action(state, play)
            choices = ai.get_legal_actions(after, gs.Side.SHADOW)
            choice = next((a for a in choices if a.action_type == 'card_option'
                           and a.params.get('choices', {}).get('dice_count') == len(faces)
                           and sorted(a.params.get('choices', {}).get('dice_faces', []))
                           == sorted(_letter2face()[face].name for face in faces)), None)
            if choice is None:
                return None
            resolved = ai.apply_action(after, choice)
            if sum(d.current == gs.ActionDieFace.EYE for d in resolved.hunt_box) != len(faces):
                return None
            kind = 'lidless_eye'
        else:
            if title or len(faces) != 1 or len(deposits) != 1:
                return None
            deposit = deposits[0]
            if str(deposit.get('from', '')).lower() not in ('sa dice', 'current action'):
                return None
            rings = [r for r in records if _ring_rec_kind(r) == 'sh_use'
                     and int(r['line']) < int(deposit['line']) and _ring_use_is_settled(r, records)]
            if len(rings) != 1:
                return None
            ring = rings[0]
            between = [r for r in records if int(ring['line']) < int(r.get('line', 0)) < int(deposit['line'])]
            if faces == ['E']:
                if len(between) != 1:
                    return None
                change = between[0]
                parsed = _DIECHANGE_PARSE_RE.search(str(change.get('raw', '')))
                actor_prefix = str(ring.get('player', '')) + ' changes '
                if (change.get('op') != 'other' or parsed is None
                        or parsed.group(1).lower() != 'shadow'
                        or parsed.group(2).upper() == 'E' or parsed.group(3).upper() != 'E'
                        or not str(change.get('raw', '')).lstrip('%').startswith(actor_prefix)):
                    return None
                faces = [parsed.group(2).upper()]
                support_lines.append(int(change['line']))
            elif between:
                return None
            state = _ring_decision_state(batch, gs.Side.SHADOW, {'drop_reasons': Counter()}, ring)
            if state is None:
                return None
            legal = ai.get_legal_actions(state, gs.Side.SHADOW)
            action = next((a for a in legal if a.action_type == 'elven_ring'
                           and a.params.get('old_face') == _letter2face()[faces[0]].value
                           and a.params.get('new_face') == gs.ActionDieFace.EYE.value), None)
            if action is None:
                return None
            after = ai.apply_action(state, action)
            if len(after.hunt_box) != len(state.hunt_box) + 1:
                return None
            kind = 'elven_ring_eye'
        if kind == 'lidless_eye':
            support_lines += [int(record['line']) for record in records
                if record.get('op') == 'other'
                and (change := _DIECHANGE_PARSE_RE.search(str(record.get('raw', '')))) is not None
                and change.group(1).lower() == 'shadow' and change.group(3).upper() == 'E']
        refaced = Counter()
        for record in records:
            if int(record.get('line', -1)) in support_lines:
                change = _DIECHANGE_PARSE_RE.search(str(record.get('raw', '')))
                if change:
                    refaced[change.group(2).upper()] += 1
        converted_dice = []
        for face in faces:
            already_refaced = refaced[face] > 0
            converted_dice.append({'face': face, 'already_refaced': already_refaced})
            if already_refaced:
                refaced[face] -= 1
        return {'kind': kind, 'lines': [r['line'] for r in deposits] + support_lines, 'faces': faces,
                'conversion_faces_by_line': {int(deposits[-1]['line']): list(faces)},
                'conversion_dice_by_line': {int(deposits[-1]['line']): converted_dice},
                'ring_line': ring['line'] if kind == 'elven_ring_eye' else None}
    except (ValueError, KeyError, TypeError, RuntimeError, AttributeError):
        return None
    finally:
        random.setstate(saved[0])
        np.random.set_state(saved[1])


def _annotate_fellowship_dice(batches):
    """Public prefix witnesses for physical movement reconstruction.

    Kept callable so bounded real-batch fixtures use the same initialization
    as extract_one, rather than manufacturing proof flags.
    """
    global _FP_HUNTBOX_IDX, _FP_HUNTBOX_FACES, _FP_HUNTBOX_BATCHES
    _FP_HUNTBOX_IDX, _FP_HUNTBOX_FACES = [], []
    _FP_HUNTBOX_BATCHES = {}
    for _b9 in batches:
        witness = _paid_hunt_restage_return(_b9)
        _b9['_rx_paid_hunt_restage'] = witness
        for _r9 in _b9['records']:
            if _r9.get('_rx_cancelled_preroll_deposit'):
                continue
            if _r9.get('_rx_hunt_repeat_deposit'):
                continue
            if witness and _r9.get('line') == witness['return_line']:
                continue
            if _r9.get('op') == 'move' \
                    and 'Hunt For The Ring' in str(_r9.get('to', '')) \
                    and 'Free Action Die' in str(_r9.get('obj', '')):
                for _side9, _face9 in _die_tokens(_r9.get('obj', '')):
                    if _side9 == 'FREE_PEOPLES':
                        _FP_HUNTBOX_IDX.append((_r9.get('line', 0), _b9['turn']))
                        _FP_HUNTBOX_FACES.append((_r9.get('line', 0), _b9['turn'], _face9))
                        _FP_HUNTBOX_BATCHES[(_r9.get('line', 0), _b9['turn'])] = _b9
    uncertain_hunt_turns = set()
    staged_hunt_turns = set()
    conversions = {}
    die_players = {r.get('player'): r['die_stage'][0]
                   for b in batches for r in b['records'] if r.get('die_stage')}
    rolled_prefixes = {}
    for b in batches:
        for record in b['records']:
            if record.get('op') == 'roll_dice' and record.get('player') in die_players:
                key = (b['turn'], die_players[record['player']])
                rolled_prefixes.setdefault(key, []).append(
                    (int(record.get('line') or 0), record.get('_rx_roll_pool_faces', record.get('faces')) or []))
    for b in batches:
        prefix_rolls = {side: sorted((ln, faces) for ln, faces in
            rolled_prefixes.get((b['turn'], side), [])
            if ln < int(b.get('start_line') or 0))
            for side in ('FREE_PEOPLES', 'SHADOW')}
        b['_rx_fellowship_roll_known'] = all(prefix_rolls.values())
        b['_rx_shadow_rolled_eyes'] = (sum(str(face).upper() == 'E'
            for face in prefix_rolls['SHADOW'][-1][1]) if prefix_rolls['SHADOW'] else 0)
        b['_rx_hunt_dice_uncertain'] = b['turn'] in uncertain_hunt_turns
        b['_rx_shadow_conversions'] = list(conversions.get(b['turn'], []))
        proof = _shadow_hunt_conversion(b)
        b['_rx_shadow_hunt_effect'] = proof
        for record in b['records']:
            if record.get('_rx_shadow_hunt_roundtrip'):
                continue
            if record.get('die_stage') and any(
                    line < int(record.get('line') or 0)
                    for line, _ in rolled_prefixes.get((b['turn'], 'SHADOW'), ())):
                # Players may stage an Eye in Current Action before rolling
                # Shadow's pool, then put it in the Hunt Box as allocation.
                # That setup gesture is not a mid-turn change to Hunt dice.
                staged_hunt_turns.add(b['turn'])
            tokens = _die_tokens(record.get('obj', ''))
            touches_box = any('hunt for the ring' in str(record.get(key, '')).lower()
                              for key in ('from', 'to', 'zone'))
            changes_eye = (record.get('op') == 'other'
                           and 'shadow action die' in str(record.get('raw', '')).lower()
                           and ' to e' in str(record.get('raw', '')).lower())
            if (b['turn'] in staged_hunt_turns and
                    (changes_eye or (touches_box and any(side == 'SHADOW' for side, _ in tokens)))):
                if proof and record.get('line') in proof['lines']:
                    conversions.setdefault(b['turn'], []).extend(
                        dict(item, line=record['line'], kind=proof['kind'])
                        for item in proof.get('conversion_dice_by_line', {}).get(int(record['line']), ()))
                    if int(record.get('line') or 0) < int(b.get('start_line') or 0):
                        b['_rx_shadow_conversions'] = list(conversions[b['turn']])
                    continue
                uncertain_hunt_turns.add(b['turn'])
                if int(record.get('line') or 0) < int(b.get('start_line') or 0):
                    b['_rx_hunt_dice_uncertain'] = True


def _is_paid_fellowship_move(batch):
    """Recognize the action's literal attempt, including a Mordor Stop tile."""
    if batch.get('die_side') != 'FREE_PEOPLES' or _card_play_title(batch) is not None:
        return False
    before, after = batch['truth_before']['fel'], batch['truth']['fel']
    if not before.get('mordor'):
        return _fsp_advance(batch)
    if after.get('step', 0) > before.get('step', 0):
        return True
    step = int(before.get('step', 0))
    for record in batch['records']:
        if (record.get('op') != 'move'
                or str(record.get('obj', '')).strip().lower() != 'the fellowship'):
            continue
        origin = re.search(r'\bmordor\s+(\d+)\b', str(record.get('from', '')), re.I)
        target = re.search(r'\bmordor\s+(\d+)\b', str(record.get('to', '')), re.I)
        if origin and target and int(origin.group(1)) == step and int(target.group(1)) == step + 1:
            return True
    return False


def _fellowship_ring_prefix(batch):
    """Prove a non-Character staged die's actual Ring conversion to Character."""
    import ai
    import game_state as gs
    if not _is_paid_fellowship_move(batch) or _ring_counter_used(batch) != 'fp':
        return None
    history = list(batch.get('_rx_ring_carry_context') or batch.get('_rx_ring_carry') or []) + list(batch['records'])
    advances = [int(r.get('line') or 0) for r in batch['records'] if r.get('op') == 'move'
                and int(r.get('line') or 0) >= int(batch.get('start_line') or 0)
                and ('fellowship' in str(r.get('obj', '')).lower()
                     or str(r.get('obj', '')).strip().lower() == 'fsp')]
    if not advances:
        return None
    records = [r for r in history if _ring_rec_kind(r) == 'fp_use'
               and float(r.get('_rx_fellowship_payment_line', r.get('line') or 0)) < min(advances)
               and _ring_use_is_settled(r, history)]
    if len(records) != 1:
        return None
    if not isinstance((records[0].get('decision_truth') or {}).get('_rx_rings'), dict):
        return None
    stats = {'drop_reasons': Counter()}
    state = _ring_decision_state(batch, gs.Side.FREE_PEOPLES, stats, records[0])
    if state is None:
        return None
    original = _letter2face().get(batch.get('die_face'))
    legal = ai.get_legal_actions(state, gs.Side.FREE_PEOPLES)
    if not any(a.action_type == 'elven_ring'
               and a.params.get('old_face') == original.value
               and a.params.get('new_face') == gs.ActionDieFace.CHARACTER.value
               for a in legal):
        return None
    return records[0]['decision_truth']['_rx_rings']


def _prepare_fellowship_dice(st, batch, pending=False, advance=False):
    """Reconstruct a paid Fellowship move's physical dice from its public prefix.

    This does not use the client's early drag of the CURRENT die into the
    Hunt Box. Its addition remains owed until the public action settles.
    """
    import game_state as gs
    from fellowship_action_timing import queue
    tb, ta = batch['truth_before']['fel'], batch['truth']['fel']
    if not _is_paid_fellowship_move(batch):
        return False

    def refuse(reason):
        raise _Quarantine('coverage_fellowship_die_unproven', {
            'batch_start_line': batch.get('start_line'), 'reason': reason})

    face = _letter2face().get(batch.get('die_face'))
    ring_prefix = None
    if face not in (gs.ActionDieFace.CHARACTER, gs.ActionDieFace.WILL_OF_WEST):
        ring_prefix = _fellowship_ring_prefix(batch)
        if ring_prefix is None:
            refuse('physical_movement_face')
    pools = batch.get('_rx_pools')
    if not isinstance(pools, dict):
        refuse('physical_pool_missing')
    if (not batch.get('_rx_fellowship_roll_known')
            or '_rx_shadow_rolled_eyes' not in batch):
        refuse('physical_pool_roll_unproven')
    if batch.get('_rx_hunt_dice_uncertain'):
        refuse('hunt_box_change_not_reconstructed')
    dice = _install_full_pools(pools)
    fp, sh = gs.Side.FREE_PEOPLES, gs.Side.SHADOW
    selected = next((d for d in dice[fp] if not d.used and d.current is face), None)
    if selected is None:
        refuse('staged_die_missing_from_unused_pool')
    dice[fp].remove(selected)
    dice[fp].insert(0, selected)
    box = []
    turn, line = batch['turn'], int(batch.get('start_line') or 0)
    for ln, tn, letter in _FP_HUNTBOX_FACES:
        if tn != turn or ln >= line:
            continue
        prior_face = _letter2face().get(letter)
        prior = next((d for d in dice[fp] if d.used and d.current is prior_face
                      and all(d is not b for b in box)), None)
        if prior is None:
            refuse('earlier_hunt_die_missing_from_used_pool')
        if prior_face not in (gs.ActionDieFace.CHARACTER, gs.ActionDieFace.WILL_OF_WEST):
            source = _FP_HUNTBOX_BATCHES.get((ln, tn))
            if source is None or _fellowship_ring_prefix(source) is None:
                refuse('earlier_changed_hunt_die_unproven')
            prior.current = gs.ActionDieFace.CHARACTER
        box.append(prior)
    if _RX_EYEALLOC is None or _RX_EYEROLL is None:
        refuse('shadow_hunt_allocation_missing')
    shadow_count = int(_RX_EYEALLOC.get(turn, 0)) + int(batch['_rx_shadow_rolled_eyes'])
    shadow = [d for d in dice[sh] if d.current is gs.ActionDieFace.EYE]
    if len(shadow) > shadow_count:
        refuse('shadow_hunt_pool_count_conflict')
    for _ in range(shadow_count - len(shadow)):
        d = gs.ActionDie(gs.SH_FACES, sh)
        d.current = gs.ActionDieFace.EYE
        dice[sh].append(d)
        shadow.append(d)
    for d in shadow:
        d.used = True
    for conversion in batch.get('_rx_shadow_conversions', []):
        if conversion.get('already_refaced'):
            # The public pool walk removes an explicitly refaced Eye from
            # the old-face used list. Its proved deposit is separate from
            # this turn's allocated/rolled Eyes and remains an owned die.
            converted = gs.ActionDie(gs.SH_FACES, sh)
            converted.current = gs.ActionDieFace.EYE
            converted.used = True
            dice[sh].append(converted)
            shadow.append(converted)
            continue
        old_face = _letter2face().get(conversion['face'])
        converted = next((d for d in dice[sh] if d.used and d.current is old_face
                          and all(d is not item for item in shadow)), None)
        if converted is None:
            refuse('shadow_converted_die_missing_from_used_pool')
        converted.current = gs.ActionDieFace.EYE
        shadow.append(converted)
    box.extend(shadow)
    st.dice, st.hunt_box = dice, box
    st.eyes_allocated = int(_RX_EYEALLOC.get(turn, 0))
    st._rx_eyes_stamped = True
    # These are already the full physical pools. An encoding-only rebuild
    # from the pre-move annotation would make the consumed index unused.
    st._rx_full_pools = None
    if ring_prefix is not None:
        import game_logic as gl
        _stamp_rings(st, ring_prefix)
        # Locate by the original physical face, then perform the same public
        # conversion. Never substitute another already-Character die.
        gl.use_elven_ring(st, fp, 0, gs.ActionDieFace.CHARACTER)
    if pending:
        selected.used = True
        queue(st, 0)
    if advance:
        if tb.get('mordor'):
            st.fellowship.mordor_step += 1
        else:
            st.fellowship.progress += 1
    return True


def _seed_fp_huntbox(st, batch):
    """Seed the recon state's Hunt Box with the FP dice placed by the
    turn's EARLIER fellowship moves (each = +1 to the hunt roll)."""
    import game_state as gs
    from game_state import Side
    prior = sum(1 for (ln, tn) in _FP_HUNTBOX_IDX
                if tn == batch['turn']
                and ln < (batch.get('start_line') or 0))
    for _ in range(prior):
        d = gs.ActionDie(gs.FP_FACES, Side.FREE_PEOPLES)
        d.current = gs.ActionDieFace.CHARACTER
        d.used = True
        st.hunt_box.append(d)
    return prior


def _ring_rec_kind(r):
    """Classify ONE record as an elven-ring counter gesture: ('fp_use' |
    'fp_back' | 'sh_use' | 'sh_back' | None).  W3-016 (2026-09-02):
    three Shadow dialects exist in the 24 cover games - the classic
    'Dark Ring N -> SA Casualties' drop, a 'Dark Ring N -> Current FP
    Card / Current SA Card / Board6' staging (swept by a kill later)
    and a direct 'kills Narya in Dark Ring N'.  A move BACK into a
    Dark Ring slot from a non-ring slot is the Shadow take-back (the
    SlickRick T7 undo); Dark Ring -> Elven Ring is the FP take-back."""
    op = r.get('op')
    if op == 'kill':
        zone = str(r.get('zone', '')).lower()
        names = [str(x).strip().lower() for x in (r.get('pieces') or [])]
        if any(n in _RING_NAMES for n in names) and 'dark ring' in zone:
            return 'sh_use'
        return None
    if op != 'move':
        return None
    if str(r.get('obj', '')).strip().lower() not in _RING_NAMES:
        return None
    frm = str(r.get('from', '')).lower()
    to = str(r.get('to', '')).lower()
    f_er, f_dr = 'elven ring' in frm, 'dark ring' in frm
    t_er, t_dr = 'elven ring' in to, 'dark ring' in to
    if f_er and not t_er:
        return 'fp_use'
    if f_dr and t_er:
        return 'fp_back'
    if f_dr and not t_dr:
        return 'sh_use'
    if t_dr and not f_dr and not f_er:
        return 'sh_back'
    return None


def _ring_counter_used(batch):
    """The elven-ring COUNTER use inside the batch: 'fp' / 'sh' / None.
    Every dialect of _ring_rec_kind counts; a use followed IN THE SAME
    BATCH by its take-back is no use.  When both sides' gestures sit in
    one batch (a glued render) the batch's own die side wins.  W3-005:
    a PRE-STAGING use carried in from the previous batch's tail
    (_annotate_ring_carry) counts as the batch's own."""
    used = {'fp': False, 'sh': False}
    for r in list(batch.get('_rx_ring_carry') or []) + batch['records']:
        k = _ring_rec_kind(r)
        if k == 'fp_use':
            used['fp'] = True
        elif k == 'fp_back':
            used['fp'] = False
        elif k == 'sh_use':
            used['sh'] = True
        elif k == 'sh_back':
            used['sh'] = False
    own = 'sh' if batch.get('die_side') == 'SHADOW' else 'fp'
    if used[own]:
        return own
    other = 'fp' if own == 'sh' else 'sh'
    return other if used[other] else None


def _ring_implied_face(batch):
    """The face the ring must have produced, inferred from the batch's
    primary gesture class (used only when the client wrote NO 'changes'
    line - the action itself is the evidence)."""
    if _has_fsp(batch):
        return 'C'
    if any(it.get('kind') == 'draw_deck' for it in batch['rng']):
        return 'P'
    if [1 for (_f, _t, _u) in _army_move_records(batch) if _u]:
        return 'A'
    if _reinf_recruits(batch):
        return 'M'
    return None


def _ring_is_tablekill_payment(batch):
    """W3-005: the FP arm of 1.149 pays the removal with a Will die OR
    any die plus one Elven Ring (ai.discard_table_card payment='ring'):
    the counter leaves the Elven Ring slot and NO die changes face
    (2023-03-24 L448-451 C + Narya, 2025-05-20 L590-593 P + Narya).
    True for an FP batch whose own ring use sits beside a die-paid
    table-card kill - the ring is the discard_table_card row's payment
    provenance, never an elven_ring decision (and never a re-face).

    The PROOF of payment is the kill group: the FP die and the removed
    card die TOGETHER in Current Action ('kills group (Free Action Die
    (C), The Palantir of Orthanc/Cruel as Death) in Current Action').
    A table-card move alone is not enough: 2025-05-20 L1142-1146 is a
    ring-made Character die walking the Fellowship (die -> Hunt Box)
    with the SHADOW seat's own Crebain slot -> Current SA Card drag
    glued into the window - a fellowship move, not a payment."""
    if batch.get('die_side') != 'FREE_PEOPLES':
        return False
    if _ring_counter_used(batch) != 'fp':
        return False
    try:
        title = _table_kill_title(batch)
        if title is None or not _table_kill_die_spend(batch):
            return False
        for r in batch['records']:
            if r.get('op') != 'kill':
                continue
            pieces = [str(x) for x in (r.get('pieces') or ())]
            if (any('Free Action Die' in x for x in pieces)
                    and any(title in x for x in pieces)):
                return True
        return False
    except Exception:
        return False


# W3-016 (2026-09-02): a SHADOW ring use WITHOUT a 'changes' line names
# no face.  The gesture-class inference above is a guess (a turn-end or
# Witch-King draw glued into the batch read as 'P', an attack as 'A'
# though a Character die with a leader or a Muster/Army die attacks
# too).  The face is PROVEN by the engine instead: the batch's primary
# gesture is re-matched under every ring-reachable face and a face
# ENABLES it when the LEAD die (index 0, the die the ring converted)
# itself binds the gesture in the engine menu (recovered die=-1 rows
# and matches on another pool die never count).  Exactly one enabler
# differing from the staged face = the elven_ring row's new_face;
# several = ring_face_ambiguous (drop beats wrong; the mirror still
# rides the FIRST enabler so the follow-up rows bind the converted die
# instead of dropping); none = ring_face_none; the staged face itself
# enabling = ring_face_staged.  Probe order = the natural face per
# gesture class (card play P, army A, muster M, flights C, then H).
_RING_FACE_ORDER = ('P', 'A', 'M', 'C', 'H')
_RING_PROBE = None      # (letter, hits) while a face probe is running


def _ring_face_implied_by_consequence(batch, faces):
    """1.268: the ONE face a rendered CONSEQUENCE proves, or None.
    Rule G (engine ai._grant_event_play_draw: Gandalf the Grey as GUIDE
    draws one card of the SAME deck when the Free Peoples play an event
    card with an Event / Will die - never on a Character, Army or Muster
    face): an FP batch whose ring-converted die paid a card play, with
    Gandalf the Grey guiding, that then draws a card of the played card's
    deck inside the same die spend, used the EVENT face ('P').  The
    played card's own printed text must not itself draw a card (the
    USER's "unrelated to the card effects").  Absence of the draw proves
    nothing (drop beats wrong): the faces stay a set."""
    try:
        if 'P' not in faces or batch.get('die_side') != 'FREE_PEOPLES':
            return None
        _fel = (batch.get('truth_before') or {}).get('fel') or {}
        if str(_fel.get('guide') or '') != 'Gandalf the Grey':
            return None
        title = _card_play_title(batch)
        if title is None:
            return None
        cid = _title_to_cid(title)
        if not cid:
            return None
        from cards import get_card as _gc9
        card = _gc9(str(cid))
        if card is None:
            return None
        deck = str(getattr(card, 'deck', '') or '').lower()
        if deck not in ('character', 'strategy'):
            return None
        if 'draw' in str(getattr(card, 'event_effect', '') or '').lower():
            return None            # the card's own text draws - no proof
        if deck in [str(d).lower() for d in _own_die_draws(batch)]:
            return 'P'
    except Exception:
        return None
    return None


def _ring_face_analysis(batch):
    """{'staged','faces','hits','unique','mirror','reason','exc'} for an
    implied-ring batch of either side (cached on batch['_rx_ringfaces']).
    W3-005 (2026-09-02): the Free Peoples ride the same probe (the old
    gesture-class inference - fsp -> C, deck draw -> P, army legs -> A,
    recruits -> M - was a guess: a turn-end draw read as P, army legs
    are A or H, a bare Mordor hide inferred nothing and 13 of 55 FP
    ring uses had no row).  One FP rule short-circuits the probe: a
    hunt-kind batch spent its die on the pawn advance itself, and a
    pawn advance is a Character (or Will) die by rule while the ring
    never produces Will - C is the only face."""
    global _RING_PROBE, _MM_FACE, _MM_SIDE, _MM_STATE
    cached = batch.get('_rx_ringfaces')
    if cached is not None:
        return cached
    import ai
    import rendered_census as rc
    import rendered_replay as rr
    from collections import Counter as _C
    staged = str(batch.get('die_face', '')).upper()
    kind = batch.get('kind')
    n_unit_moves = len([1 for (_f, _t, _u) in _army_move_records(batch)
                        if _u])
    res = {'staged': staged, 'faces': [], 'hits': {}, 'unique': None,
           'mirror': None, 'reason': 'none', 'exc': []}
    if kind == 'hunt' and batch.get('die_side') == 'FREE_PEOPLES':
        _tb5 = (batch.get('truth_before') or {}).get('fel') or {}
        _ta5 = (batch.get('truth') or {}).get('fel') or {}
        if _has_fsp(batch) or (_tb5.get('mordor') and
                               _ta5.get('step', 0) > _tb5.get('step', 0)):
            res['faces'] = ['C']
            res['hits']['C'] = ('fellowship_move', 'rule:character_die')
            kind = 'hunt:rule'      # the probe loop below is skipped
    # W3-005: a draw_card hit is never the enabler while the batch holds
    # any other gesture (a glued draw must not read a battle as P), but
    # a batch whose ONLY gesture is one own deck draw while the die is
    # staged (2023-09-11 L858-861: M die + Nenya, draws Free Character
    # Card, die killed) is a ring-made Event die by rule - the Event
    # face is the only ring-reachable face that draws (Will is not).
    _sole_draw = False
    try:
        _sole_draw = (
            kind not in ('combat', 'combat?', 'hunt', 'hunt:rule')
            and n_unit_moves == 0 and not _has_fsp(batch)
            and not _reinf_recruits(batch)
            and _card_play_title(batch) is None
            and _table_kill_title(batch) is None
            and len(_own_die_draws(batch)) == 1
            and not any(
                r.get('op') in ('move', 'flip') and (
                    rr.zone_kind(str(r.get('to', '')))[0] == 'political'
                    or rr.zone_kind(str(r.get('from', '')))[0]
                    == 'political')
                for r in batch['records']))
    except Exception:
        _sole_draw = False
    draw_faces = []
    saved = (_MM_FACE, _MM_SIDE, _MM_STATE)
    # The probe must leave NO trace: the per-game random.seed(0) stream
    # is the hunt/combat drivers' fallback on roll exhaustion (a re-roll
    # the log carries as a SECOND roll line draws from it), so a probe
    # that consumed it shifted every later fallback draw (2023-11-26 Mol
    # T12 L998: the re-roll fallback flipped from a miss to a hit and a
    # phantom hunt_gollum row appeared).  Save/restore both generators.
    import random as _rnd9p
    _rng9 = (_rnd9p.getstate(), np.random.get_state())
    try:
        for L in _RING_FACE_ORDER:
            # RB19: FP Army appears only on the combined Muster/Army face.
            # Probing a synthetic pure Army result invents a positive label
            # that the physical die and current AI menu cannot represent.
            if L == 'A' and batch.get('die_side') == 'FREE_PEOPLES':
                continue
            if kind in ('hunt', 'hunt:rule'):
                break   # the hunt walk is not a probe target
            if L == 'C' and n_unit_moves >= 2:
                continue        # a Character die moves ONE (led) army
            b2 = dict(batch)
            b2['die_face'] = L
            hits = []
            pst = {'drop_reasons': _C(), 'kinds_emit': _C(),
                   'detected': _C(), 'nomatch': 0, 'drove': 0,
                   'diverged': 0, 'ambiguous': 0, 'tablekill_paid': _C(),
                   # a COPY: the probe must never mark a card as played
                   'played_cids': set((_RX_STATS or {}).get('played_cids')
                                      or ())}
            _RING_PROBE = (L, hits)
            try:
                if kind in ('combat', 'combat?'):
                    st, sd = _mk_state_with_die(b2, install_hands=True)
                    if st is None:
                        continue
                    _ch0 = [r for r in b2['rng']
                            if r.get('kind') == 'roll_num']
                    _ed0 = len(_ch0[0]['ints']) if _ch0 else None
                    plan = rc._plan_combat(st, b2, expected_dice=_ed0,
                                           chats=_batch_chats(b2))
                    if not plan:
                        continue
                    atk, dfn, _sg = plan
                    legal = ai.get_legal_actions(st, sd)

                    def _to9(a):
                        return rr.sanitize(str(a.params.get(
                            'to', a.params.get('to_region', ''))))
                    ok = [a for a in legal
                          if a.action_type == 'army_attack'
                          and a.die_idx == 0 and _to9(a) == dfn
                          and rr.sanitize(str(a.params.get('from', '')))
                          == atk]
                    if not ok:
                        ok = [a for a in legal
                              if a.action_type == 'army_attack'
                              and a.die_idx == 0 and _to9(a) == dfn]
                    if ok:
                        hits.append(('army_attack', 'army_attack', 0))
                else:
                    if _emit_chat_battle(b2, pst) is None:
                        # W3-037 (2026-09-03): an attack WITHOUT a battle
                        # (the defender withdrew into its stronghold
                        # before the move-in, no roll - 2024-05-21
                        # SlickRick L338-343: Muster die + Nenya, Lamedon
                        # -> Dol Amroth) is anchored on the withdrawal
                        # gesture, which only _emit_siege_withdrawal
                        # matches (army_attack die 0); the bookkeeping
                        # matcher knows army_move only and read every
                        # face as 'none'.  Probe the withdrawal shape
                        # first - it self-gates on the gesture and on
                        # the probe face (P never attacks).
                        _emit_siege_withdrawal(b2, pst)
                        st, sd = _mk_state_with_die(b2, install_hands=True)
                        if st is None:
                            continue
                        legal = ai.get_legal_actions(st, sd)
                        _emit_bookkeeping(b2, st, sd, legal, pst)
            except Exception as _e9:
                res['exc'].append('%s:%s' % (L, str(_e9)[:24]))
                continue
            finally:
                _RING_PROBE = None
            first = [h for h in hits if h[0] != 'draw_card']
            if not first:
                if _sole_draw and any(
                        h[0] == 'draw_card' and h[2] == 0 for h in hits):
                    draw_faces.append(L)
                continue
            _ph, _at, _di = first[0]
            if _di != 0:
                continue
            if any('leg2' in str(k) for k in pst['drop_reasons']):
                continue
            res['faces'].append(L)
            res['hits'][L] = (_ph, _at)
    finally:
        _RING_PROBE = None
        _MM_FACE, _MM_SIDE, _MM_STATE = saved
        _rnd9p.setstate(_rng9[0])
        np.random.set_state(_rng9[1])
    if not res['faces'] and draw_faces:
        res['faces'] = list(draw_faces)
        for L in draw_faces:
            res['hits'][L] = ('draw_card', 'draw_card:sole_gesture')
    # 1.268 (USER 2026-09-03): before declaring the enabling faces
    # equivalent, IMPLY the face from a consequence the log renders
    # (never from the extractor's own guesses): today's one proven rule
    # is the Free Peoples' Gandalf the Grey guide redraw.
    if len(res['faces']) > 1:
        _imp9 = _ring_face_implied_by_consequence(batch, res['faces'])
        if _imp9 in res['faces']:
            _old9 = res['hits'].get(_imp9)
            res['faces'] = [_imp9]
            res['hits'][_imp9] = ((_old9[0] if _old9 else '?'),
                                  'implied:gandalf_grey_same_deck_draw')
            res['implied'] = _imp9
    faces = res['faces']
    if not faces:
        res['reason'] = 'none'
    elif staged in faces:
        res['reason'] = 'staged'
    elif len(faces) == 1:
        res['reason'] = 'unique'
        res['unique'] = faces[0]
        res['mirror'] = faces[0]
    else:
        res['reason'] = 'ambiguous'
        res['mirror'] = faces[0]
    batch['_rx_ringfaces'] = res
    return res


_SARUMAN_SETTLEMENTS = ('orthanc', 'north dunland', 'south dunland')


def _isengard_regulars_removed_at_orthanc(batch):
    """Isengard Regulars the batch takes OFF Orthanc back to the pool - the
    swap half of Saruman's Muster-die ability (USER ruling 1.266): rendered
    either as `kills group ([2]Isengard Regular) in Orthanc` (the usual
    VASSAL gesture) or as `moves group ([2]Isengard Regular) from Orthanc
    to SA Reinforcements` (2023-01-05 L410 / 2023-11-26 L487 dialect).
    Returns the unit count (0 when the batch removes none)."""
    import rendered_replay as rr
    n = 0
    for r in batch.get('records', ()):
        op = r.get('op')
        if op == 'kill':
            if rr.sanitize(str(r.get('zone', ''))) != 'orthanc':
                continue
            toks = list(r.get('pieces') or ())
        elif op == 'move':
            if rr.sanitize(str(r.get('from', ''))) != 'orthanc' or \
                    rr.zone_kind(str(r.get('to', '')))[0] != 'reinf':
                continue
            toks = _group_tokens(str(r.get('obj', '')))
        else:
            continue
        for tok in toks:
            p = rr.parse_piece(tok)
            if p and p[0] == 'unit' and p[1] == 'ISENGARD' \
                    and p[2] == 'REGULAR':
                n += int(p[3] or 1)
    return n


def _saruman_ability_shape(batch, recruits, eff_letter, st):
    """W4-036 (USER ruling 1.266 / W4-Q10; engine half W4-032 HELD with
    GEN43): with a Muster die while Saruman is in play the Shadow player
    may EITHER place one Isengard Regular in each free Isengard settlement
    (Orthanc, North Dunland, South Dunland - all three, all-or-nothing) OR
    replace two Isengard Regulars in Orthanc with two Isengard Elites.
    Returns 'R3' / 'E2' when the batch's FINAL placements are exactly that
    shape on a plain (no card play) Shadow Muster / Muster-Army die - or a
    die the Shadow player's own Elven Ring re-faced for a muster (2023-09-07
    L645 C + Narya: _ring_implied_face 'M') - with Saruman in play on the
    reconstructed state, else None.  Mixed shapes
    are never the ability: E1R2 (2023-01-20 L702) is a human illegal move
    and A New Power is Rising (R6 / R4E2 / R5E1 across North + South
    Dunland x2, always a card-play batch) is the W1 card family.  Meter
    only - the caller emits no row either way."""
    import rendered_replay as rr
    if batch.get('die_side') != 'SHADOW':
        return None
    if eff_letter not in ('M', 'H') and not (
            _ring_counter_used(batch) == 'sh'
            and _ring_implied_face(batch) == 'M'):
        return None
    if _card_play_title(batch) is not None or st is None:
        return None
    try:
        import game_logic as _gl
        if not _gl.saruman_in_play(st):
            return None
    except Exception:
        return None
    placed = {}
    for (nat, reg, kind, cnt) in recruits:
        if str(nat).upper() != 'ISENGARD':
            return None
        key = (rr.sanitize(str(reg)), str(kind).upper())
        placed[key] = placed.get(key, 0) + int(cnt)
    if placed == dict(((s, 'REGULAR'), 1) for s in _SARUMAN_SETTLEMENTS):
        return 'R3'
    if placed == {('orthanc', 'ELITE'): 2} and \
            _isengard_regulars_removed_at_orthanc(batch) == 2:
        return 'E2'
    return None


def _emit_bookkeeping(batch, st, side, legal, stats):
    """Match the human's PRIMARY die-usage in a bookkeeping batch against the
    (board-only, faithfully reconstructed) legal set and emit ONE BC sample.
    Matchers run most-specific first; the die face is not gated on â€” the legal
    set already reflects it (a face that never generates army_move simply has
    no army_move to match, and we fall through).  Returns a sample or None."""
    global _MM_FACE, _RX_ALIAS_V41
    import rendered_replay as rr
    from game_state import Side
    turn = batch['turn']

    # 0-pre) RETRACTED SPEND (GAME02 T3): the staged die goes BACK to
    # the dice pool (not killed) and the region moves cancel out - the
    # spend never happened, nothing may emit.  Guarded on zero net
    # board effect so quirk batches that still act (GAME02 T2's
    # political-piece spend also returns its die) keep emitting.
    _die_ret9 = any(
        r.get('op') == 'move' and 'Action Die' in str(r.get('obj', ''))
        and 'current action' in str(r.get('from', '')).lower()
        and str(r.get('to', '')).strip().lower().endswith('dice')
        for r in batch['records'])
    if _die_ret9:
        _net9 = []
        for r in batch['records']:
            if r.get('op') != 'move':
                continue
            fk9 = rr.zone_kind(r.get('from', ''))
            tk9 = rr.zone_kind(r.get('to', ''))
            if fk9[0] not in ('region', 'stronghold') or \
                    tk9[0] not in ('region', 'stronghold'):
                continue
            ob9 = str(r.get('obj', ''))
            if _net9 and _net9[-1] == (tk9[1], fk9[1], ob9):
                _net9.pop()
            else:
                _net9.append((fk9[1], tk9[1], ob9))
        _pol9 = any(r.get('op') in ('move', 'flip') and (
            rr.zone_kind(str(r.get('to', '')))[0] == 'political'
            or rr.zone_kind(str(r.get('from', '')))[0] == 'political')
            for r in batch['records'])
        _rf9 = any(r.get('op') == 'move' and
                   rr.zone_kind(str(r.get('from', '')))[0] == 'reinf'
                   for r in batch['records'])
        if not _net9 and not _pol9 and not _rf9 and not _has_fsp(batch):
            stats['drop_reasons']['spend_retracted'] += 1
            return None

    # The staged face and the encoded full pool must follow the same change.
    _eff_letter = _effective_lead_letter(batch)
    if _eff_letter != batch['die_face']:
        _f9 = _letter2face().get(_eff_letter)
        if _f9 is not None and st.dice.get(side):
            # convert only the LEAD (staged) die â€” rewriting the whole
            # pool (incl. used dice) misstated the marginal menu
            _set_effective_lead_face(st, side, _f9)
            try:
                import ai as _ai9r
                legal = _ai9r.get_legal_actions(st, side)
                # the cross-face disambiguation key follows the
                # converted lead (W4-018: with the rebuilt pools a
                # second die of the card's own type is now present)
                _MM_FACE = _f9
            except Exception:
                _eff_letter = batch['die_face']

    # 0-tk) SHADOW-PAID TABLE KILL WITH SHOWN PAYMENT CARDS (W1-010) ----
    # The Shadow arm of 1.149 (APTG / Tom Bombadil text: "using any one
    # Action Die and discarding one Army [Event] card and one Character
    # card from his hand") is rendered by many humans as: die staged,
    # the table card dragged BoardN -> Current card zone, and the two
    # payment cards SHOWN (hand -> Current card zone, i.e. the same move
    # a play makes) instead of - or as well as - killed hidden in hand;
    # then die + cards killed as one group (2023-03-16 L189-197,
    # 2023-11-26 L1426-1430, 2024-05-21 L1192-1198, 2025-05-20 L784-788).
    # Step 0 claimed those batches on the shown card's hand exit and the
    # payment trained as a play_event (three of them with die=-1, i.e. as
    # a FREE play) while the kill and its payment emitted nothing.  The
    # one die cannot buy both, and nothing but the paid removal moves the
    # opponent's APTG / Tom Bombadil off the board into the Shadow's card
    # zone, so the hand exits of such a batch are payments: run the
    # table-kill emitter FIRST and never fall through to a play.
    tk_title = _table_kill_title(batch)
    if (tk_title is not None and side == Side.SHADOW
            and _title_to_ptype(tk_title) in _TABLEKILL_SHADOW_PAID
            and _table_kill_die_spend(batch)
            and _card_play_titles(batch)):
        _tk0 = _emit_table_kill(batch, st, side, legal, tk_title, stats)
        if _tk0 is not None:
            return _tk0
        # the removal itself was dropped (its own counter says why): a
        # payment card is still not a play - drop, never a wrong label
        stats['drop_reasons']['tablekill_shown_payment_unemitted'] += 1
        return None

    # 0) EVENT-CARD PLAY (highest priority) ------------------------------
    # A hand-exit to a play area means the die BOUGHT the card; label it
    # play_event and NEVER fall through (a card's board EFFECTS must not be
    # mislabelled as an army move / muster / political advance).
    play_title = _card_play_title(batch)
    if play_title is not None:
        if _commit_in_missed_battle(batch):
            # W1-001: a combat-card COMMIT in a battle the classifier
            # missed is never an event play - drop (the battle rows are
            # the combat family's), never a wrong play_event
            stats['drop_reasons']['play_is_combat_commit'] += 1
            return None
        return _emit_play_event(batch, st, side, legal, play_title, stats)

    # 0.1) 1.149 TABLE-CARD KILL (USER 2026-08-20) -----------------------
    # Checked AFTER the hand-exit play (a card entering a slot is a
    # PLAY) and BEFORE everything else, because the payment discards
    # would otherwise be read as a hand-limit discard and the die as a
    # plain spend.  Detection-only until now; see _emit_table_kill.
    tk_title = _table_kill_title(batch)
    if tk_title is not None:
        _tk = _emit_table_kill(batch, st, side, legal, tk_title, stats)
        if _tk is not None:
            return _tk

    # 0.5) DRAW CARD (Event die spent to DRAW â€” deck choice) -------------
    # A P (Event) die that drew exactly one deck card (and played nothing â€”
    # the play_event branch above already returned) is the draw-vs-play +
    # which-deck decision.  Multiple draws in one batch = a card effect, not a
    # single die's draw -> skip (never conflate).
    if _LIST2 and _eff_letter in ('P', 'W'):
        # W (wildcard) spent as an Event die is the same draw decision
        # (truth row T9-7)
        # W1-006: only the acting side's OWN deck draws made while the
        # die is still staged count (see _own_die_draws) - the batch
        # window runs to the next staging, so the turn's LAST batch also
        # carries the end-of-turn dice recovery and both sides' phase-1
        # draws, which used to push the count past 1 and drop the row.
        draws = _own_die_draws(batch)
        if len(draws) == 1:
            _deck = draws[0]
            _stag = 'fp' if side == Side.FREE_PEOPLES else 'sh'
            hidx = _match_index(
                legal, lambda a: a.action_type == 'draw_card' and
                a.params.get('deck') == _deck and
                a.params.get('side') == _stag)
            if hidx is not None:
                stats['kinds_emit']['draw_card'] += 1
                return _emit_sample(st, side, legal, hidx, 'draw_card', turn)
            stats['drop_reasons']['draw_card_unmatched'] += 1
            return None

    # 1) ARMY MOVE (exact composition among the split grid) --------------
    # W4-001 / W4-002: the batch's drags are NETTED first (partial
    # return -> the net count; full reversal -> no flow at all), so the
    # human's real second move becomes leg 2 instead of the return drag.
    # W4-003: with ONE die spent a drag A->B->C of the figures that just
    # arrived at B is the single move A->C (A adjacent to C) - the
    # intermediate drop is a VASSAL gesture, not a second army move.
    try:
        _through = _batch_dice_spent(
            batch, (batch.get('die_face'), _eff_letter)) == 1
    except Exception:
        _through = False
    _flows = _net_batch_flows(batch, keep_empty=True, through=_through)
    if any(fl.get('via') for fl in _flows):
        stats['detected']['move_through_netted'] += 1
    if batch.get('_rx_through_ambiguous'):
        stats['detected']['move_through_ambiguous'] += 1
    if any(fl['returned'] for fl in _flows):
        stats['detected']['move_return_netted'] += 1
    if any(not fl['keys'] for fl in _flows):
        stats['detected']['move_reversal_netted'] += 1
    _flows = [fl for fl in _flows if fl['keys']]
    _flow_comp = dict(((fl['from'], fl['to']), fl['comp'])
                      for fl in _flows)
    unit_moves = [(fl['from'], fl['to']) for fl in _flows
                  if fl['has_unit']]
    if unit_moves and _eff_letter == 'M' and _reinf_recruits(batch):
        # a MUSTER die whose batch also shows a region drag is the
        # muster-then-drag rendering (GAME19 T6: 2 regs into Minas
        # Tirith, one dragged on to Pelargir = a two-settlement
        # muster) - never an army move
        unit_moves = []
    if unit_moves:
        frm, to = unit_moves[0]
        comp = _flow_comp[(frm, to)]
        hidx = _match_army_move(legal, st, side, frm, to, comp)
        if hidx is not None:
            stats['kinds_emit']['army_move'] += 1
            s1 = _emit_sample(st, side, legal, hidx, 'army_move', turn)
            # USER truth-table #1 (game01_verification.md): an Army
            # die moves TWO armies - the SECOND LEG is its own decision.
            # State for leg 2 = leg 1 applied on a clone + a fresh die
            # of the same face restaged (the engine consumes the staged
            # die on leg 1); menu = full legal set at that state.
            out1 = [s1] if s1 is not None else []
            if len(unit_moves) >= 2:
                try:
                    import ai as _ai
                    import game_state as gs
                    from game_state import Side as _S
                    st2 = _ai.clone_state(st)
                    la1 = _ai.get_legal_actions(st2, side)
                    h1 = _match_army_move(la1, st2, side, frm, to, comp)
                    if h1 is not None:
                        st2 = _ai.apply_action(st2, la1[h1])
                        st2.pending_discard = None  # ledger-owned
                        d2 = gs.ActionDie(
                            gs.SH_FACES if side == _S.SHADOW
                            else gs.FP_FACES, side)
                        d2.current = _letter2face().get(_eff_letter)
                        d2.used = False
                        st2.dice = {_S.FREE_PEOPLES: [], _S.SHADOW: []}
                        st2.dice[side] = [d2]
                        # leg 2 = a mid-die PENDING decision (single-die
                        # menu matches live pending semantics); carry
                        # the pool context so its STATE encoding still
                        # shows realistic dice
                        st2._rx_full_pools = getattr(
                            st, '_rx_full_pools', None)
                        f2, t2 = unit_moves[1]
                        c2 = _flow_comp[(f2, t2)]
                        la2 = _ai.get_legal_actions(st2, side)
                        h2 = _match_army_move(la2, st2, side, f2, t2, c2)
                        if h2 is not None:
                            stats['kinds_emit']['army_move_leg2'] += 1
                            s2 = _emit_sample(st2, side, la2, h2,
                                              'army_move', turn)
                            if s2 is not None:
                                out1.append(s2)
                        else:
                            stats['drop_reasons']['move_leg2_unmatched'] += 1
                except Exception as _e2:
                    stats['drop_reasons'][
                        'move_leg2_exc:%s' % str(_e2)[:16]] += 1
            return out1 if len(out1) != 1 else out1[0]
        # from-region has no legal army move at all -> the die was NOT an
        # army move (event-card-granted move, character flight w/o leader in
        # the reconstruction, or a post-combat advance): fall through to the
        # other matchers rather than force a wrong army_move label.
        any_from = any(rr.sanitize(str(a.params.get('from', ''))) == frm
                       for a in legal if a.action_type == 'army_move')
        if any_from:
            stats['drop_reasons']['move_comp_unmatched'] += 1
            return None
        # Diagnose unmatched, owned, die-staged movement separately from
        # card effects and enemy retreat gestures. The rendered destination
        # alone cannot repair a missing legal candidate.
        _own_comp = True
        try:
            import ai as _ai9
            import game_state as _gs9
            for _r9 in batch['records']:
                if _r9.get('op') != 'move':
                    continue
                if rr.sanitize(str(_r9.get('from', ''))) != frm or \
                        rr.sanitize(str(_r9.get('to', ''))) != to:
                    continue
                for _tok9 in _group_tokens(str(_r9.get('obj', ''))):
                    _p9 = rr.parse_piece(_tok9)
                    if _p9 and _p9[0] == 'unit' and \
                            _ai9._side_of_nation(
                                _gs9.Nation[_p9[1]]) != side:
                        # enemy units moved on OUR die = a defender
                        # retreat/reposition gesture inside a battle
                        # window, never our army-move decision (T4
                        # Pelargir: the FP retreat drag mis-emitted
                        # as a SHADOW army_move)
                        _own_comp = False
        except Exception:
            pass
        if _own_comp and batch['die_face'] in ('A', 'C', 'H', 'M', 'W') \
                and _card_play_title(batch) is None:
            # W3-037 (2026-09-03): a move-in on a region holding ONLY
            # enemy units is an ATTACK by rule (ai._get_army_move_actions:
            # has_enemy and not has_own -> never offered as army_move) -
            # the recovery must not fabricate an army_move label for it
            # (2024-05-21 SlickRick L338: Muster die + Nenya, the FP
            # garrison withdrew into Dol Amroth before the move-in, no
            # battle; the attack rides the ring probe + the withdrawal
            # emitter).  Drop with its own reason, never relabel.
            _enemy_dest = False
            try:
                import ai as _ai9d
                _ex9 = None
                for _k9, _a9 in st.armies.items():
                    if rr.sanitize(str(_k9)) == to:
                        _ex9 = _a9
                        break
                _us9 = list(getattr(_ex9, 'units', None) or ())
                if _us9:
                    _he9 = any(_ai9d._side_of_nation(u.nation) != side
                               for u in _us9)
                    _ho9 = any(_ai9d._side_of_nation(u.nation) == side
                               for u in _us9)
                    _enemy_dest = bool(_he9 and not _ho9)
            except Exception:
                _enemy_dest = False
            if _enemy_dest:
                stats['drop_reasons']['move_recovery_enemy_dest'] += 1
                return None
            # A rendered move does not establish a legal payment or army
            # composition. GAME10 L1196 spends a Muster die on two army
            # moves without a recorded Ring. Never fabricate a candidate
            # merely to match that gesture.
            stats['drop_reasons']['move_recovery_not_in_public_menu'] += 1
            return None
        # else fall through (P/M face card move, or leaderless C flight)

    # 2) MUSTER: recruit / elite / minion (first placed figure) ----------
    # W4-007/W4-008: FINAL placements (muster-then-drag / staging drag /
    # put-back followed for every kind, not only regulars) - see
    # _muster_placements.  The engine's Nazgul recruit is muster_recruit
    # type=leader nation=SAURON (game_logic._muster_one appends a Nazgul
    # for Sauron), so kind NAZGUL rides the LEADER matcher.
    recruits = _muster_placements(batch)
    if recruits:
        # W4-026: one Muster die = 1 Elite, or 2 Regulars, or 2 Leaders,
        # or 1 Regular + 1 Leader (rulebook p.26; a Minion is the whole
        # die).  More figure-slots than that is NOT one die's muster
        # (five 24-game batches place THREE Isengard regulars with
        # Saruman in Orthanc, OPEN_QUESTIONS W4-Q4; a 2-regulars ->
        # 2-elites swap is W4-Q3): no engine action carries it, so no
        # row - the batch drops under muster_over_capacity:<signature>
        # (R3 = three regulars, E2 = two elites, ...) instead of a
        # first-figure row that mislabels the decision.
        _slots = 0
        _sig = {}
        for (_n8, _r8, _k8, _c8) in recruits:
            _slots += int(_c8) * (2 if _k8 in ('ELITE', 'MINION') else 1)
            _sig[_k8[0]] = _sig.get(_k8[0], 0) + int(_c8)
        if _slots > 2 * _batch_dice_spent(batch, (batch.get('die_face'),
                                                  _eff_letter)):
            _sig8 = ''.join('%s%d' % (k8, _sig[k8]) for k8 in sorted(_sig))
            # W4-036: Saruman's Muster-die ability (USER ruling 1.266;
            # engine half W4-032 HELD with GEN43) - one Regular in each
            # of Orthanc / North Dunland / South Dunland (R3) or two
            # Orthanc Regulars replaced by two Elites (E2) on a plain
            # Shadow Muster die with Saruman in play is a LEGAL human
            # decision the v41 surface cannot carry yet: meter it as
            # muster_saruman_held:<sig> so the HELD class is visible
            # per game, apart from the genuine over-capacity drops
            # (folded double musters, E1R2 illegal moves, card musters
            # keep the old key).  Still no row either way.
            _shape32 = _saruman_ability_shape(batch, recruits, _eff_letter,
                                              st)
            if _shape32 is not None:
                # W4-032 (engine half landed; USER rulings 1.266 /
                # 1.266-b / 1.270): the ability is a LEARNED Muster-die
                # action - R3 = muster_recruit saruman='settlements'
                # (one Isengard Regular in each of Orthanc / North
                # Dunland / South Dunland), E2 = muster_elite
                # saruman='swap2' at Orthanc.  Match the batch's shape
                # against the live menu and emit the row; a shape the
                # engine does not offer in the reconstructed state (pool
                # short, settlement captured, no Saruman on the engine
                # board) keeps the W4-036 meter tick below.
                if _shape32 == 'R3':
                    _hidx32 = _match_index(
                        legal, lambda a: a.action_type == 'muster_recruit'
                        and a.params.get('saruman') == 'settlements')
                else:
                    _hidx32 = _match_index(
                        legal, lambda a: a.action_type == 'muster_elite'
                        and a.params.get('saruman') == 'swap2'
                        and rr.sanitize(str(a.params.get('region', '')))
                        == 'orthanc')
                if _hidx32 is not None:
                    _ph32 = ('muster_recruit' if _shape32 == 'R3'
                             else 'muster_elite')
                    stats['kinds_emit']['muster_saruman_%s'
                                        % _shape32.lower()] += 1
                    stats['detected']['muster_saruman_row:%s' % _sig8] += 1
                    # sponsor condition 1: while the action encoder
                    # aliases this action with the plain Orthanc twin,
                    # stamp the sample so the corpus packer refuses it.
                    # S-001 (2026-09-04): the encoder now tells them
                    # apart (ai.ACT_ENC_SARUMAN_DISTINCT) - the stamp
                    # stays off and the row packs like any other; the
                    # literal below survives for a tree without the line.
                    import ai as _ai32
                    if not getattr(_ai32, 'ACT_ENC_SARUMAN_DISTINCT', False):
                        _RX_ALIAS_V41 = True
                    return _emit_sample(st, side, legal, _hidx32, _ph32,
                                        turn)
                stats['drop_reasons']['muster_saruman_held:%s' % _sig8] += 1
                # meta drop_by_reason is most_common(15)-truncated (12 of
                # the 24 cover games sit at the cap): mirror the tick in
                # the FULL detected dict so the meter is exact per game.
                stats['detected']['muster_saruman_held:%s' % _sig8] += 1
            else:
                stats['drop_reasons']['muster_over_capacity:%s' % _sig8] += 1
            return None
        nat, reg, kind, _cnt = recruits[0]
        if kind == 'MINION':
            hidx = _match_index(
                legal, lambda a: a.action_type == 'muster_minion' and
                rr.sanitize(str(a.params.get('region', ''))) == reg and
                str(a.params.get('minion', '')).lower() == nat.lower())
            if hidx is not None:
                stats['kinds_emit']['muster_minion'] += 1
                return _emit_sample(st, side, legal, hidx, 'muster_minion', turn)
            stats['drop_reasons']['muster_minion_unmatched'] += 1
            return None
        if kind == 'ELITE':
            # W4-032: the plain matcher names the PLAIN action only - the
            # Saruman ability twins (saruman='swap2' / 'settlements') live
            # at the same (type, nation, region) and would make the unique
            # match ambiguous (_match_index returns None on 2 hits)
            hidx = _match_index(
                legal, lambda a: a.action_type == 'muster_elite' and
                not a.params.get('saruman') and
                rr.sanitize(str(a.params.get('region', ''))) == reg and
                _nation_name(a) == nat)
            if hidx is not None:
                stats['kinds_emit']['muster_elite'] += 1
                return _emit_sample(st, side, legal, hidx, 'muster_elite', turn)
            stats['drop_reasons']['muster_elite_unmatched'] += 1
            return None
        if kind in ('REGULAR', 'LEADER', 'NAZGUL'):
            typ = 'regular' if kind == 'REGULAR' else 'leader'
            hidx = _match_index(
                legal, lambda a: a.action_type == 'muster_recruit' and
                not a.params.get('saruman') and
                rr.sanitize(str(a.params.get('region', ''))) == reg and
                _nation_name(a) == nat and
                str(a.params.get('type', '')) == typ)
            if hidx is not None:
                stats['kinds_emit']['muster_recruit'] += 1
                s1 = _emit_sample(st, side, legal, hidx, 'muster_recruit',
                                  turn)
                # USER fix #1 (muster variant): a Muster die recruits 2
                # regulars at 2 DIFFERENT settlements - the second
                # placement is its own decision (GAME01 T8: Edoras +
                # Folde; ledger had Edoras only).
                out1 = [s1] if s1 is not None else []
                if len(recruits) >= 2 and \
                        recruits[1][2] in ('REGULAR', 'LEADER', 'NAZGUL') \
                        and kind in ('REGULAR', 'LEADER', 'NAZGUL'):
                    try:
                        import ai as _ai
                        import game_state as gs
                        from game_state import Side as _S
                        n2, r2, k2, _c2 = recruits[1]
                        st2 = _ai.clone_state(st)
                        la1 = _ai.get_legal_actions(st2, side)
                        h1 = _match_index(
                            la1, lambda a: a.action_type == 'muster_recruit'
                            and not a.params.get('saruman')
                            and rr.sanitize(str(a.params.get('region', '')))
                            == reg and _nation_name(a) == nat and
                            str(a.params.get('type', '')) == typ)
                        if h1 is not None:
                            st2 = _ai.apply_action(st2, la1[h1])
                            st2.pending_discard = None  # ledger-owned
                            d2 = gs.ActionDie(
                                gs.SH_FACES if side == _S.SHADOW
                                else gs.FP_FACES, side)
                            # W4-007: the EFFECTIVE face (ring-converted
                            # die, step 0a) and the pool context for the
                            # leg-2 STATE encoding, as the army leg 2
                            # carries since W4-018
                            d2.current = _letter2face().get(_eff_letter)
                            d2.used = False
                            st2.dice = {_S.FREE_PEOPLES: [],
                                        _S.SHADOW: []}
                            st2.dice[side] = [d2]
                            st2._rx_full_pools = getattr(
                                st, '_rx_full_pools', None)
                            la2 = _ai.get_legal_actions(st2, side)
                            _t2 = ('regular' if k2 == 'REGULAR'
                                   else 'leader')
                            h2 = _match_index(
                                la2, lambda a:
                                a.action_type == 'muster_recruit' and
                                not a.params.get('saruman') and
                                rr.sanitize(str(a.params.get('region', '')))
                                == r2 and _nation_name(a) == n2 and
                                str(a.params.get('type', '')) == _t2)
                            if h2 is not None:
                                stats['kinds_emit']['muster_leg2'] += 1
                                s2 = _emit_sample(st2, side, la2, h2,
                                                  'muster_recruit', turn)
                                if s2 is not None:
                                    out1.append(s2)
                            else:
                                stats['drop_reasons'][
                                    'muster_leg2_unmatched'] += 1
                    except Exception as _e3:
                        stats['drop_reasons'][
                            'muster_leg2_exc:%s' % str(_e3)[:14]] += 1
                return out1 if len(out1) != 1 else out1[0]
            stats['drop_reasons']['muster_recruit_unmatched'] += 1
            return None
        stats['drop_reasons']['muster_kind_unmatched:%s' % kind] += 1
        return None

    # 3) POLITICAL ADVANCE ----------------------------------------------
    pol_nat = _political_nation(batch)
    if pol_nat is not None:
        hidx = _match_index(
            legal, lambda a: a.action_type == 'advance_political' and
            _nation_of(a) == pol_nat)
        if hidx is not None:
            stats['kinds_emit']['advance_political'] += 1
            return _emit_sample(st, side, legal, hidx, 'advance_political', turn)
        has_adv = any(a.action_type == 'advance_political' for a in legal)
        if not has_adv and legal:
            stats['nomatch'] += 1
            stats['drop_reasons']['nomatch_political'] += 1
        else:
            stats['drop_reasons']['political_unmatched'] += 1
        return None

    # 4) COMPANION SEPARATION (fellowship -> region) --------------------
    # W4-003: a separation moves the companion (progress + level)
    # regions, rendered as the fellowship->region drop plus one drag per
    # further region (2023-09-11 L88-90 Merry Fellowship->Trollshaws->
    # Weather Hills->Bree = separate TO BREE); one die separates one
    # companion or one GROUP (2023-03-03 L40-41 Strider and Boromir both
    # to Moria = the group action).  The decision is each companion's
    # FINAL region; two companions ending in different regions is no
    # single legal action (dropped, never the first one's row).
    seps = _sep_moves(batch)
    if seps:
        import rendered_replay as _rr2
        _sfin, _sord = {}, []
        for (comp_name, reg0) in seps:
            cn0 = _rr2.COMP_ALIAS.get(comp_name.lower(), comp_name)
            if cn0 not in _sfin:
                _sord.append(cn0)
            _sfin[cn0] = reg0
        for _r4 in batch['records']:
            if _r4.get('op') != 'move':
                continue
            _fk4 = rr.zone_kind(_r4.get('from', ''))
            _tk4 = rr.zone_kind(_r4.get('to', ''))
            if _fk4[0] not in ('region', 'stronghold') or \
                    _tk4[0] not in ('region', 'stronghold') or \
                    _fk4[1] == _tk4[1]:
                continue
            for _t4 in _group_tokens(_r4.get('obj', '')):
                _p4 = rr.parse_piece(_t4)
                if not _p4 or _p4[0] != 'named':
                    continue
                _cn4 = _rr2.COMP_ALIAS.get(str(_p4[1]).lower(), _p4[1])
                if _cn4 in _sfin:
                    _sfin[_cn4] = _tk4[1]
        _dests = set(_sfin[n] for n in _sord)
        if len(_dests) != 1:
            stats['drop_reasons']['separate_split_dest'] += 1
            return None
        reg = _sfin[_sord[0]]
        if reg != seps[0][1] or len(_sord) > 1:
            stats['detected']['separate_chain_netted'] += 1
        if len(_sord) == 1:
            cn = _sord[0]
            hidx = _match_index(
                legal, lambda a: a.action_type == 'companion_separate' and
                rr.sanitize(str(a.params.get('to_region', ''))) == reg and
                str(a.params.get('companion', '')) == cn)
            if hidx is None:        # fall back to destination-only (unique)
                hidx = _match_index(
                    legal, lambda a: a.action_type == 'companion_separate'
                    and 'companions' not in a.params and
                    rr.sanitize(str(a.params.get('to_region', ''))) == reg)
        else:
            _want4 = sorted(_sord)
            hidx = _match_index(
                legal, lambda a: a.action_type == 'companion_separate' and
                rr.sanitize(str(a.params.get('to_region', ''))) == reg and
                sorted(a.params.get('companions', [])) == _want4)
        if hidx is not None:
            stats['kinds_emit']['companion_separate'] += 1
            return _emit_sample(st, side, legal, hidx, 'companion_separate',
                                turn)
        stats['drop_reasons']['separate_unmatched'] += 1
        return None

    # 5) CHARACTER FLIGHTS: nazgul / companion / mouth (leader-only) -----
    lmoves = _leader_only_moves(batch)
    if lmoves:
        if side == Side.SHADOW:
            # W4-004: nazgul-class flights on netted flows (returns
            # cancelled, drags through an arrival region folded), the
            # exact count on the die's nazgul_move, one chain leg per
            # figure; no synthetic rows
            return _emit_nazgul_flights(batch, st, side, legal, stats,
                                        turn)
        frm, to, kinds, named = lmoves[0]
        # W4-003: a companion's flight is its ORIGIN -> FINAL region
        # (one drag per region in VASSAL: Legolas Moria->Dimrill
        # Dale->Lorien is ONE move Moria->Lorien; a drag back is a
        # retraction), grouped by (origin, final).  The first group
        # is the die's companion_move (the engine's full-group
        # action carries no 'companions' param, a proper subset
        # names its members); every further group is a leg of the
        # engine's '__companion_moves__' chain, matched on the
        # state after the previous leg (rulebook p.20: one Character
        # die moves any number of groups).  Supersedes the W4-001
        # companion_chain_first_hop parking (17 first-hop rows).
        cgroups = _companion_flows(batch) if named else []
        if named and not cgroups:
            stats['drop_reasons']['companion_flight_retracted'] += 1
            return None
        if cgroups:
            from game_state import Companion as _Cp3
            frm, to, named = cgroups[0]
            if len(cgroups) > 1 or any(
                    lm[0] == to for lm in lmoves):
                stats['detected']['companion_flows_netted'] += 1

            def _comps_at(_st, _reg):
                return set(
                    str(l.name) for k1, a1 in _st.armies.items()
                    if rr.sanitize(k1) == _reg
                    for l in a1.leaders if isinstance(l, _Cp3))

            def _cm_match(_la, _st, _f, _t, _nm):
                _w = sorted(_rr2_alias(list(_nm)))
                _h = _match_index(
                    _la, lambda a: a.action_type == 'companion_move'
                    and rr.sanitize(str(a.params.get('from', '')))
                    == _f and
                    rr.sanitize(str(a.params.get('to', ''))) == _t
                    and sorted(a.params.get('companions', [])) == _w)
                if _h is None and _comps_at(_st, _f) == set(_w):
                    # the full group (no 'companions' param)
                    _h = _match_index(
                        _la, lambda a: a.action_type ==
                        'companion_move' and
                        'companions' not in a.params and
                        rr.sanitize(str(a.params.get('from', '')))
                        == _f and
                        rr.sanitize(str(a.params.get('to', '')))
                        == _t)
                return _h
            hidx = _cm_match(legal, st, frm, to, named)
            if hidx is None:
                stats['drop_reasons']['companion_move_unmatched'] += 1
                return None
            kind_e = legal[hidx].action_type
            stats['kinds_emit'][kind_e] += 1
            s1 = _emit_sample(st, side, legal, hidx, kind_e, turn)
            out1 = [s1] if s1 is not None else []
            if len(cgroups) >= 2:
                stats['detected']['comp_leg2_enter'] += 1
                try:
                    import ai as _ai
                    st2 = _ai.clone_state(st)
                    la1 = _ai.get_legal_actions(st2, side)
                    h1 = _cm_match(la1, st2, frm, to, named)
                    if h1 is None:
                        stats['drop_reasons']['comp_leg1_reclone'] += 1
                    else:
                        st2 = _ai.apply_action(st2, la1[h1])
                        st2.pending_discard = None  # ledger-owned
                        for (f2, t2, nm2) in cgroups[1:]:
                            _w2 = sorted(_rr2_alias(list(nm2)))
                            la2 = _ai.get_legal_actions(st2, side)
                            co2 = [a for a in la2 if
                                   a.action_type == 'card_option']

                            def _leg_ok(a, _f=f2, _t=t2, _w=_w2):
                                ch = a.params.get('choices',
                                                  {}) or {}
                                if ch.get('_comp_done'):
                                    return False
                                return (rr.sanitize(str(ch.get(
                                    'from_region', ''))) == _f and
                                    rr.sanitize(str(ch.get(
                                        'to_region', ''))) == _t
                                    and sorted(ch.get(
                                        'companions', []) or [])
                                    == _w)
                            h2 = _match_index(la2, _leg_ok)
                            if len(co2) < 2:
                                stats['drop_reasons'][
                                    'comp_leg2_chain_closed'] += 1
                                break
                            if h2 is None:
                                # this group's leg is not in the
                                # engine's menu (2023-03-06 L194
                                # Strider Gap of Rohan->Anfalas is
                                # 4 regions for a level-3
                                # companion): skip ONLY that group
                                # - the remaining groups are
                                # independent legs of the same
                                # chain (their options carry their
                                # own names), so the human's other
                                # decisions in the die survive
                                stats['drop_reasons'][
                                    'comp_leg2_unmatched'] += 1
                                continue
                            stats['kinds_emit']['comp_leg2'] += 1
                            s2s = _emit_sample(st2, side, la2, h2,
                                               'card_option', turn)
                            if s2s is not None:
                                out1.append(s2s)
                            st2 = _ai.apply_action(st2, la2[h2])
                            st2.pending_discard = None
                except Exception as _ec2:
                    stats['drop_reasons'][
                        'comp_leg2_exc:%s' % str(_ec2)[:14]] += 1
            return out1 if len(out1) != 1 else (
                out1[0] if out1 else None)
        # no named companion in the flight (a plain-leader drag):
        # the legacy (from,to) full-group / army_move fallback
        hidx = _match_index(
            legal, lambda a: a.action_type in
            ('companion_move', 'army_move') and
            'companions' not in a.params and
            rr.sanitize(str(a.params.get('from', ''))) == frm and
            rr.sanitize(str(a.params.get('to', ''))) == to)
        if hidx is not None:
            kind_e = legal[hidx].action_type
            stats['kinds_emit'][kind_e] += 1
            return _emit_sample(st, side, legal, hidx, kind_e, turn)
        stats['drop_reasons']['companion_move_unmatched'] += 1
        return None

    # 6) WILL-OF-WEST specials: crown Aragorn / transform Gandalf --------
    ct = _crown_transform_flip(batch)
    if ct is not None:
        kind_ct, reg = ct
        if kind_ct == 'crown':
            hidx = _match_index(
                legal, lambda a: a.action_type == 'crown_aragorn')
            name_ct = 'crown_aragorn'
        else:
            hidx = _match_index(
                legal, lambda a: a.action_type == 'transform_gandalf' and
                rr.sanitize(str(a.params.get('region', ''))) == reg)
            if hidx is None:
                hidx = _match_index(
                    legal, lambda a: a.action_type == 'transform_gandalf')
            name_ct = 'transform_gandalf'
        if hidx is not None:
            stats['kinds_emit'][name_ct] += 1
            return _emit_sample(st, side, legal, hidx, name_ct, turn)
        stats['drop_reasons'][name_ct + '_unmatched'] += 1
        return None

    # 6.5) FELLOWSHIP PAWN ADVANCE (Char/Will die: feltrack move or Mordor
    # step) that never reached the hunt resolver â€” a 'fellowship'-kind batch
    # (pawn advanced with an empty Hunt box, so no roll) or a Mordor step whose
    # tile drew no roll.  The pawn-advance die-policy choice is RNG-independent
    # (the tile damage, if any, is forced randomness), so it labels cleanly.
    tb_fel = batch['truth_before']['fel']
    ta_fel = batch['truth']['fel']
    if _eff_letter in ('C', 'W') and side == Side.FREE_PEOPLES:
        # _eff_letter, not die_face: a ring-made C (GAME02 T1 H->C, no
        # 'changes' line) carries the pawn advance too
        advanced = _fsp_advance(batch) or (
            tb_fel.get('mordor') and
            ta_fel.get('step', 0) > tb_fel.get('step', 0))
        if advanced:
            hidx = _match_index(
                legal, lambda a: a.action_type == 'fellowship_move')
            if hidx is not None:
                stats['kinds_emit']['fellowship_move'] += 1
                return _emit_sample(st, side, legal, hidx,
                                    'fellowship_move', turn)

    # 7) FELLOWSHIP HIDE (revealed -> hidden, no pawn move).
    # A public flip proves the player's gesture, not a legal payment or Guide.
    # Strider permits any die only when his Guide ability is established.
    # An unmatched hide makes the reconstructed history inadmissible; never
    # fabricate a die=-1 action or infer a different Guide from the gesture.
    _hide_flip = any(
        r.get('op') == 'flip' and 'Fellowship' in str(r.get('obj', ''))
        and not r.get('revealed')
        and int(r.get('line') or 0) >= int(batch.get('start_line') or 0)
        for r in batch['records'])
    if side == Side.FREE_PEOPLES and ((not tb_fel['hidden'] and ta_fel['hidden'])
            or (_hide_flip and tb_fel.get('mordor')
                and not any(it.get('kind') == 'draw_tile'
                            for it in batch['rng']))) \
            and not _fsp_advance(batch):
        # Mordor branch (truth row T8-2): the mirror's hidden flag is
        # unreliable inside Mordor (both snapshots said hidden), so the
        # in-batch to-hidden FLIP is the anchor; the true pre-decision
        # state is REVEALED
        if tb_fel.get('mordor') and _hide_flip:
            st.fellowship.hidden = False
            # W3-004 (2026-09-02): the menu above was enumerated while
            # the mirror still said hidden, so it held NO
            # fellowship_hide and every Mordor hide bound the recovered
            # die=-1 stub below (20 rows / 24 games, 17 of them with a
            # C or W die staged: 2023-01-05 L489/L625/L675, 2023-09-11
            # L640/L753/L891 ...; the ring-made C of W3-005 too).
            # Re-enumerate on the true (revealed) state: the hide binds
            # its die and the menu is the engine's, not a stub.
            try:
                import ai as _ai7h
                legal = _ai7h.get_legal_actions(st, side)
                stats['detected']['hide_menu_reenumerated'] += 1
            except Exception:
                pass
        hidx = _match_index(
            legal, lambda a: a.action_type == 'fellowship_hide')
        if hidx is None:
            raise _Quarantine('coverage_fellowship_hide_unproven', {
                'batch_start_line': batch.get('start_line'),
                'turn': turn,
                'physical_die_face': batch.get('die_face'),
                'effective_die_face': _eff_letter,
                'reconstructed_guide': getattr(st.fellowship.guide, 'name', None),
                'reason': 'hide_payment_not_in_legal_menu'})
        stats['kinds_emit']['fellowship_hide'] += 1
        return _emit_sample(st, side, legal, hidx, 'fellowship_hide', turn)

    stats['drop_reasons']['book_' + batch['kind']] += 1
    return None


def _rr2_alias(named):
    import rendered_replay as rr
    return [rr.COMP_ALIAS.get(n.lower(), n) for n in named]


def _has_fsp(batch):
    for r in batch['records']:
        if r.get('op') == 'move' and \
                str(r.get('obj', '')).strip().lower() in ('fsp', 'the fellowship'):
            return True
    return False


_MORDOR_STEP_RE = re.compile(r'^\s*Mordor\s+(\d+)\s*-')


def _first_hunt_tile_line(batch):
    """W3-030: line of the batch's first hunt-tile DRAW record (an Eye /
    numeric face, or one of the special names in the ONE table
    log_replay.HUNT_TILE_PIECES via rendered_replay._SPECIAL_TILE_NAMES),
    else None.  Read from the records, not batch['rng']:
    rendered_replay._draw_kind classes six of the eight special tiles
    (The Ring is Mine!, On, On They Went, Give it to Uss!, Phial of
    Galadriel, Smeagol Helps Nice Master, Elven Cloaks) as DECK draws
    (W3-036), so the rng 'draw_tile' items miss them."""
    import rendered_replay as rr
    best = None
    for r in batch['records']:
        if r.get('op') != 'draw':
            continue
        w = str(r.get('what') or '').strip().lower()
        if not (w == 'eye' or re.match(r'^\d+r?$', w)
                or w in rr._SPECIAL_TILE_NAMES):
            continue
        ln = int(r.get('line') or 0)
        if best is None or ln < best:
            best = ln
    return best


def _fsp_advance(batch):
    """W3-030: did the batch ADVANCE the pawn?  Like _has_fsp, except that
    (a) the Mordor ENTRY drag (a 'The Fellowship' move onto 'Mordor 0'
    from a map region, plus the map legs dragged to the gate before it)
    is the turn-start declaration, not a die-funded pawn advance, and
    (b) Mordor track steps NET (an undone 0 -> 1 -> 0 is no advance) -
    EXCEPT that a step BACK after the batch's hunt-tile draw is the
    Stop icon's retraction (rulebook p.42 'if the tile shows a Stop
    icon, the Fellowship remains on the same step'; every Shadow
    special tile: Shelob's Lair, Give it to Uss!, The Ring is Mine!,
    On, On They Went), i.e. a HUNT EFFECT of a move that WAS made (in
    Mordor every move draws a tile), never a retraction of the decision
    (GenericPie 2023-01-02 L783-802, Genoah77 2023-03-16 L842-845, GJB
    2023-03-24 L1064-1068 / L1313-1316).  Used by the fellowship-move /
    fellowship-hide deciders only; the ring-face and spend heuristics
    keep _has_fsp.  Evidence for (a)/(b): the revealed-on-gate entry
    followed by the entry-turn die-funded hide (FateevVV 2023-01-05
    L462-464, LuconiusRex 2023-02-10 L976-978, Thingchilla 2023-09-07
    L1241-1248 with an undone step and NO tile, Thingchilla 2023-09-11
    L576-578, Columbo 2024-08-11 L678-680, Vayze 2025-01-18 L793-795)
    was labelled fellowship_move by the re-hiding mirror and, once the
    mirror is honest (W3-017), dropped book_fellowship because the entry
    drag tripped _has_fsp; corpus 1,432 such batches / 1,415 games."""
    start = int(batch.get('start_line') or 0)
    actor = next((r.get('player') for r in batch['records']
                  if int(r.get('line') or 0) == start
                  and r.get('die_stage', (None,))[0] == 'FREE_PEOPLES'), None)
    # Declaration housekeeping belongs to the preceding phase. Keep a pawn
    # moved before parking its die when both gestures follow FP's roll.
    after_roll = max((int(r.get('line') or 0) for r in batch['records']
                      if actor is not None and r.get('op') == 'roll_dice'
                      and r.get('player') == actor
                      and int(r.get('line') or 0) < start), default=0)
    recs = [r for r in batch['records'] if r.get('op') == 'move'
            and int(r.get('line') or 0) > after_roll
            and str(r.get('obj', '')).strip().lower()
            in ('fsp', 'the fellowship')]
    entry_at = None
    for i, r in enumerate(recs):
        mt = _MORDOR_STEP_RE.match(str(r.get('to') or ''))
        mf = _MORDOR_STEP_RE.match(str(r.get('from') or ''))
        if mt and not mf and \
                str(r.get('obj', '')).strip().lower() == 'the fellowship':
            entry_at = i
    tile_at = _first_hunt_tile_line(batch)
    net = 0
    for i, r in enumerate(recs):
        if entry_at is not None and i <= entry_at:
            continue
        mt = _MORDOR_STEP_RE.match(str(r.get('to') or ''))
        mf = _MORDOR_STEP_RE.match(str(r.get('from') or ''))
        if mt and mf:
            d = int(mt.group(1)) - int(mf.group(1))
            if d < 0 and tile_at is not None and \
                    int(r.get('line') or 0) > tile_at:
                continue        # Stop-icon retraction, a hunt effect
            net += d
            continue
        return True
    return net > 0


# ==========================================================================
# v27 List-2 decision emitters + unemitted-type census.
# --------------------------------------------------------------------------
# ADDITIVE by construction: no gate, ledger or admission logic is touched, so
# the per-game admit/quarantine verdict (canary 15/6, pilot 166/34) is
# byte-identical.  Each new type is legality-matched on a faithfully
# reconstructed state (DROP on no-match with a named census bucket, never a
# guess) and counted behind its own census counter (stats['detected'] =
# gesture seen; stats['kinds_emit'] = sample emitted).
# ==========================================================================
_DIECHANGE_PARSE_RE = re.compile(
    r'changes\s+(Shadow|Free)\s+Action\s+Die\s+\(([ACEHMPW])\)\s+to\s+'
    r'([ACEHMPW])\b', re.I)
_DECK_CACHE = None


def _deck_catalogues():
    """The immutable card catalogue; never installed as remaining decks."""
    global _DECK_CACHE
    if _DECK_CACHE is None:
        from cards import get_cards_by_deck
        from game_state import Side
        _DECK_CACHE = {code: [c.card_id for c in get_cards_by_deck(side, kind)]
                       for code, side, kind in (
                           ('2004', Side.FREE_PEOPLES, 'Character'),
                           ('2005', Side.FREE_PEOPLES, 'Strategy'),
                           ('2006', Side.SHADOW, 'Character'),
                           ('2007', Side.SHADOW, 'Strategy'))}
    return _DECK_CACHE


def _ensure_decks(st, line=None):
    """Install proven prefix contents; missing provenance is a counted refusal."""
    from rendered_history import HistoryUnproven
    if line is None and getattr(st, '_rx_deck_prefix', None) is not None:
        return
    if line is None:
        marker = getattr(st, '_rx_hunt_observation', None)
        line = marker[0] if marker else _RX_LINE_CUR
    if _LEDGER is None or not getattr(_LEDGER, 'enabled', False):
        raise HistoryUnproven('deck_prefix_ledger_missing')
    decks = _LEDGER.decks.before(line)
    for code, attr in (('2004', 'fp_char_deck'), ('2005', 'fp_strat_deck'),
                       ('2006', 'sh_char_deck'), ('2007', 'sh_strat_deck')):
        setattr(st, attr, decks[code])
    st._rx_deck_prefix = line


def _deck_type(name):
    nl = str(name).lower()
    if 'character' in nl:
        return 'character'
    if 'strategy' in nl:
        return 'strategy'
    return None


def _own_die_draws(batch):
    """W1-006: deck names ('character'/'strategy') of the draws that the
    P/W die itself paid for: draws of the ACTING side's own decks, after
    the staging line and BEFORE the die's action is over.

    A batch runs from one die staging to the next, so the last batch of
    a turn also carries the end-of-turn dice recovery and the phase-1
    draws of BOTH sides (2023-01-20 L659-671: SH P die draws one Strategy
    card L660, die killed L661, then the FP recovers its dice and draws
    L662-666).  One dialect draws its own phase-1 cards right after
    killing the die and before any recovery (2025-01-18 L59-66).  The
    die's action is over at the first record after the staging line
    that (a) kills an Action Die in Current Action or (b) moves an Action
    Die into a dice pool ('... Dice' zone: recovery, or a die returned to
    the pool).  Hunt-tile draws carry no deck word and are ignored."""
    own = ('shadow' if str(batch.get('die_side', '')).upper() == 'SHADOW'
           else 'free')
    start = int(batch.get('start_line') or 0)
    cut = None
    for r in batch.get('records', ()):
        ln = int(r.get('line') or 0)
        if ln <= start:
            continue        # pending_pre records of the previous batch
        if (r.get('op') == 'kill'
                and 'current action' in str(r.get('zone', '')).lower()
                and any('Action Die' in str(p)
                        for p in (r.get('pieces') or ()))):
            cut = ln
            break
        if (r.get('op') == 'move' and 'Action Die' in str(r.get('obj', ''))
                and str(r.get('to', '')).strip().lower().endswith(' dice')):
            cut = ln
            break
    out = []
    for it in batch.get('rng', ()):
        if it.get('kind') != 'draw_deck':
            continue
        ln = int(it.get('line') or 0)
        if cut is not None and ln >= cut:
            break
        val = str(it.get('value', '')).lower()
        if not val.startswith(own):
            continue        # the opponent's draws are never this die's
        d = _deck_type(val)
        if d:
            out.append(d)
    return out


def _turn_eye_counts(rendered_path, batches=None):
    """{turn: N} Shadow PRE-ROLL eye allocation = the COUNT of
    'allocates an eye' gesture lines before that turn's first die-staging.
    USER truth-table fix #2 (game01_verification.md): the old source line
    'adds N eyes to the hunt' is the POST-ROLL rolled-Eyes RNG result, NOT
    the allocation decision - it made the count wrong in 5/9 GAME01 turns
    (T4 real 1 vs 0; T7 real 3 'allocates' lines vs 1).  Turns with zero
    allocate-lines = 0 (absence-is-the-choice)."""
    import rendered_replay as rr
    ignored_stages = {int(proof['stage_line']) for batch in (batches or [])
                      for proof in list(batch.get('_rx_pre_roll_returns', [])) + list(batch.get('_rx_pre_roll_allocations', []))}
    allocated_lines = {int(proof['box_line']) for batch in (batches or [])
                       for proof in batch.get('_rx_pre_roll_allocations', [])}
    counts = {}
    staged = set()
    turn = 1
    with open(rendered_path, encoding='utf-8', errors='ignore') as f:
        for line_number, raw in enumerate(f, 1):
            s = raw.strip()
            if not s.startswith('%'):
                continue
            body = s[1:].strip()
            mt = rr._TURN_RE.match(body)
            if mt:
                turn = int(mt.group(1))
                continue
            if rr._DIE_STAGE_RE.match(body) and line_number not in ignored_stages:
                staged.add(turn)
                continue
            if turn in staged:
                continue
            if line_number in allocated_lines:
                counts[turn] = counts.get(turn, 0) + 1
                continue
            if 'allocates an eye' in body:
                counts[turn] = counts.get(turn, 0) + 1
                continue
            # GAME19 client dialect: the allocation is the pre-roll
            # DIE MOVE into the box ('Shadow Action Die (X) from SA
            # Dice to The Hunt For The Ring'), no 'allocates' line -
            # honor the group multiplier ('[2]Shadow Action Die (E)'
            # in ONE line = 2 eyes)
            if 'Shadow Action Die' in body \
                    and 'from SA Dice to The Hunt For The Ring' in body:
                _mm = re.search(r'\[(\d+)\]Shadow Action Die', body)
                counts[turn] = counts.get(turn, 0) + \
                    (int(_mm.group(1)) if _mm else 1)
    return counts


def _emit_eye_allocations(batches, eye_counts, stats):
    """One Shadow eye_allocate BC sample per turn (incl. the 0-eye case).  The
    eye-phase menu range(min_eyes, max_eyes+1) is INVARIANT to the exact Shadow
    die count whenever unused >= companions â€” always true pre-roll (>=7 dice,
    <=7 companions) â€” so staging max(companions,1) fresh unused dice reproduces
    the true menu exactly.  min_eyes defaults to 0 (the dominant case); the +1
    hunt-box-retrieval exception is not reconstructable from the snapshot and
    would only add a spurious count=0 NEGATIVE, never a mislabel (the human's
    choice is matched by count VALUE, so its index is always correct)."""
    import ai
    import game_state as gs
    from game_state import Side, ActionDieFace
    out = []
    first = {}
    for b in batches:
        first.setdefault(b['turn'], b)
    for turn in sorted(first):
        b = first[turn]
        n = int(eye_counts.get(turn, 0))
        stats['detected']['eye_allocate'] += 1
        try:
            st = _recon_state(b['truth_before'], b.get('turn'))
        except Exception:
            stats['drop_reasons']['eye_recon'] += 1
            continue
        n_comp = len(st.fellowship.companions)
        max_eyes = max(n_comp, 1)
        if n > max_eyes:
            # a card-driven eye add beyond the allocation cap -> not the
            # pre-roll allocation decision; drop (never force a wrong count)
            stats['drop_reasons']['eye_over_max'] += 1
            continue
        pool = []
        for _ in range(max_eyes):
            d = gs.ActionDie(gs.SH_FACES, Side.SHADOW)
            d.current = ActionDieFace.MUSTER
            d.used = False
            pool.append(d)
        st.dice = {Side.FREE_PEOPLES: [], Side.SHADOW: pool}
        st._eye_allocation_phase = True
        st._fp_used_hunt_box_last_turn = False
        boundaries = [int(r['line']) for r in b['records']
                      if r.get('op') in ('alloc_eye', 'roll_dice')
                      and 0 < int(r.get('line', 0)) < int(b['start_line'])]
        boundary = min(boundaries) if boundaries else None
        _install_ledger_hands(st, b, at_line=boundary)
        # Unknown pre-roll timing remains policy-only, never sorted after actions.
        st._rx_decision_time = (boundary, 0) if boundary is not None else None
        st._rx_decision_scope = 'explicit'
        try:
            legal = ai.get_legal_actions(st, Side.SHADOW)
        except Exception:
            stats['drop_reasons']['eye_legal_exc'] += 1
            continue
        hidx = _match_index(legal, lambda a: a.action_type == 'eye_allocate'
                            and int(a.params.get('count', -1)) == n)
        if hidx is None:
            stats['drop_reasons']['eye_unmatched'] += 1
            continue
        stats['kinds_emit']['eye_allocate'] += 1
        out.append(_emit_sample(st, Side.SHADOW, legal, hidx, 'eye_allocate',
                                turn))
    return out


def _dwarven_eye_return_prefix(batches, rendered_rolls, turn, roll_count,
                              player, count, raw, raw_index):
    """Pair the ordered allocation/return prefix, not a future count match.

    The raw index is the current return. Matching all earlier Eye gestures
    prevents a reordered return or a changed allocation count in either
    transport from certifying the same hunt pool. Physical IDs and whether
    a reset is idempotent are checked by the caller's current inventory.
    """
    import log_census as lc
    allowed = lc.BASE_SH_DIE_PIDS | lc.KNOWN_BONUS_SH
    expected = []
    raw_turn, rolls = 1, 0
    raw_faces = {}
    initial_faces_witnessed = False
    initial_faces_blocked = False
    for index, line in enumerate(raw[:raw_index + 1]):
        marker = re.search(r'<~Game\.7~>(\d+)\b', line)
        if marker:
            raw_turn = int(marker.group(1))
        if '<~Controls.397~>' in line:
            initial_faces_blocked = True
        if '<~Controls.408~>' in line:
            # This announcement witnesses constructor defaults only at the
            # initial fresh setup. It never resets previously observed faces
            # during play or supplies faces after an unobserved loaded state.
            if not initial_faces_witnessed and not initial_faces_blocked and not rolls and raw_turn == 1:
                raw_faces = {pid:'E' for pid in allowed}
                initial_faces_witnessed = True
        if not initial_faces_witnessed:
            early_move = lc.MOVE_RE.search(line)
            if (lc.ROLL_RE.search(line) or lc.CHANGE_RE.search(line) or lc.EYE_BTN_RE.search(line)
                    or '<~Controls.1~>' in line
                    or (early_move and set(early_move[1].split()) & allowed)):
                initial_faces_blocked = True
        roll = lc.ROLL_RE.search(line)
        if roll:
            rolls += 1
            ids, nums = roll.group(1).split(), roll.group(2).split()
            for pid, number in zip(ids, nums):
                if pid in allowed and int(number) in (*range(6), *range(10, 16)):
                    raw_faces[pid] = 'MHACPE'[int(number) % 10]
        change = lc.CHANGE_RE.search(line)
        if change and change.group(1) in allowed:
            raw_faces[change.group(1)] = change.group(2)
        actor = line.split('<~', 1)[0].lstrip('$')
        eye = lc.EYE_BTN_RE.search(line)
        if eye:
            expected.append((raw_turn, rolls, 'allocate', actor, 1, ('E',)))
            raw_faces[eye.group(1)] = 'E'
        move = lc.MOVE_RE.search(line)
        if move:
            ids, origin, target = move.group(1).split(), move.group(2), move.group(3)
            if ids and set(ids) <= allowed:
                if target == 'Game.1997' and origin in ('Game.2035', 'Game.2036'):
                    prior = index - 1
                    while prior >= 0 and (not raw[prior].strip() or
                            (re.fullmatch(r'<[^<>~]+> .*', raw[prior]) and '<~' not in raw[prior])):
                        prior -= 1
                    stage = lc.MOVE_RE.search(raw[prior]) if prior >= 0 else None
                    if (stage is None or stage.group(1).split() != ids
                            or stage.group(2) != 'Game.2043' or stage.group(3) != origin
                            or raw[prior].split('<~', 1)[0].lstrip('$') != actor
                            or any(raw_faces.get(pid) != 'E' for pid in ids)):
                        return None
                    expected.append((raw_turn, rolls, 'staged_in:' + origin, actor, len(ids), tuple('E' for _ in ids)))
                if (origin, target) == ('Game.2043', 'Game.1997'):
                    if any(pid not in raw_faces for pid in ids):return None
                    expected.append((raw_turn, rolls, 'move_in', actor, len(ids), tuple(sorted(raw_faces[pid] for pid in ids))))
                elif (origin, target) == ('Game.1997', 'Game.2043'):
                    if any(pid not in raw_faces for pid in ids):return None
                    expected.append((raw_turn, rolls, 'return', actor, len(ids), tuple(sorted(raw_faces[pid] for pid in ids))))
    if not expected or expected[-1][:5] != (turn, roll_count, 'return', player, count):
        return None

    witnessed = []
    previous = None
    for batch in batches:
        for record in batch['records']:
            prior, previous = previous, record
            event = None
            if record.get('op') == 'alloc_eye':
                event = ('allocate', record.get('player'), 1, ('E',))
            elif record.get('op') == 'move':
                direction = (record.get('from'), record.get('to'))
                if (direction[1] == 'The Hunt For The Ring'
                        and direction[0] in ('Current Action', 'Current SA Card')
                        and any(side == 'SHADOW' for side, _ in _die_tokens(str(record.get('obj', ''))))):
                    tokens = _die_tokens(str(record.get('obj', '')))
                    residual = _DIE_TOKEN_RE.sub('', str(record.get('obj', ''))).replace('group', '').strip(' (),')
                    if (residual or not tokens or any(side != 'SHADOW' or face != 'E' for side, face in tokens)
                            or prior is None or prior.get('op') != 'move'
                            or prior.get('player') != record.get('player')
                            or prior.get('from') != 'SA Dice' or prior.get('to') != direction[0]
                            or _die_tokens(str(prior.get('obj', ''))) != tokens
                            or _DIE_TOKEN_RE.sub('', str(prior.get('obj', ''))).replace('group', '').strip(' (),')):
                        return None
                    origin = 'Game.2035' if direction[0] == 'Current Action' else 'Game.2036'
                    event = ('staged_in:' + origin, record.get('player'), len(tokens),tuple(sorted(face for side,face in tokens)))
                if direction in (('SA Dice', 'The Hunt For The Ring'),
                                 ('The Hunt For The Ring', 'SA Dice')):
                    cargo = str(record.get('obj', ''))
                    residual = _DIE_TOKEN_RE.sub('', cargo).replace('group', '').strip(' (),')
                    tokens = _die_tokens(cargo)
                    if residual or not tokens or any(side != 'SHADOW' for side, face in tokens):
                        return None
                    event = ('move_in' if direction[0] == 'SA Dice' else 'return',
                             record.get('player'), len(tokens),tuple(sorted(face for side,face in tokens)))
            if event is None:
                continue
            before = sum(int(r['line']) < int(record['line']) for _, r in rendered_rolls)
            witnessed.append((int(batch['turn']), before, *event))
            if witnessed[-1] != expected[len(witnessed) - 1]:
                return None
            if len(witnessed) == len(expected):
                return record
    return None


def _dwarven_closed_group_prefix(raw, group_index, batches, ring_record,
                                 side, ring_id, target_id, face_map, ids):
    """Prove other unused dice plus a Ring selected and returned unchanged.

    The target is already in its own card area. Exactly the other dice and
    one Ring make a closed Current Action round trip before the separate
    single-Ring use. The complete raw/rendered interval is checked below;
    merely containing a Ring never certifies a mixed group as neutral.
    """
    import log_census as lc
    pool = 'Game.2042' if side == 'FREE_PEOPLES' else 'Game.2043'
    card_zone = 'Game.2037' if side == 'FREE_PEOPLES' else 'Game.2036'
    if (len(ids) != len(set(ids)) or target_id not in face_map
            or set(ids) != (set(face_map) - {target_id}) | {ring_id}
            or any(face_map[p] == 'E' for p in ids if p != ring_id)):
        return None
    if group_index < 1 or group_index + 2 >= len(raw):
        return None
    stage = lc.MOVE_RE.search(raw[group_index - 1])
    out = lc.MOVE_RE.search(raw[group_index])
    back = lc.MOVE_RE.search(raw[group_index + 1])
    ring = lc.MOVE_RE.search(raw[group_index + 2])
    actor = ring_record.get('player')
    if (any(m is None for m in (stage,out,back,ring))
            or any(raw[i].split('<~',1)[0].lstrip('$') != actor
                   for i in range(group_index - 1,group_index + 3))
            or stage.group(1).split() != [target_id] or stage.group(2) != pool or stage.group(3) != card_zone
            or out.group(1).split() != ids or out.group(2) != pool or out.group(3) != 'Game.2035'
            or back.group(1).split() != ids or back.group(2) != 'Game.2035' or back.group(3) != pool
            or ring.group(1).split() != [ring_id] or ring.group(2) != pool or ring.group(3) != 'Game.2035'):
        return None
    neutral = {'raw_stage_index':group_index - 1,'ids':ids,
               'tokens':tuple(sorted((side,face_map[p]) for p in ids if p != ring_id))}
    proof = _dwarven_staged_die_prefix(raw,group_index + 2,batches,ring_record,
                                      side,target_id,face_map[target_id],ring_id,neutral_group=neutral)
    return proof


def _dwarven_staged_die_prefix(raw, ring_index, batches, ring_record,
                              side, pid, old_face, ring_id, *, stage_after=False, selection_faces=None, neutral_group=None):
    """Prove one selected die's closed stage/reroll/return interval.

    A physical die put in Current Action solely to select a ring reroll is
    not a paid game action. Both streams must contain the same bounded
    stage, ring, single-die rolls, return and cleanup. Any intervening
    command or board effect refuses; later paid actions cannot supply proof.
    """
    import log_census as lc
    from location_codes_authoritative import AUTH_LOCATION_CODES

    pool = 'Game.2042' if side == 'FREE_PEOPLES' else 'Game.2043'
    def owner(line):
        return line.split('<~', 1)[0].lstrip('$')
    def chat(line):
        match = re.fullmatch(r'<([^<>~]+)> .*', line)
        return bool(match and '<~' not in line and match.group(1).lower() not in
                    ('auto', 'game', 'client', 'host', 'server', 'replay', 'system'))
    def zone(code):
        return AUTH_LOCATION_CODES.get(code)
    def die_move(actor, face, origin, target):
        return ('move_die', actor, face, zone(origin), zone(target))

    step = 1 if stage_after else -1
    stage_index = ring_index + step
    while 0 <= stage_index < len(raw) and (not raw[stage_index].strip() or chat(raw[stage_index])):
        stage_index += step
    if not 0 <= stage_index < len(raw):
        return None
    hesitation_events, hesitated_ids = [], []
    if stage_after and selection_faces is not None:
        # A player can briefly select a different die before committing to
        # the reroll. Only closed, unchanged single-die card-area selections
        # are neutral. No roll, card, army gesture or mixed cargo is skipped.
        card_zone = 'Game.2037' if side == 'FREE_PEOPLES' else 'Game.2036'
        while stage_index < len(raw):
            selected = lc.MOVE_RE.search(raw[stage_index])
            if selected is None:
                return None
            after = stage_index + 1
            while after < len(raw) and (not raw[after].strip() or chat(raw[after])):
                after += 1
            returned = lc.MOVE_RE.search(raw[after]) if after < len(raw) else None
            if returned is None:
                break  # The final selection must be followed by a roll.
            selected_ids = selected.group(1).split()
            if (len(selected_ids) != 1 or selected_ids[0] not in selection_faces
                    or selected.group(2) != pool or selected.group(3) != card_zone
                    or returned.group(1).split() != selected_ids
                    or returned.group(2) != card_zone or returned.group(3) != pool
                    or owner(raw[stage_index]) != owner(raw[ring_index])
                    or owner(raw[after]) != owner(raw[ring_index])):
                return None
            neutral_id = selected_ids[0]
            neutral_face = selection_faces[neutral_id]
            if neutral_face == 'E':
                return None
            hesitation_events.extend((die_move(owner(raw[ring_index]), neutral_face, pool, card_zone),
                                      die_move(owner(raw[ring_index]), neutral_face, card_zone, pool)))
            hesitated_ids.append(neutral_id)
            stage_index = after + 1
            while stage_index < len(raw) and (not raw[stage_index].strip() or chat(raw[stage_index])):
                stage_index += 1
        if stage_index >= len(raw):
            return None
    if neutral_group is not None:
        if stage_after:
            return None
        stage_index = neutral_group['raw_stage_index']
    stage = lc.MOVE_RE.search(raw[stage_index])
    token = lc.MOVE_RE.search(raw[ring_index])
    if (stage is None or token is None or stage.group(1).split() != [pid]
            or stage.group(2) != pool or stage.group(3) not in ('Game.2035', 'Game.2037' if side == 'FREE_PEOPLES' else 'Game.2036')
            or token.group(1).split() != [ring_id]
            or token.group(2) != pool
            or owner(raw[stage_index]) != owner(raw[ring_index])
            or owner(raw[ring_index]) != ring_record.get('player')):
        return None
    actor = owner(raw[ring_index])
    selection_zone = stage.group(3)
    events = [die_move(actor, old_face, pool, selection_zone),
              ('ring', actor, zone(pool), zone(token.group(3)))]
    if neutral_group is not None:
        cargo = neutral_group['tokens']
        events[1:1] = [('move_group', actor, cargo, 1, zone(pool), 'Current Action'),
                       ('move_group', actor, cargo, 1, 'Current Action', zone(pool))]
    if stage_after:
        events.reverse()
        events[1:1] = hesitation_events
    phase, face, rolls = 'roll', old_face, 0
    ring_zone = token.group(3)
    end_index = None
    return_event_index = None
    paid_stage_events = set()
    for index in range(max(ring_index, stage_index) + 1, len(raw)):
        line = raw[index]
        if not line.strip() or chat(line):
            continue
        roll = lc.ROLL_RE.search(line)
        move = lc.MOVE_RE.search(line)
        kill = lc.KILL_RE.search(line)
        if (move is not None and move.group(1).split() == [ring_id]
                and move.group(2) == ring_zone
                and move.group(3) in ('Game.2035', 'Game.2036', 'Game.2037')):
            # Moving the same compensation counter between current-card
            # areas does not consume another ring or a game action.
            events.append(('ring', owner(line), zone(ring_zone), zone(move.group(3))))
            ring_zone = move.group(3)
        elif roll is not None and phase == 'roll':
            if owner(line) != actor or roll.group(1).split() != [pid]:
                return None
            numbers = roll.group(2).split()
            if len(numbers) != 1:
                return None
            value = int(numbers[0])
            if side == 'FREE_PEOPLES' and value in range(5):
                face = 'WCPMH'[value]
            elif side == 'SHADOW' and value in (*range(6), *range(10, 16)):
                face = 'MHACPE'[value % 10]
            else:
                return None
            events.append(('roll', actor, (face,)))
            rolls += 1
        elif move is not None and phase == 'roll' and rolls:
            # Either human may return the selected physical die. Actor
            # identity must still match the paired rendered return below.
            if (move.group(1).split() != [pid]
                    or move.group(2) != selection_zone or move.group(3) != pool):
                return None
            return_event_index = len(events)
            events.append(die_move(owner(line), face, selection_zone, pool))
            phase = 'cleanup'
        elif move is not None and phase == 'cleanup' and selection_faces is not None:
            # A completed selection can be cleaned up after staging the next
            # action die. This new stage is preserved as an actual action;
            # it is never folded away with the earlier selection gesture.
            next_ids = move.group(1).split()
            if (len(next_ids) != 1 or next_ids[0] == pid or next_ids[0] not in selection_faces
                    or move.group(2) != pool or move.group(3) not in ('Game.2035', 'Game.2036', 'Game.2037')
                    or owner(line) != actor):
                return None
            paid_stage_events.add(len(events))
            events.append(die_move(actor, selection_faces[next_ids[0]], pool, move.group(3)))
        elif kill is not None and phase == 'cleanup':
            if kill.group(1).split() != [ring_id]:
                return None
            origin = re.match(r'<~([\w.]+)~>', line[kill.end():])
            if origin is None or not (origin.group(1) == ring_zone or ('<~Controls.644~>' in line and origin.group(1) == 'Game.2035' and ring_zone in ('Game.2035','Game.2036','Game.2037'))):
                return None
            events.append(('kill', owner(line), zone(origin.group(1))))
            end_index = index
            break
        else:
            return None
    if end_index is None:
        return None

    records = sorted((r for b in batches for r in b['records']), key=lambda r: int(r['line']))
    token_index = next((i for i, r in enumerate(records) if r is ring_record), None)
    if token_index is None or (token_index == 0 and not stage_after):
        return None
    start = token_index if stage_after else token_index - (3 if neutral_group is not None else 1)
    if start < 0:
        return None
    normalized = []
    end_line = None
    for record in records[start:]:
        op, player = record.get('op'), record.get('player')
        if op == 'move':
            cargo = str(record.get('obj', ''))
            if cargo.strip().lower() == 'dwarven ring':
                event = ('ring', player, record.get('from'), record.get('to'))
            else:
                residual = _DIE_TOKEN_RE.sub('', cargo).replace('group', '').strip(' (),')
                tokens = _die_tokens(cargo)
                if neutral_group is not None and residual.lower() == 'dwarven ring':
                    event = ('move_group', player, tuple(sorted(tokens)), 1, record.get('from'), record.get('to'))
                else:
                    if residual or len(tokens) != 1 or tokens[0][0] != side:
                        return None
                    event = ('move_die', player, tokens[0][1], record.get('from'), record.get('to'))
        elif op == 'roll_dice':
            event = ('roll', player, tuple(record.get('faces', [])))
        elif op == 'kill' and [str(p).strip().lower() for p in record.get('pieces', [])] == ['dwarven ring']:
            event = ('kill', player, record.get('zone'))
        else:
            return None
        normalized.append(event)
        if len(normalized) > len(events) or event != events[len(normalized) - 1]:
            return None
        if len(normalized) == len(events):
            end_line = int(record['line'])
            break
    if end_line is None:
        return None
    return {'pid': pid, 'raw_stage_line': stage_index + 1,
            'raw_cleanup_line': end_index + 1,
            'rendered_stage_line': int(records[token_index + 1 + len(hesitation_events) if stage_after else start]['line']),
            'rendered_selection_lines': [int(records[start + k]['line']) for k, event in enumerate(events)
                                         if event[0] in ('move_die','move_group') and k not in paid_stage_events],
            'hesitated_ids': hesitated_ids,
            'neutral_group_ids': list(neutral_group['ids']) if neutral_group is not None else [],
            'rendered_cleanup_line': end_line,
            'rendered_return_line': int(records[start + return_event_index]['line'])}

def _fold_dwarven_selection_batches(batches):
    """Retain certified selection gestures without inventing paid actions."""
    import rendered_replay as rr
    proofs = [r['_rx_dwarven_physical']['staged_die_prefix']
              for b in batches for r in b['records']
              if r.get('_rx_dwarven_physical', {}).get('staged_die_prefix')]
    for proof in proofs:
        index = next((i for i,b in enumerate(batches)
                      if b['start_line'] == proof['rendered_stage_line']), None)
        if index is None:
            # Selecting a die in the player's card area does not itself open
            # a paid-action batch. The complete certified window may already
            # be a prelude to the next real payment.
            contained = [b for b in batches
                         if set(proof.get('rendered_selection_lines', (proof['rendered_stage_line'], proof['rendered_return_line']))) | {proof['rendered_cleanup_line']} <= {r['line'] for r in b['records']}
                         and (b.get('_phase_only') or b['start_line'] > proof['rendered_cleanup_line'])]
            if len(contained) != 1:
                raise _Quarantine('coverage_dwarven_ring_unproven', {'detail':'selection_batch_boundary_unproven'})
            for record in contained[0]['records']:
                if record['line'] in proof.get('rendered_selection_lines', (proof['rendered_stage_line'],proof['rendered_return_line'])):
                    record['_rx_dwarven_selection'] = True
                    record.pop('die_stage',None)
            continue
        if index + 1 >= len(batches):
            raise _Quarantine('coverage_dwarven_ring_unproven', {'detail':'selection_batch_boundary_unproven'})
        h,n = batches[index:index+2]
        if h['turn'] != n['turn'] or any(int(r['line']) > proof['rendered_cleanup_line'] for r in h['records']):
            raise _Quarantine('coverage_dwarven_ring_unproven', {'detail':'selection_batch_has_unproved_tail'})
        for record in h['records']:
            if record['line'] in proof.get('rendered_selection_lines', (proof['rendered_stage_line'],proof['rendered_return_line'])):
                record['_rx_dwarven_selection'] = True
                record.pop('die_stage',None)
        merged=dict(n)
        merged['records']=list(h['records'])+list(n['records'])
        merged['rng']=list(h.get('rng',[]))+list(n.get('rng',[]))
        merged['truth_before']=h['truth_before']
        merged['_chat_lo']=min(int(h.get('_chat_lo') or h['start_line']),int(n.get('_chat_lo') or n['start_line']))
        merged['_rx_dwarven_selection_proofs']=[proof]+list(n.get('_rx_dwarven_selection_proofs',[]))
        merged['kind']=rr.classify_batch(merged)
        batches[index+1]=merged
        del batches[index]
    return len(proofs)


def _annotate_legacy_bid_ring_markers(batches, rendered_path, raw_path=None):
    """Recognize old compensation-ring counters by their complete raw lifecycle.

    A normal settlement marker is never globally renamed. Only IDs visibly
    granted into FP Dice before the first roll, then consumed around a real
    single-die reroll, can supply this per-game alias.
    """
    import log_census as lc
    from location_codes_authoritative import AUTH_LOCATION_CODES
    records=[r for b in batches for r in b['records']]
    uses=[r for r in records if r.get('op')=='move' and r.get('obj')=='Free Control Marker'
          and r.get('from')=='FP Dice' and str(r.get('to','')).lower().startswith('current')]
    if not uses:return set()
    def refuse(reason):
        raise _Quarantine('coverage_dwarven_ring_unproven',{'detail':reason})
    if raw_path is None:raw_path=_raw_log_path(os.path.basename(str(rendered_path)).replace('.rendered.txt',''))
    if not raw_path or not os.path.isfile(raw_path):refuse('legacy_marker_raw_missing')
    raw=lc.undo_redo_net(read_raw_log_text(raw_path).splitlines())
    markers={str(i) for i in range(405,417)}
    first_roll=next((i for i,line in enumerate(raw) if lc.ROLL_RE.search(line)),len(raw))
    grants={};events=[]
    for i,line in enumerate(raw):
        move=lc.MOVE_RE.search(line);kill=lc.KILL_RE.search(line)
        ids=(move[1].split() if move else kill[1].split() if kill else [])
        selected=markers.intersection(ids)
        if not selected:continue
        actor=line.split('<~',1)[0].lstrip('$')
        if move and i<first_roll and (move[2],move[3])==('Game.2038','Game.2042'):
            for pid in selected:
                if pid in grants:refuse('legacy_marker_duplicate_grant')
                grants[pid]={'grant_raw_line':i+1,'actor':actor}
        events.append({'raw_line':i+1,'ids':sorted(selected),'op':'move' if move else 'kill','player':actor,
                       'from':AUTH_LOCATION_CODES.get(move[2]) if move else None,
                       'to':AUTH_LOCATION_CODES.get(move[3]) if move else None})
    if not grants:refuse('legacy_marker_no_physical_grant')
    relevant=[event for event in events if set(event['ids']).intersection(grants)]
    lifetimes={}
    hesitation_lines=set()
    for pid,grant in grants.items():
        life=[e for e in relevant if pid in e['ids']]
        # Killing the compensation counter returns its physical marker to
        # reinforcements. Later ordinary settlement uses are a new role.
        end=next((i for i,event in enumerate(life) if event['op']=='kill'),None)
        if end is not None:life=life[:end+1]
        if life:lifetimes[pid]=life[-1]['raw_line']
        from dwarven_ring_prefix import remove_marker_hesitations
        life, cancelled = remove_marker_hesitations(life, raw)
        hesitation_lines.update(cancelled)
        if len(life) == 1 and life[0]['to'] == 'FP Dice':
            continue  # A granted but never used compensation counter is valid.
        transfers = life[2:-1]
        if (len(life)<3 or life[0]['op']!='move' or life[1]['op']!='move' or life[-1]['op']!='kill'
                or life[1]['ids']!=[pid] or life[1]['from']!='FP Dice'
                or not str(life[1]['to']).lower().startswith('current')
                or any(e['op']!='move' or e['ids']!=[pid]
                    or not str(e['from']).lower().startswith('current')
                    or not str(e['to']).lower().startswith('current') for e in transfers)
                or any(e['from']!=previous['to'] for previous,e in zip(life[1:-2],transfers))):
            refuse('legacy_marker_lifecycle_unproven')
        rerolls=[];results=[];result_lines=[]
        for number,text in enumerate(raw[life[1]['raw_line']:life[-1]['raw_line']-1],life[1]['raw_line']+1):
            roll=lc.ROLL_RE.search(text)
            if roll:
                ids=roll[1].split()
                if (len(ids)!=1 or ids[0] not in lc.BASE_FP_DIE_PIDS|lc.KNOWN_BONUS_FP
                        or text.split('<~',1)[0].lstrip('$')!=life[1]['player']):
                    refuse('legacy_marker_reroll_identity')
                rerolls.append(ids[0])
                values=roll[2].split()
                if len(values)!=1 or int(values[0]) not in range(5):refuse('legacy_marker_reroll_face')
                results.append('WCPMH'[int(values[0])]);result_lines.append(number)
        if not rerolls or len(set(rerolls))!=1:refuse('legacy_marker_reroll_missing')
        # A completed face-class reroll cannot consume another free result.
        # The original physical die face is read from its preceding raw roll;
        # no later action or final face supplies this premise.
        old=None
        for text in raw[:life[1]['raw_line']-1]:
            roll=lc.ROLL_RE.search(text)
            if roll and rerolls[0] in roll[1].split():
                ids=roll[1].split();values=roll[2].split()
                if len(ids)==len(values) and int(values[ids.index(rerolls[0])]) in range(5):
                    old='WCPMH'[int(values[ids.index(rerolls[0])])]
            change=lc.CHANGE_RE.search(text)
            if change and change[1]==rerolls[0]:old=change[2]
        if old is not None:
            import game_logic as gl
            classes=gl._DWARVEN_FACE_CLASSES;original=classes[_letter2face()[old]]
            if any(not (classes[_letter2face()[face]] & original) for face in results[:-1]) or classes[_letter2face()[results[-1]]] & original:
                raise _Quarantine('coverage_dwarven_ring_unproven',{
                    'detail':'legacy_reroll_class_or_completion','ring_id':pid,'die_id':rerolls[0],
                    'original_face':old,'results':results,'raw_roll_lines':result_lines})
    relevant=[e for e in relevant if any(e['raw_line']<=lifetimes[pid] for pid in e['ids'] if pid in lifetimes)]
    public=[r for r in records if
            (r.get('op')=='move' and 'Free Control Marker' in str(r.get('obj',''))
             and (r.get('from')=='FP Dice' or r.get('to')=='FP Dice'
                  or str(r.get('from','')).lower().startswith('current')
                  and str(r.get('to','')).lower().startswith('current')))
            or (r.get('op')=='kill' and any('Free Control Marker' in str(p) for p in r.get('pieces',[]))
                and str(r.get('zone','')).lower().startswith('current'))]
    if len(public)!=len(relevant):refuse('legacy_marker_transport_count')
    for record,event in zip(public,relevant):
        text=str(record.get('obj','')) if record['op']=='move' else ', '.join(record.get('pieces',[]))
        n=sum(int(m[1]) if m[1] else 1 for m in re.finditer(r'(?:\[(\d+)\])?Free Control Marker',text))
        if (record['op']!=event['op'] or record.get('player')!=event['player'] or n!=len(event['ids'])
                or (record['op']=='move' and any(record.get(k)!=event[k] for k in ('from','to')))):
            refuse('legacy_marker_transport_identity')
        record['_rx_legacy_bid_ring']=event
        if event['raw_line'] in hesitation_lines:
            record['_rx_legacy_ring_hesitation'] = True
        if record['op']=='move':record['obj']=record['obj'].replace('Free Control Marker','Dwarven Ring')
        else:record['pieces']=[str(p).replace('Free Control Marker','Dwarven Ring') for p in record['pieces']]
    batches[0]['_rx_legacy_ring_hesitation_lines']=sorted(hesitation_lines)
    batches[0]['_rx_legacy_bid_ring_ids']=sorted(grants)
    batches[0]['_rx_legacy_bid_ring_cleanup_lines']=lifetimes
    return set(grants)


def _annotate_dwarven_physical_rolls(batches, rendered_path, raw_path=None):
    """Certify ring rerolls from raw piece IDs, without borrowing a later roll.

    Scope is logs containing a Dwarven Ring use. Rendered letters alone do
    not identify the rerolled die. Unknown IDs, transport disagreement, an
    unproved original face or non-start-of-turn use quarantine the game.
    """
    import copy
    import log_census as lc
    if raw_path is None:
        raw_path = _raw_log_path(os.path.basename(str(rendered_path)).replace('.rendered.txt', ''))
    cancelled_raw = None
    if raw_path and os.path.isfile(raw_path):
        from cancelled_ring_roll import normalize
        try:
            original_text = read_raw_log_text(raw_path)
        except RawLogEncodingError as error:
            raise _Quarantine(error.reason, error.detail) from error
        normalize(batches, rendered_path, original_text.splitlines())
        cancelled_raw = _paired_physical_raw_view(batches, original_text)
    from unstaged_ring_payment import discover
    unstaged = discover(batches, cancelled_raw) if cancelled_raw is not None else {}
    legacy_ring_ids=set(batches[0].get('_rx_legacy_bid_ring_ids',[])) if batches else set()
    ring_pids=lc.DWARVEN_RING_PIDS|legacy_ring_ids
    legacy_cleanup=batches[0].get('_rx_legacy_bid_ring_cleanup_lines',{}) if batches else {}
    legacy_hesitations=set(batches[0].get('_rx_legacy_ring_hesitation_lines', ())) if batches else set()
    ring_records = [(b, r) for b in batches for r in b['records']
                    if r.get('op') == 'move' and not r.get('_rx_legacy_ring_hesitation')
                    and str(r.get('obj', '')).strip().lower() == 'dwarven ring'
                    and str(r.get('from', '')).strip().lower() in ('fp dice', 'sa dice')
                    and str(r.get('to', '')).lower().startswith('current')]
    ring_records += [(b, payment) for b, selection, payment, proof in unstaged.values()]
    ring_records.sort(key=lambda value: int(value[1]['line']))
    def refuse(detail, line=None):
        raise _Quarantine('coverage_dwarven_ring_unproven',
                          {'detail': detail, 'raw_line': line})

    if not ring_records:
        if any(r.get('op') == 'kill' and any(str(p).strip().lower() == 'dwarven ring'
                for p in r.get('pieces', [])) for b in batches for r in b['records']):
            refuse('ring_cleanup_without_use_prefix')
        if not any(b.get('_phase_only') or b.get('_rx_cancelled_preroll_move') for b in batches):
            return

    # A setup die stage/return can open a batch before the ring. Its board
    # snapshot is usable on either side of that staging only when every
    # intervening gesture leaves the board unchanged. Hands/discards and
    # dice are reconstructed separately at the ring's literal prefix.
    prefix_snapshots = {}
    for batch, record in ring_records:
        if record.get('_rx_unstaged_ring_payment'):
            proof = record['_rx_unstaged_ring_payment']
            entry = unstaged[proof['raw_stage_line']]
            record['_rx_dwarven_decision_line'] = proof['rendered_stage_line']
            prefix_snapshots[int(record['line'])] = copy.deepcopy(entry[1].get('decision_truth', entry[0]['truth_before']))
            continue
        from dwarven_ring_prefix import literal_snapshot
        literal = literal_snapshot(record)
        if literal is not None:
            prefix_snapshots[int(record['line'])] = literal
            continue
        if batch.get('_phase_only'):
            snapshot = record.get('decision_truth')
            if snapshot is None:
                refuse('ring_phase_prefix_snapshot_missing', record.get('line'))
            prefix_snapshots[int(record['line'])] = snapshot
            continue
        lo, hi = sorted((int(record['line']), int(batch['start_line'])))
        for between in batch['records']:
            if not lo <= int(between.get('line') or 0) < hi:
                continue
            op = between.get('op')
            harmless = op in ('roll_dice', 'pass', 'alloc_eye',
                              'add_eye', 'draw', 'view_hand')
            if op == 'other':
                harmless = bool(re.fullmatch(r'\S+ views the (?:Free Peoples|Shadow) event hand\.',
                                            str(between.get('raw', ''))))
            if op == 'move':
                cargo = str(between.get('obj', ''))
                residual = _DIE_TOKEN_RE.sub('', cargo)
                pure_dice = bool(_die_tokens(cargo)) and not \
                    residual.replace('group', '').strip(' (),')
                harmless = pure_dice or cargo.strip().lower() == 'dwarven ring'
            if op == 'kill':
                harmless = all((_die_tokens(str(p)) and not _DIE_TOKEN_RE.sub('', str(p)).strip(' (),'))
                               or str(p).lower() == 'dwarven ring'
                               for p in between.get('pieces', []))
            if not harmless:
                refuse('ring_intervening_board_gesture', between.get('line'))
        prefix_snapshots[int(record['line'])] = batch['truth_before']

    if raw_path is None:
        stem = os.path.basename(str(rendered_path)).replace('.rendered.txt', '')
        raw_path = _raw_log_path(stem)
    if not raw_path or not os.path.isfile(raw_path):
        refuse('raw_identity_missing')
    try:
        raw_text = read_raw_log_text(raw_path)
    except RawLogEncodingError as error:
        raise _Quarantine(error.reason, error.detail) from error
    raw = cancelled_raw if cancelled_raw is not None else _paired_physical_raw_view(batches, raw_text)
    sides = ('FREE_PEOPLES', 'SHADOW')
    allowed = {'FREE_PEOPLES': lc.BASE_FP_DIE_PIDS | lc.KNOWN_BONUS_FP,
               'SHADOW': lc.BASE_SH_DIE_PIDS | lc.KNOWN_BONUS_SH}
    own_zone = {'FREE_PEOPLES': 'Game.2042', 'SHADOW': 'Game.2043'}
    values = {'FREE_PEOPLES': dict(enumerate('WCPMH')),
              'SHADOW': dict(enumerate('MHACPE'))}
    rendered_rolls = [(b, r) for b in batches for r in b['records']
                      if r.get('op') == 'roll_dice']
    rendered_changes = [(b, r, _DIE_CHANGE_RE.match(str(r.get('raw', ''))))
                        for b in batches for r in b['records']
                        if r.get('op') == 'other'
                        and _DIE_CHANGE_RE.match(str(r.get('raw', '')))]
    roll_i = ring_i = 0
    change_i = 0
    turn = 1
    inventory = {'FREE_PEOPLES': set(lc.BASE_FP_DIE_PIDS),
                 'SHADOW': set(lc.BASE_SH_DIE_PIDS)}
    known_faces = {}
    zones = {pid: own_zone[side] for side in sides for pid in inventory[side]}
    faces = {s: {} for s in sides}
    available = {s: set() for s in sides}
    staged_unpaid = {s: set() for s in sides}
    allocated = set()
    returned_eyes = set()
    ring_zones = {}
    pending = None
    completed_waiting_cleanup = {}
    reroll_cleanup_dice = {}
    neutral_group_proofs = {}
    token_spent = False

    for raw_line, line in enumerate(raw, 1):
        mt = re.search(r'<~Game\.7~>(\d+)\b', line)
        if mt:
            if pending is not None:
                refuse('ring_not_completed', raw_line)
            turn = int(mt.group(1))
            reroll_cleanup_dice.clear()
            faces = {s: {} for s in sides}
            available = {s: set() for s in sides}
            staged_unpaid = {s: set() for s in sides}
            allocated = set()
            token_spent = False
        eye = lc.EYE_BTN_RE.search(line)
        if eye:
            pid = eye.group(1)
            if pid not in inventory['SHADOW'] or faces['SHADOW']:
                refuse('eye_allocation_identity_or_timing', raw_line)
            allocated.add(pid)
            returned_eyes.discard(pid)
            known_faces[pid] = 'E'
            zones[pid] = 'Game.1997'
        if raw_line in unstaged:
            batch, selection, payment, selection_proof = unstaged[raw_line]
            side = selection_proof['side']; pid = selection_proof['pid']; ring_id = selection_proof['ring_id']
            if (pending is not None or token_spent or not all(faces.values())
                    or any(f != 'E' and p not in available[sd] for sd in sides for p, f in faces[sd].items())
                    or ring_zones.get(ring_id) != selection_proof['pool']
                    or faces[side].get(pid) != selection_proof['old_face']
                    or int(batch['turn']) != turn or ring_i >= len(ring_records)
                    or ring_records[ring_i][1] is not payment):
                refuse('unstaged_ring_not_at_proven_turn_start', raw_line)
            ring_i += 1
            rline = int(payment['line'])
            payment['decision_truth'] = copy.deepcopy(prefix_snapshots[rline])
            payment['decision_truth']['_rx_hunt_observation'] = (selection_proof['rendered_stage_line'], 'snapshot')
            pending = {'side': side, 'raw_line': raw_line, 'line': selection_proof['rendered_stage_line'],
                       'ring_id': ring_id, 'before': copy.deepcopy(faces), 'allocated': sorted(allocated),
                       'inventory': {sd: sum(z == own_zone[sd] for z in ring_zones.values()) for sd in sides},
                       'results': [], 'record': payment, 'staged_die_prefix': selection_proof}
        mm = lc.MOVE_RE.search(line)
        if mm:
            ids, origin, target = mm.group(1).split(), mm.group(2), mm.group(3)
            if origin == 'Game.1997' and target == own_zone['SHADOW']:
                # Controls.757 is Java's reset-all-Eyes command. Both players
                # can issue it around the turn marker; its listed origins are
                # stale on the second reset. Only this command may repeat a
                # witnessed return of known Eye IDs already in their own pool.
                reset = '<~Controls.757~>' in line
                if (not ids or len(ids) != len(set(ids))
                        or not set(ids) <= inventory['SHADOW']
                        or any(zones.get(p) != origin and not (
                            reset and zones.get(p) == own_zone['SHADOW']
                            and p in returned_eyes and p not in allocated) for p in ids)
                        or (not faces['SHADOW'] and not reset and not set(ids) <= allocated)):
                    refuse('eye_return_identity_unproven', raw_line)
                returned = _dwarven_eye_return_prefix(
                    batches, rendered_rolls, turn, roll_i,
                    line.split('<~', 1)[0].lstrip('$'), len(ids), raw, raw_line - 1)
                if returned is None:
                    refuse('eye_return_rendered_prefix_unproven', raw_line)
                allocated.difference_update(ids)
            if (origin in own_zone.values() and target.startswith('Game.203')
                    and len(ids) > 1 and any(p in lc.DWARVEN_RING_PIDS for p in ids)):
                grouped_rings = [p for p in ids if p in lc.DWARVEN_RING_PIDS]
                side = next(s for s in sides if own_zone[s] == origin)
                other_side = next(s for s in sides if s != side)
                missing = [p for p,f in faces[side].items() if f != 'E' and p not in available[side]]
                if (len(grouped_rings) != 1 or pending is not None or token_spent
                        or not all(faces.values()) or len(missing) != 1
                        or ring_zones.get(grouped_rings[0]) != origin
                        or any(f != 'E' and p not in available[other_side] for p,f in faces[other_side].items())
                        or any(p not in available[side] or zones.get(p) != origin for p in ids if p != grouped_rings[0])
                        or ring_i >= len(ring_records)):
                    refuse('ring_group_selection_unproven', raw_line)
                selected = missing[0]
                if zones.get(selected) != ('Game.2037' if side == 'FREE_PEOPLES' else 'Game.2036'):
                    refuse('ring_group_selection_unproven', raw_line)
                proof = _dwarven_closed_group_prefix(raw,raw_line - 1,batches,ring_records[ring_i][1],
                                                     side,grouped_rings[0],selected,faces[side],ids)
                if proof is None:
                    refuse('ring_group_selection_unproven', raw_line)
                neutral_group_proofs[raw_line + 1] = proof
            for pid in ids:
                if pid in ring_pids and (pid not in legacy_cleanup or raw_line<=legacy_cleanup[pid]):
                    if (pid in ring_zones and ring_zones[pid] != origin) or \
                            (pid not in ring_zones and origin != ('Game.2038' if pid in legacy_ring_ids else 'Game.1997')):
                        refuse('ring_source_zone_mismatch', raw_line)
                    if (target.startswith('Game.203') and origin in own_zone.values()
                            and len(ids) == 1 and raw_line not in legacy_hesitations):
                        if pending is not None and pid != pending['ring_id']:
                            from dwarven_ring_prefix import reroll_complete
                            if reroll_complete(pending):
                                completed_waiting_cleanup[pending['ring_id']] = pending
                                pending = None
                        if pending is not None or ring_zones.get(pid) != origin:
                            refuse('ring_ownership_or_overlap', raw_line)
                        side = next(s for s in sides if own_zone[s] == origin)
                        if ring_i >= len(ring_records):
                            refuse('ring_transport_count', raw_line)
                        batch, record = ring_records[ring_i]
                        ring_i += 1
                        if (line.split('<~', 1)[0].lstrip('$') != record.get('player')
                                or int(batch['turn']) != turn):
                            refuse('ring_transport_side_or_turn', raw_line)
                        missing = {sd: [p for p, f in faces[sd].items()
                                        if f != 'E' and p not in available[sd]] for sd in sides}
                        stage_proof = None
                        if (all(faces.values()) and not token_spent
                                and len(missing[side]) == 1
                                and not missing[next(sd for sd in sides if sd != side)]):
                            selected = missing[side][0]
                            stage_proof = neutral_group_proofs.pop(raw_line - 1, None)
                            if stage_proof is None:
                                stage_proof = _dwarven_staged_die_prefix(
                                    raw, raw_line - 1, batches, record, side,
                                    selected, faces[side][selected], pid, selection_faces=faces[side])
                            if stage_proof is not None:
                                available[side].add(selected)
                        if not all(faces.values()):
                            refuse('ring_not_at_proven_turn_start', raw_line)
                        timing_variant = token_spent or any(
                            f != 'E' and p not in available[s]
                            for s in sides for p, f in faces[s].items())
                        paid_before_ring = token_spent or any(
                            f != 'E' and p not in available[s] and p not in staged_unpaid[s]
                            for s in sides for p, f in faces[s].items())
                        exclusion_reason = ('dwarven_ring_timing_variant' if paid_before_ring
                                            else 'dwarven_ring_selection_interval_unproven')
                        rline = int(record['line'])
                        if rline not in prefix_snapshots:
                            refuse('ring_board_prefix_missing', raw_line)
                        record['decision_truth'] = copy.deepcopy(prefix_snapshots[rline])
                        record['decision_truth']['_rx_hunt_observation'] = (rline, 'snapshot')
                        pending = {'side': side, 'raw_line': raw_line,
                                   'line': rline, 'ring_id': pid,
                                   'before': copy.deepcopy(faces),
                                   'allocated': sorted(allocated),
                                   'inventory': {s: sum(z == own_zone[s]
                                                       for z in ring_zones.values())
                                                 for s in sides},
                                   'results': [], 'record': record,
                                   'timing_variant': timing_variant,
                                   'exclusion_reason': exclusion_reason if timing_variant else None,
                                   'available_before': {s: sorted(available[s] | staged_unpaid[s]) for s in sides},
                                   'staged_die_prefix': stage_proof}
                    ring_zones[pid] = target
                if pid in lc.TOKEN_PIDS and origin in own_zone.values():
                    token_spent = True
                for side in sides:
                    if pid not in allowed[side]:
                        continue
                    if (pid in reroll_cleanup_dice and target == own_zone[side]
                            and origin == ('Game.2039' if side == 'FREE_PEOPLES' else 'Game.2041')):
                        cleanup = reroll_cleanup_dice.pop(pid)
                        candidates = [r for b in batches for r in b['records']
                            if r.get('op') == 'move' and r.get('player') == line.split('<~', 1)[0].lstrip('$')
                            and r.get('from') == ('FP Casualties' if side == 'FREE_PEOPLES' else 'SA Casualties')
                            and r.get('to') == ('FP Dice' if side == 'FREE_PEOPLES' else 'SA Dice')
                            and _die_tokens(str(r.get('obj', ''))) == [(side, faces[side][pid])]
                            and sum(int(rr['line']) < int(r['line']) for _, rr in rendered_rolls) == roll_i]
                        if len(candidates) != 1 or zones.get(pid) != origin:
                            refuse('reroll_cleanup_restore_transport', raw_line)
                        candidates[0]['_rx_physical_die_restore'] = {
                            'die_id': pid, 'cleanup_raw_line': cleanup, 'return_raw_line': raw_line}
                    if target == own_zone[side] and origin in (
                            'Game.2038', 'Game.2039', 'Game.2040', 'Game.2041'):
                        inventory[side].add(pid)
                    elif (side == 'SHADOW' and pid in lc.KNOWN_BONUS_SH
                          and origin == 'Game.2040' and target == 'Game.1997'):
                        # Players can introduce the new minion die via the
                        # Hunt Box before using the reset-Eyes button.
                        inventory[side].add(pid)
                        known_faces[pid] = 'E'
                    elif target in ('Game.2038', 'Game.2040'):
                        inventory[side].discard(pid)
                    if origin == own_zone[side] and pid not in inventory[side] \
                            and target not in ('Game.2038', 'Game.2040'):
                        refuse('die_pool_exit_without_inventory', raw_line)
                    if side == 'SHADOW' and target == 'Game.1997' and not faces[side]:
                        if (pid in lc.KNOWN_BONUS_SH and origin == 'Game.2040'):
                            zones[pid] = target
                        elif (origin not in (own_zone[side], 'Game.2035', 'Game.2036')
                                or zones.get(pid) != origin or pid not in inventory[side]):
                            refuse('eye_allocation_source_unproven', raw_line)
                        allocated.add(pid)
                        known_faces[pid] = 'E'
                    if origin == own_zone[side] and target != origin:
                        if (pid in available[side]
                                and target in ('Game.2035', 'Game.2036', 'Game.2037')):
                            staged_unpaid[side].add(pid)
                        available[side].discard(pid)
                    elif target == own_zone[side] and pid in faces[side]:
                        available[side].add(pid)
                    if target not in ('Game.2035', 'Game.2036', 'Game.2037'):
                        staged_unpaid[side].discard(pid)
                    zones[pid] = target
                    if side == 'SHADOW':
                        if origin == 'Game.1997' and target == own_zone[side]:
                            returned_eyes.add(pid)
                        else:
                            returned_eyes.discard(pid)
        if raw_line in unstaged:
            available[unstaged[raw_line][3]['side']].add(unstaged[raw_line][3]['pid'])
        mc = lc.CHANGE_RE.search(line)
        manual = re.match(r'^\$.*\s+changes\s', line)
        if pending is not None and (mc or manual):
            refuse('manual_face_change_during_ring', raw_line)
        if manual and mc is None:
            refuse('manual_face_change_format', raw_line)
        if mc:
            pid, face = mc.groups()
            if change_i >= len(rendered_changes):
                refuse('manual_face_change_transport', raw_line)
            change_batch, record, rendered_change = rendered_changes[change_i]
            change_i += 1
            side = 'FREE_PEOPLES' if rendered_change.group(1).lower() == 'free' else 'SHADOW'
            old, new = rendered_change.group(2).upper(), rendered_change.group(3).upper()
            player = line[1:].split(' changes', 1)[0]
            rendered_player = str(record['raw']).split(' changes', 1)[0]
            prefix_rolls = sum(int(r.get('line') or 0) < int(record['line'])
                               for _, r in rendered_rolls)
            prefix_rings = sum(int(r.get('line') or 0) < int(record['line'])
                               for _, r in ring_records)
            if (int(change_batch['turn']) != turn or prefix_rolls != roll_i
                    or prefix_rings != ring_i):
                refuse('manual_face_change_prefix', raw_line)
            if (pid not in inventory[side] or player != rendered_player or face != new
                    or (pid in known_faces and known_faces[pid] != old)):
                refuse('manual_face_change_transport', raw_line)
            known_faces[pid] = face
            for side in sides:
                if pid in faces[side]:
                    faces[side][pid] = face
        mr = lc.ROLL_RE.search(line)
        if mr:
            if roll_i >= len(rendered_rolls):
                refuse('roll_transport_count', raw_line)
            batch, record = rendered_rolls[roll_i]
            roll_i += 1
            player = line.split('<~', 1)[0].lstrip('$')
            ids, nums = mr.group(1).split(), mr.group(2).split()
            side = next((s for s in sides if ids and set(ids) <= allowed[s]), None)
            if (side is None or player != record.get('player')
                    or int(batch['turn']) != turn or len(ids) != len(nums)
                    or len(set(ids)) != len(ids) or not set(ids) <= allowed[side]):
                refuse('roll_identity_or_transport', raw_line)
            try:
                if side == 'SHADOW' and any(int(n) not in (*range(6), *range(10, 16)) for n in nums):
                    refuse('roll_face_value', raw_line)
                result = [values[side][int(n) if side == 'FREE_PEOPLES' else int(n) % 10]
                          for n in nums]
            except (ValueError, KeyError):
                refuse('roll_face_value', raw_line)
            if result != record.get('faces'):
                refuse('roll_rendered_face_mismatch', raw_line)
            if pending is not None:
                if (pending.get('timing_variant') and side == pending['side']
                        and len(ids) == 1 and ids[0] in pending['available_before'][side]
                        and zones.get(ids[0]) in ('Game.2035', 'Game.2036', 'Game.2037')):
                    # This corpus variant permits a reroll during the action
                    # phase. Its learned target is excluded; exact original
                    # identity still proves the subsequent physical pool.
                    available[side].add(ids[0])
                if (side == pending['side'] and len(ids) == 1
                        and ids[0] in faces[side] and ids[0] not in available[side]
                        and not pending['results']):
                    proof = _dwarven_staged_die_prefix(
                        raw, pending['raw_line'] - 1, batches, pending['record'],
                        side, ids[0], faces[side][ids[0]], pending['ring_id'],
                        stage_after=True, selection_faces=pending['before'][side])
                    if proof is not None:
                        available[side].add(ids[0])
                        pending['staged_die_prefix'] = proof
                if (side != pending['side'] or len(ids) != 1
                        or ids[0] not in faces[side] or ids[0] not in available[side]):
                    refuse('ring_target_identity', raw_line)
                selected = pending.get('staged_die_prefix')
                if ((selected is not None and selected['pid'] != ids[0])
                        or (not pending.get('timing_variant') and any(f != 'E' and p not in available[sd]
                               for sd in sides for p, f in faces[sd].items()))):
                    refuse('ring_selection_identity', raw_line)
                pid = ids[0]
                if pending['results'] and pid != pending['target_id']:
                    refuse('ring_target_changed', raw_line)
                if not pending['results']:
                    pending['target_id'] = pid
                    pending['old_face'] = faces[side][pid]
                old = faces[side][pid]
                faces[side][pid] = result[0]
                record['_rx_partial_roll'] = (old, result[0])
                if pending.get('timing_variant'):
                    record['_rx_partial_roll_zone'] = zones.get(pid)
                pending['results'].append({'raw_line': raw_line, 'face': result[0]})
            else:
                expected = inventory[side] - (allocated if side == 'SHADOW' else set())
                if (side == 'SHADOW' and not faces[side] and ids
                        and set(ids) <= allocated
                        and all(zones.get(pid) == 'Game.1997' for pid in ids)
                        and all(face == 'E' for face in result)):
                    # The client can roll an already allocated physical die
                    # to show its Eye face. It is not part of the action-dice
                    # pool and cannot replace the subsequent complete roll.
                    record['_rx_allocated_eye_display'] = tuple(ids)
                    record['op'] = 'allocated_eye_display'
                    record['_rx_roll_raw_line'] = raw_line
                    known_faces.update({pid: 'E' for pid in ids})
                    continue
                if set(ids) != expected:
                    refuse('full_roll_inventory_mismatch', raw_line)
                if faces[side] and (set(ids) != set(faces[side]) or any(
                        f != 'E' and p not in available[side] for p, f in faces[side].items())):
                    refuse('unproved_partial_or_midturn_roll', raw_line)
                faces[side] = dict(zip(ids, result))
                available[side] = set(ids)
            returned_eyes.difference_update(ids)
            known_faces.update(zip(ids, result))
            record['_rx_roll_pool_faces'] = list(faces[side].values())
            record['_rx_roll_physical'] = dict(faces[side])
            record['_rx_roll_raw_line'] = raw_line
        added_eyes = re.search(r'<~Controls\.350~>([\d ]+)', line)
        if added_eyes:
            for pid in added_eyes.group(1).split():
                if pid in inventory['SHADOW'] and known_faces.get(pid) == 'E':
                    zones[pid] = 'Game.1997'
                    returned_eyes.discard(pid)
        mk = lc.KILL_RE.search(line)
        if mk:
            for pid in mk.group(1).split():
                if pid in ring_pids and (pid not in legacy_cleanup or raw_line<=legacy_cleanup[pid]):
                    completed = completed_waiting_cleanup.pop(pid, None)
                    if completed is not None:
                        record = completed.pop('record')
                        record['_rx_dwarven_physical'] = completed
                        ring_zones[pid] = 'spent'
                        continue
                    if pending is None or pid != pending['ring_id'] or not pending['results']:
                        refuse('ring_cleanup_without_proven_reroll', raw_line)
                    pending['cleanup_raw_line'] = raw_line
                    if pending.get('target_id') in mk.group(1).split():
                        reroll_cleanup_dice[pending['target_id']] = raw_line
                    record = pending.pop('record')
                    record['_rx_dwarven_physical'] = pending
                    pending = None
                    ring_zones[pid] = 'spent'
                for side in sides:
                    available[side].discard(pid)
                    staged_unpaid[side].discard(pid)
                    if pid in allowed[side]:
                        zones[pid] = 'Game.2039' if side == 'FREE_PEOPLES' else 'Game.2041'
    if (pending is not None or completed_waiting_cleanup or roll_i != len(rendered_rolls) or ring_i != len(ring_records)
            or change_i != len(rendered_changes)):
        refuse('incomplete_physical_transport')
    _validate_unpaid_eye_transport(batches, raw)
def _roll_pools(batches):
    """{(turn, side_str): [face_letters]} from the latest complete roll prefix of
    each side each turn (username->side inferred from that side's die-staging
    records).  Used to reconstruct the elven-ring which-die menu."""
    u2side = {}
    for b in batches:
        for r in b['records']:
            if r.get('op') == 'move' and r.get('die_stage'):
                u2side[r.get('player')] = r['die_stage'][0]
    # W4-018 (2026-09-02): the turn-start letter roll PRECEDES the turn's
    # first die-staging, so parse_game files it under the batch's RECORDS
    # (pending_pre) and never under its 'rng' list - the old rng-only scan
    # returned {} for every game (the remaining-dice annotation fell back
    # to "faces the player went on to spend", _RX_EYEROLL stayed 0 and the
    # elven-ring menu was enumerated on a single die).  Scan the records;
    # the LAST roll of a side in a turn wins (a second letter roll is a
    # re-roll after a mis-roll - 2023-01-24 L641/L646, 2023-03-24
    # L1187/L1201 - the first one was never played).
    pools = {}
    for b in batches:
        for r in b['records']:
            if r.get('op') != 'roll_dice':
                continue
            side = u2side.get(r.get('player'))
            if side is None:
                continue
            pools[(b['turn'], side)] = [str(x).upper()
                                        for x in r.get('_rx_roll_pool_faces', r.get('faces', []))]
    return pools


# ---- Elven-Ring purity discriminators (amendment 1.215, 2026-07-23) --------
# A rendered "changes <side> Action Die (X) to Y" line is NOT necessarily an
# Elven Ring use.  Four confirmed NON-ring sources emit the identical line and
# a genuine ring use frequently emits NO such line at all (the counter is
# dragged and the new-face action is taken with the die never re-faced):
#   1. Will-of-the-West wildcard spend  ("changes W to X"): the FP wild die
#      being converted before use (rulebook p.20) â€” no ring involved.
#   2. Mirror of Galadriel (FP card):   sets ONE FP die to any result the
#      player chooses, INCLUDING a legal ring face (e.g. C->H) â€” so face
#      legality alone can NOT exclude it; it must be excluded by provenance.
#   3. The Lidless Eye / Words of Power (SH card): re-faces SH dice to Eye.
#   4. Manual hunt-box Eye allocation (SH die -> Eye) done as a die re-face.
# The genuine Elven-Ring use is identified by its COUNTER move: FP flips the
# counter out of an "Elven Ring" slot; Shadow consumes it from a "Dark Ring"
# slot.  We emit ONLY with that positive evidence as the nearest preceding
# gesture within the same die-spend; everything else DROPS (purity).
_RING_NAMES = frozenset({'narya', 'nenya', 'vilya'})


def _ring_use_side_of_move(rec):
    """'FREE_PEOPLES'/'SHADOW' if rec moves an Elven-Ring counter in a USE
    direction, else None.  W3-016: the 'kills <ring> in Dark Ring N'
    dialect is a Shadow use too (Vayze T5/T12)."""
    return {'fp_use': 'FREE_PEOPLES', 'sh_use': 'SHADOW'}.get(_ring_rec_kind(rec))


def _is_card_play_move(rec):
    """A card leaving a Hand for a Current-Action card slot (Mirror of
    Galadriel, The Lidless Eye, etc.)."""
    if rec.get('op') != 'move':
        return False
    frm = str(rec.get('from', '')).lower()
    to = str(rec.get('to', '')).lower()
    return 'hand' in frm and 'current' in to


def _is_die_stage_move(rec):
    """An Action Die entering the Current-Action zone = start of a die-spend."""
    if rec.get('op') != 'move':
        return False
    obj = str(rec.get('obj', '')).lower()
    to = str(rec.get('to', '')).lower()
    return 'action die' in obj and 'current action' in to


def _nearest_gesture_is_ring(recs, k, side_str):
    """Walk records backward from the change at index k, bounded to the current
    die-spend (stop at the die-stage that opened it).  Return 'ring' if the
    nearest actionable gesture is an Elven-Ring USE by side_str, 'card' for a
    card play, or None (no gesture within this die-spend)."""
    for j in range(k - 1, -1, -1):
        rj = recs[j]
        if (_ring_rec_kind(rj) or '').endswith('_back'):
            return 'retracted'
        if _ring_use_side_of_move(rj) == side_str:
            return 'ring'
        if _is_card_play_move(rj):
            return 'card'
        if _is_die_stage_move(rj):
            break   # reached the opening of this die-spend; no ring seen
    return None


def _explicit_ring_anchor(batch, index, side):
    """Find a counter behind an explicit change, respecting card boundaries.

    A counter paid just before die staging belongs to the preceding rendered
    batch. Staging is not another payment; its annotated tail remains evidence.
    """
    records = batch['records']
    # A literal reface may precede handing over the counter. Only the paired
    # raw stage/reface/counter proof can move that observation backwards.
    proved_later = [r for r in records[index + 1:]
                    if (r.get('_rx_fellowship_payment_witness') or {}).get('kind') == 'explicit_reface_before_ring'
                    and r['_rx_fellowship_payment_witness'].get('reface_line') == records[index].get('line')
                    and _ring_use_side_of_move(r) == side]
    if len(proved_later) == 1:
        return proved_later[0]
    kind = _nearest_gesture_is_ring(records, index, side)
    if kind == 'ring':
        return next(r for r in reversed(records[:index])
                    if _ring_use_side_of_move(r) == side)
    if kind is not None:
        return None
    carry = batch.get('_rx_ring_carry_context') or batch.get('_rx_ring_carry') or []
    for record in reversed(carry):
        if (_ring_rec_kind(record) or '').endswith('_back') or _is_card_play_move(record):
            return None
        if _ring_use_side_of_move(record) == side:
            return record
    return None


def _emit_token_uses(batch, stats):
    """Bid-token uses (W1-011 rewrite).  The module renders each FP
    compensation token as a die-like piece 'Hunt for the ring
    (Political)' / '(Card)' staged from FP Dice into Current Action (or
    'Current FP Card'), followed by its effect - a political advance or
    a deck draw - and the piece's kill.  Engine action: use_token
    (token_type='muster', nation) / use_token(token_type='draw', deck),
    a free action that takes an FP turn-order slot.

    Every stage record of the batch is a candidate whatever die framed
    the window (a token use has no die stage of its own, so its lines
    ride the neighbouring batch of EITHER side, or batch 0 before the
    first staging).  The effect is looked up strictly between the stage
    line and the piece's kill (never a later political move of the
    same batch); a stage returned to FP Dice is a retraction (no row).

    Decision state = truth_before + the ledger's holdings at the stage
    line (BOTH tokens when both are held) + the TRUE remaining dice of
    both sides at that moment (batch pools, minus the host batch's own
    die when it was spent before the token) - so the menu is the full
    surface the human faced (die actions + token targets + pass), not a
    token-targets-only stub.  A menu of one option is a non-decision
    and drops counted (token_single_option)."""
    import ai
    import copy
    import game_state as gs
    import rendered_replay as rr
    from game_state import Side
    out = []
    recs = sorted((r for r in batch.get('records') or []
                   if isinstance(r, dict) and r.get('line')),
                  key=lambda r: r.get('line', 0))
    stages = [r for r in recs
              if r.get('op') == 'move' and _token_kinds(r.get('obj'))
              and 'fp dice' in str(r.get('from', '')).lower()
              and 'current' in str(r.get('to', '')).lower()]
    for sr in stages:
        kind = _token_kinds(sr.get('obj'))[0]
        sline = sr.get('line', 0)
        tag = 'political' if kind == 'political' else 'card'
        # completion window: up to the piece's kill in the Current slot;
        # a move back to FP Dice before it = retraction
        kline = None
        retracted = False
        for r in recs:
            ln = r.get('line', 0)
            if ln <= sline:
                continue
            if r.get('op') == 'move' and kind in _token_kinds(r.get('obj')):
                if 'fp dice' in str(r.get('to', '')).lower():
                    retracted = True
                break
            if r.get('op') == 'kill' \
                    and kind in _token_kinds(r.get('pieces')
                                             or r.get('body')) \
                    and 'current' in str(r.get('zone', '')).lower():
                kline = ln
                break
        if retracted:
            stats['drop_reasons']['token_retracted'] += 1
            continue
        stats['detected']['token_%s_use' % tag] += 1
        window = [r for r in recs
                  if r.get('line', 0) > sline
                  and (kline is None or r.get('line', 0) < kline)]
        nation = deck = None
        if kind == 'political':
            for r in window:
                if r.get('op') == 'move' and \
                        rr.zone_kind(str(r.get('to', '')))[0] == 'political':
                    nation = rr.NATION_NAME.get(
                        str(r.get('obj', '')).strip().lower())
                    break
            if not nation:
                stats['drop_reasons']['token_pol_no_nation'] += 1
                continue
        else:
            for r in window:
                if r.get('op') == 'draw':
                    deck = _deck_type(r.get('what'))
                    if deck:
                        break
            if deck is None:
                for it in batch.get('rng') or []:
                    if it.get('kind') == 'draw_deck':
                        deck = _deck_type(it.get('value'))
                        if deck:
                            break
            if deck is None:
                stats['drop_reasons']['token_no_draw'] += 1
                continue
        try:
            st = _recon_state(batch['truth_before'], batch.get('turn'))
        except Exception:
            stats['drop_reasons']['token_recon'] += 1
            continue
        tk = _token_holdings_at(batch, sline)
        if tk is None:
            tk = {'political': int(kind == 'political'),
                  'card': int(kind == 'card'), 'used': False}
        _stamp_tokens(st, tk)
        # true dice context at the stage line: the batch pools, minus the
        # host batch's own die when the token followed its staging
        pools = batch.get('_rx_pools')
        if isinstance(pools, dict):
            pools = copy.deepcopy(pools)
            own = str(batch.get('die_side') or '')
            face = str(batch.get('die_face') or '').upper()
            if face and own in pools and \
                    sline > int(batch.get('start_line') or 0):
                info = pools.get(own) or {}
                un = list(info.get('unused') or [])
                if face in un:
                    un.remove(face)
                    info['unused'] = un
                    info['used'] = list(info.get('used') or []) + [face]
                    pools[own] = info
            try:
                st.dice = _install_full_pools(pools)
                st._rx_full_pools = pools
            except Exception:
                pools = None
        if not isinstance(pools, dict):
            d = gs.ActionDie(gs.FP_FACES, Side.FREE_PEOPLES)
            d.used = False
            st.dice = {Side.FREE_PEOPLES: [d], Side.SHADOW: []}
        _install_ledger_hands(st, batch, at_line=sline)
        # the draw-token enumeration requires non-empty decks (the
        # DECK CHOICE is the decision; contents are irrelevant)
        try:
            _ensure_decks(st)
        except Exception:
            for _dk9 in ('fp_char_deck', 'fp_strat_deck'):
                if not getattr(st, _dk9, None):
                    setattr(st, _dk9, ['__stub__'])
        try:
            legal = ai.get_legal_actions(st, Side.FREE_PEOPLES)
        except Exception:
            stats['drop_reasons']['token_legal_exc'] += 1
            continue
        if kind == 'political':
            hidx = _match_index(
                legal, lambda a: a.action_type == 'use_token'
                and a.params.get('token_type') == 'muster'
                and _nation_of(a) == str(nation).upper())
        else:
            hidx = _match_index(
                legal, lambda a: a.action_type == 'use_token'
                and a.params.get('token_type') == 'draw'
                and str(a.params.get('deck', '')).lower() == deck)
        if hidx is None:
            stats['drop_reasons']['token_%s_unmatched' % tag] += 1
            continue
        if len(legal) < 2:
            stats['drop_reasons']['token_single_option'] += 1
            continue
        stats['kinds_emit']['use_token'] += 1
        out.append(_emit_sample(st, Side.FREE_PEOPLES, legal, hidx,
                                'use_token', batch['turn']))
    return out


def _ring_decision_state(batch, side, stats, record=None):
    """Use one observation for both the ring menu and its stored encoding.

    A physical counter gesture is evidence to inspect, not permission to
    invent ownership or reset the once-per-turn latch. The original turn
    roll is never a replacement for the remaining dice.
    """
    from game_state import Side
    import copy
    snapshot = record.get('decision_truth') if record is not None else None
    if record is not None and snapshot is None and batch.get('record_truth_version'):
        stats['drop_reasons']['ring_missing_record_context'] += 1
        return None
    inventory = (snapshot.get('_rx_rings') if snapshot is not None
                 else (batch.get('_rx_rings') or {}).get('pre'))
    pools = copy.deepcopy(record.get('_rx_pools') if snapshot is not None
                          else batch.get('_rx_pools'))
    if inventory is None or pools is None:
        stats['drop_reasons']['ring_missing_context'] += 1
        return None
    key = 'fp' if side == Side.FREE_PEOPLES else 'sh'
    if inventory.get(key, 0) <= 0:
        stats['drop_reasons']['ring_no_inventory'] += 1
        return None
    if inventory.get(key + '_used'):
        stats['drop_reasons']['ring_used_this_turn'] += 1
        return None
    if record is not None and snapshot is not None:
        # A player may stage the die before changing its face. At the ring
        # decision that die is still available; later card/discard snapshots
        # correctly keep it paid. Restore only the actually staged die(s),
        # never every die spent earlier in the turn.
        for side_key, staged_faces in (record.get('_rx_staged') or {}).items():
            info = pools.get(side_key) or {}
            for face in staged_faces:
                if face in info.get('used', []):
                    info['used'].remove(face)
                    info['unused'].append(face)
            info['unused'] = sorted(info.get('unused', []))
    try:
        state = _recon_state(snapshot if snapshot is not None else batch['truth_before'],
                             batch.get('turn'))
        _stamp_rings(state, inventory)
        state.dice = _install_full_pools(pools)
        state._rx_full_pools = pools
        line = record.get('line') if record is not None else batch.get('start_line')
        _install_ledger_hands(state, batch, at_line=line)
        _ensure_decks(state)
        _install_persistent_cards(state, line or 0, killable=True)
        return state
    except Exception:
        stats['drop_reasons']['ring_recon'] += 1
        return None


def _ring_use_is_settled(record, records):
    """A later manual return cancels this use just as a native undo does."""
    kind = _ring_rec_kind(record)
    if kind not in ('fp_use', 'sh_use'):
        return False
    names = {str(record.get('obj', '')).strip().lower()}
    names.update(str(p).strip().lower() for p in record.get('pieces', []))
    for later in records:
        if later.get('line', 0) <= record.get('line', 0):
            continue
        if (_ring_rec_kind(later) == kind.replace('_use', '_back')
                and str(later.get('obj', '')).strip().lower() in names):
            return False
    return True


def _emit_shadow_hunt_ring(batch, stats):
    """A counter-funded Eye is a learned Ring choice, including unstaged use."""
    import random
    saved = random.getstate(), np.random.get_state()
    try:
        return _emit_shadow_hunt_ring_preserving_prefix(batch, stats)
    finally:
        random.setstate(saved[0])
        np.random.set_state(saved[1])


def _emit_shadow_hunt_ring_preserving_prefix(batch, stats):
    import ai
    import game_state as gs
    proof = batch.get('_rx_shadow_hunt_effect')
    if not proof or proof['kind'] != 'elven_ring_eye':
        return []
    if not batch.get('_rx_fellowship_roll_known') or batch.get('_rx_hunt_dice_uncertain'):
        return []
    record = next(r for r in batch['records'] if r.get('line') == proof['ring_line'])
    state = _ring_decision_state(batch, gs.Side.SHADOW, stats, record)
    if state is None:
        return []
    box = []
    for line, turn, face in _FP_HUNTBOX_FACES:
        if turn != batch['turn'] or line >= record['line']:
            continue
        die = next((d for d in state.dice[gs.Side.FREE_PEOPLES]
                    if d.used and d.current == _letter2face().get(face)
                    and all(d is not old for old in box)), None)
        if die is None:
            raise _Quarantine('coverage_fellowship_die_unproven', {'reason':'ring_prior_fp_hunt_die'})
        if face not in ('C','W'):
            source = _FP_HUNTBOX_BATCHES.get((line,turn))
            if source is None or _fellowship_ring_prefix(source) is None:
                raise _Quarantine('coverage_fellowship_die_unproven', {'reason':'ring_prior_fp_conversion'})
            die.current = gs.ActionDieFace.CHARACTER
        box.append(die)
    shadow = [d for d in state.dice[gs.Side.SHADOW] if d.current is gs.ActionDieFace.EYE]
    count = int(_RX_EYEALLOC.get(batch['turn'],0)) + int(batch['_rx_shadow_rolled_eyes'])
    if len(shadow) > count:
        raise _Quarantine('coverage_fellowship_die_unproven', {'reason':'ring_shadow_eye_count'})
    for _ in range(count-len(shadow)):
        die = gs.ActionDie(gs.SH_FACES,gs.Side.SHADOW);die.current=gs.ActionDieFace.EYE
        state.dice[gs.Side.SHADOW].append(die);shadow.append(die)
    for die in shadow:die.used=True
    for prior in batch.get('_rx_shadow_conversions',[]):
        die=next((d for d in state.dice[gs.Side.SHADOW] if d.used
                  and d.current==_letter2face().get(prior['face'])
                  and all(d is not old for old in shadow)),None)
        if die is None:
            raise _Quarantine('coverage_fellowship_die_unproven', {'reason':'ring_prior_shadow_conversion'})
        die.current=gs.ActionDieFace.EYE;shadow.append(die)
    state.hunt_box=box+shadow
    state.eyes_allocated=int(_RX_EYEALLOC.get(batch['turn'],0))
    state._rx_full_pools=None;state._rx_eyes_stamped=True
    legal=ai.get_legal_actions(state,gs.Side.SHADOW)
    matches=[i for i,a in enumerate(legal) if a.action_type=='elven_ring'
             and a.params.get('old_face')==_letter2face()[proof['faces'][0]].value
             and a.params.get('new_face')==gs.ActionDieFace.EYE.value]
    if not matches:
        raise _Quarantine('coverage_fellowship_die_unproven', {'reason':'ring_eye_not_in_public_menu'})
    stats['kinds_emit']['elven_ring']+=1
    stats['detected']['elven_ring_hunt_counter']+=1
    return [_emit_sample(state,gs.Side.SHADOW,legal,matches[0],'elven_ring',batch['turn'])]


def _emit_elven_rings(batch, pools, stats):
    """Extract ring uses backed by a counter gesture and a legal face change.

    Prefer explicit changes. A counter-only record may label a set of
    physically possible faces when the funded action establishes that set.
    Ordinary Will wildcard conversion without a counter is not a ring use;
    a counter-backed Will conversion is permitted (Almanac31-32).

    Menu and encoding share the remaining pools, inventory, and record-time
    board/hand context. Spent dice are not restored from the original roll.
    Missing ownership, a second use, an illegal target, and a returned use
    are rejected; no inventory or once-per-turn flag is invented to match.
    """
    import ai
    import game_state as gs
    from game_state import Side
    out = []
    explicit_action_change = False
    recs = batch['records']
    for k, r in enumerate(recs):
        if r.get('op') != 'other':
            continue
        m = _DIECHANGE_PARSE_RE.search(str(r.get('raw', '')))
        if not m:
            continue
        stats['detected']['elven_ring'] += 1
        ch_str = 'SHADOW' if m.group(1).lower() == 'shadow' else 'FREE_PEOPLES'
        ch_side = Side.SHADOW if ch_str == 'SHADOW' else Side.FREE_PEOPLES
        if ch_str == batch['die_side']:
            explicit_action_change = True
        of = _letter2face().get(m.group(2).upper())
        nf = _letter2face().get(m.group(3).upper())
        if of is None or nf is None:
            stats['drop_reasons']['ring_face'] += 1
            continue
        ring_record = _explicit_ring_anchor(batch, k, ch_str)
        if ring_record is None:
            reason = ('ring_src_wotw_wild' if m.group(2).upper() == 'W'
                      else 'ring_no_counter_move')
            stats['drop_reasons'][reason] += 1
            continue
        # AL31-32 explicitly permits a counter-backed W -> other change.
        ring_history = list(batch.get('_rx_ring_carry_context') or []) + list(recs)
        if not _ring_use_is_settled(ring_record, ring_history):
            stats['drop_reasons']['ring_retracted'] += 1
            continue
        st = _ring_decision_state(batch, ch_side, stats, ring_record)
        if st is None:
            continue
        try:
            legal = ai.get_legal_actions(st, ch_side)
        except Exception:
            stats['drop_reasons']['ring_legal_exc'] += 1
            continue
        matches = [i for i, a in enumerate(legal)
                   if a.action_type == 'elven_ring'
                   and str(a.params.get('old_face')) == of.value
                   and str(a.params.get('new_face')) == nf.value]
        if not matches:
            stats['drop_reasons']['ring_unmatched'] += 1
            continue
        stats['kinds_emit']['elven_ring'] += 1
        out.append(_emit_sample(st, ch_side, legal, matches[0], 'elven_ring',
                                batch['turn']))
    # Counter-only dialect: infer a set of physical faces supported by the
    # funded action. An explicit change with missing/conflicting provenance
    # must not become an unrelated inferred Ring label. Carried pre-staging
    # counters belong to their owner; a table-card removal payment is separate.
    if not out and not explicit_action_change:
        _use9 = _ring_counter_used(batch)
        if _use9 and (('SHADOW' if _use9 == 'sh' else 'FREE_PEOPLES')
                      == batch['die_side']):
            stats['detected']['elven_ring_implied'] += 1
            if batch.get('_rx_ring_carry'):
                stats['detected']['elven_ring_carried'] += 1
            of_l = batch['die_face']
            if _ring_is_tablekill_payment(batch):
                stats['detected']['ring_is_tablekill_payment'] += 1
                return out
            _an9 = _ring_face_analysis(batch)
            stats['detected']['ring_face_probe:%s' % _an9['reason']] += 1
            if _an9.get('exc'):
                stats['detected']['ring_face_probe_exc'] += \
                    len(_an9['exc'])
            nf_l = _an9.get('unique')
            _alt_faces9 = []
            if _an9.get('implied'):
                stats['detected']['ring_face_implied'] += 1
            if not nf_l and _an9.get('reason') == 'ambiguous' \
                    and len(_an9.get('faces') or []) >= 2:
                # 1.268 (USER 2026-09-03; sponsor pick (a)): the K enabling
                # faces are EQUALLY right - one row whose positive SET is
                # every enabling face (train_corpus listwise CE sums their
                # probabilities); the row's own face and the mirror ride
                # faces[0] (the natural face per gesture class, the
                # mirror's choice since W3-016).  The old unique-enabler
                # drop is dead.
                nf_l = _an9['faces'][0]
                _alt_faces9 = list(_an9['faces'][1:])
                stats['detected']['ring_face_set'] += 1
                stats['detected']['ring_face_set_k%d' % len(_an9['faces'])] += 1
            if not nf_l:
                stats['drop_reasons']['ring_face_%s' % _an9['reason']] += 1
                return out
            # RNG-NEUTRAL set path (W2-035 coupling, HELD): before 1.268
            # this branch returned here without touching the per-game
            # random stream; the staples below (deck population) consume
            # it, and a shifted cursor flips the hunt drivers' roll
            # fallback rows downstream (2023-11-26 Mol L998).  The set
            # row leaves the cursor exactly where the old drop left it.
            import random as _rnd9s
            _rng9s = ((_rnd9s.getstate(), np.random.get_state())
                      if _alt_faces9 else None)

            def _restore9s():
                if _rng9s is not None:
                    _rnd9s.setstate(_rng9s[0])
                    np.random.set_state(_rng9s[1])
            ch_str = batch['die_side']
            ch_side = (Side.SHADOW if ch_str == 'SHADOW'
                       else Side.FREE_PEOPLES)
            of = _letter2face().get(of_l)
            nf = _letter2face().get(nf_l)
            if of is None or nf is None:
                stats['drop_reasons']['ring_face'] += 1
                return out
            ring_records = list(batch.get('_rx_ring_carry') or []) + list(recs)
            ring_record = next((r for r in reversed(ring_records)
                                if _ring_use_side_of_move(r) == ch_str), None)
            st = _ring_decision_state(batch, ch_side, stats, ring_record)
            if st is None:
                _restore9s()
                return out
            try:
                legal = ai.get_legal_actions(st, ch_side)
            except Exception:
                stats['drop_reasons']['ring_legal_exc'] += 1
                _restore9s()
                return out
            matches = [i for i, a in enumerate(legal)
                       if a.action_type == 'elven_ring'
                       and str(a.params.get('old_face')) == of.value
                       and str(a.params.get('new_face')) == nf.value]
            if not matches:
                stats['drop_reasons']['ring_unmatched'] += 1
                _restore9s()
                return out
            _alts9 = []
            for _L9 in _alt_faces9:
                _f9 = _letter2face().get(_L9)
                _m9 = [i for i, a in enumerate(legal)
                       if a.action_type == 'elven_ring'
                       and str(a.params.get('old_face')) == of.value
                       and _f9 is not None
                       and str(a.params.get('new_face')) == _f9.value]
                if not _m9:
                    stats['drop_reasons']['ring_face_set_unmatched'] += 1
                    _restore9s()
                    return out
                _alts9.append(_m9[0])
            stats['kinds_emit']['elven_ring'] += 1
            global _RX_ALT_HIDX
            _RX_ALT_HIDX = _alts9 or None
            out.append(_emit_sample(st, ch_side, legal, matches[0],
                                    'elven_ring', batch['turn']))
            _restore9s()
    return out


# ==========================================================================
# THE ATTRIBUTION PRINCIPLE  (amendment 1.216, USER 2026-07-23)
# --------------------------------------------------------------------------
# "Identify every card decision by THE CARD BEING PLAYED + THE ACTION OCCURRING
# WITH NO OTHER ROOT CAUSE."  ONE generic mechanism, then per-key mappers:
#   1. re-drive the matched play_event through the ENGINE (ai.apply_action) so
#      state.pending_card_option holds the interior menu (get_legal_actions
#      purity â€” the SAME card_option actions self-play faces);
#   2. diff the batch mirror PRE (truth_before) vs POST (truth) resolution
#      window â€” the card is the only die-effect in its batch, so the state delta
#      HAS no other root cause;
#   3. match the UNIQUE legal card_option whose choice params are consistent
#      with that delta (region mustered / nation advanced / net source->dest /
#      companion separated / unit upgraded / casualty region), keyed on the
#      option's OWN param signature (card-agnostic).  0 or >=2 matches DROP
#      (purity, never a guess) with a per-family census bucket.
# Multi-hop drags collapse to net source->dest automatically (the mirror delta
# is PRE vs POST, never per-drag).  Per-figure nazgul chains and fuzzy-region
# battle cards are CENSUSED, not emitted (multi-decision / need chat fuzzy-match
# â€” deferred, see report).
# ==========================================================================
# Forced upgrades do not introduce an interior choice. Human recruitment
# choices remain in legal replay; AI policy projection excludes fixed types.
# 'there is another way' is NOT listed: ruling 1.229-b makes its
# Gollum-guide move-vs-heal a LEARNED interior decision.
_NONDECISION_CARDS = {
    'return of the witch-king', 'return of the witch king'}


def _region_units(snap):
    """region_key -> Counter{(nation,kind)/('SAURON','NAZGUL')/(nat,'LEADER')/
    ('NAMED',name): count} for a mirror snapshot (kinds merged, region
    sanitized)."""
    import rendered_replay as rr
    out = {}
    for k, (units, leaders) in snap['armies'].items():
        rk = rr.sanitize(k)
        c = out.setdefault(rk, Counter())
        for (nat, kind) in units:
            c[(nat, kind)] += 1
        for ld in leaders:
            if ld[0] == 'NAZGUL':
                c[('SAURON', 'NAZGUL')] += 1
            elif ld[0] == 'LEADER':
                c[(ld[1], 'LEADER')] += 1
            elif ld[0] == 'NAMED':
                c[('NAMED', ld[1])] += 1
    return out


def _snapshot_delta(tb, ta):
    """PRE->POST state delta for a card-play batch: per-region figure adds /
    removes, political changes, fellowship relocation, companion separations,
    corruption walk.  This IS the attribution surface."""
    import rendered_replay as rr
    pre, post = _region_units(tb), _region_units(ta)
    added, removed = {}, {}
    for rk in set(pre) | set(post):
        p, q = pre.get(rk, Counter()), post.get(rk, Counter())
        for key in set(p) | set(q):
            d = q[key] - p[key]
            if d > 0:
                added.setdefault(rk, Counter())[key] = d
            elif d < 0:
                removed.setdefault(rk, Counter())[key] = -d
    pol = {}
    for nat, val in ta['pol'].items():
        old = tb['pol'].get(nat)
        if old is not None and tuple(old) != tuple(val):
            pol[nat] = (tuple(old), tuple(val))
    fb = tb['fel']
    fa = ta['fel']
    fel_from = rr.sanitize(str(fb.get('region')))
    fel_to = rr.sanitize(str(fa.get('region')))
    comps_removed = [c for c in fb.get('comps', [])
                     if c not in fa.get('comps', [])]
    comps_added = [c for c in fa.get('comps', [])
                   if c not in fb.get('comps', [])]
    return {'added': added, 'removed': removed, 'pol': pol,
            'fel_from': fel_from, 'fel_to': fel_to,
            'fel_moved': fel_from != fel_to,
            'comps_removed': comps_removed, 'comps_added': comps_added,
            'corr_delta': fa.get('corruption', 0) - fb.get('corruption', 0)}


def _opt_region(ch, key):
    import rendered_replay as rr
    v = ch.get(key)
    return rr.sanitize(str(v)) if v else None


def _nat_norm(v):
    if v is None:
        return None
    return v.name if hasattr(v, 'name') else str(v).upper()


def _comp_norm(v):
    import rendered_replay as rr
    return rr.COMP_ALIAS.get(str(v).lower(), str(v))


def _added_has(added, region, pred):
    return region in added and any(pred(k) for k in added[region])


def _action_matches(ch, delta, own_side, ctx=None):
    """Modal 'action' options: BCC recruit/nazgul_moves, Lure corruption/
    eliminate, and Nazgul Strike hunt/discard_table_card/none.
    There Is Another Way uses its bounded physical mode matcher."""
    act = str(ch.get('action', '')).lower()
    added, removed = delta['added'], delta['removed']
    naz_added = any(k == ('SAURON', 'NAZGUL') for r in added for k in added[r])
    naz_removed = any(k == ('SAURON', 'NAZGUL')
                      for r in removed for k in removed[r])
    if act == 'recruit':                       # BCC: new nazgul from reinf
        # recruit = net NEW figure (no source loss); a flight delta
        # (removed at source AND added at dest) must match only the
        # nazgul_moves arm, never both (ambiguous-drop class)
        return naz_added and not naz_removed
    if act in ('nazgul_moves', 'nazgul_move'):  # BCC: nazgul flew (move)
        return naz_removed
    if act == 'eliminate':                      # Lure: companion removed
        return bool(delta['comps_removed'])
    if act == 'corruption':                     # Lure: corruption taken
        return delta['corr_delta'] > 0 and not delta['comps_removed']
    # Nazgul Strike either/or (spec 1.216): "table card dies => discard arm;
    # dice rolled for hunt => hunt arm"; neither => 'none'.
    if ctx is not None and act in ('hunt', 'discard_table_card', 'none'):
        batch = ctx['batch']
        rolled = any(it.get('kind') in ('roll_num', 'draw_tile')
                     for it in batch['rng'])
        killed = _table_card_kill_cids(batch)
        if act == 'hunt':
            return rolled and not killed
        if act == 'discard_table_card':
            tc = ch.get('target_card')
            return bool(killed) and (tc is None or str(tc) in killed
                                     or None in killed)
        return not rolled and not killed        # 'none'
    return False


def _table_card_kill_cids(batch):
    """card_ids ('/'-titled cards) killed off a table/board slot or moved
    from a Board slot off-board this batch (Nazgul Strike discard-arm
    evidence).  Unresolvable titles contribute None."""
    import rendered_replay as rr
    out = set()
    for r in batch['records']:
        if r.get('op') == 'kill':
            zk = str(r.get('zone', '')).strip().lower()
            if _BOARD_DEST_RE.match(zk):
                for p in r.get('pieces', []):
                    if '/' in str(p):
                        out.add(str(_title_to_cid(p)) if
                                _title_to_cid(p) is not None else None)
        elif r.get('op') == 'move':
            frm = str(r.get('from', '')).strip().lower()
            to = rr.zone_kind(r.get('to', ''))
            if _BOARD_DEST_RE.match(frm) and to[0] not in (
                    'region', 'stronghold') and '/' in str(r.get('obj', '')):
                cid = _title_to_cid(r.get('obj'))
                out.add(str(cid) if cid is not None else None)
    return out


def _nazgul_hops(batch):
    """Ordered per-figure nazgul flights this batch: [(from, to, is_wk)].
    A '[N] Nazgul' group token expands to N single-figure hops (the engine
    chain moves one figure per decision).  WK ALWAYS distinct (spec 1.216)."""
    import rendered_replay as rr
    hops = []
    for r in batch['records']:
        if r.get('op') != 'move':
            continue
        fk = rr.zone_kind(r.get('from', ''))
        tk = rr.zone_kind(r.get('to', ''))
        if fk[0] not in ('region', 'stronghold') or \
                tk[0] not in ('region', 'stronghold') or fk[1] == tk[1]:
            continue
        for t in _group_tokens(r.get('obj', '')):
            p = rr.parse_piece(t)
            if not p:
                continue
            if p[0] == 'nazgul':
                hops.extend([(fk[1], tk[1], False)] * p[3])
            elif p[0] == 'named' and p[1] == 'Witch King':
                hops.append((fk[1], tk[1], True))
    return hops


def _collapsed_move_chains(batch):
    """Region->region UNIT-move drags with multi-hop chains collapsed to
    their net (start, end, first_leg_to): consecutive (A->B, B->C) becomes
    (A, C, B).  Direct gesture evidence for the multi-army movement cards
    (a 3-region drag matches the single (from,to) engine option; the first
    leg carries the moved composition)."""
    moves = [(f, t) for (f, t, u) in _army_move_records(batch) if u]
    out = []
    for (f, t) in moves:
        if out and out[-1][1] == f:
            out[-1][1] = t
        else:
            out.append([f, t, t])
    return [(a, b, c) for (a, b, c) in out]


def _option_matches(ch, delta, own_side, ctx=None):
    """True iff card_option `ch` is UNIQUELY consistent with the PRE/POST delta.
    Card-agnostic: dispatched on the option's OWN param-key signature."""
    added, removed, pol = delta['added'], delta['removed'], delta['pol']
    # teleport: net source->dest (multi-hop drags already collapsed in delta)
    if ch.get('from_region') and ch.get('to_region') and '_naz_post' not in ch \
            and '_then_followup' not in ch and 'count' not in ch:
        fr, to = _opt_region(ch, 'from_region'), _opt_region(ch, 'to_region')
        return fr in removed and to in added
    # unit UPGRADE (Grey Company / Hill-trolls single): elite of nation appears
    if ch.get('strider_region') and ch.get('nation'):
        rk, nat = _opt_region(ch, 'strider_region'), _nat_norm(ch['nation'])
        return _added_has(added, rk, lambda k: k == (nat, 'ELITE'))
    # companion separation: named companions left the fellowship, land at dest
    if isinstance(ch.get('companions'), (list, tuple)) and ch['companions']:
        names = {_comp_norm(c) for c in ch['companions']}
        gone = {_comp_norm(c) for c in delta['comps_removed']}
        # W1-017: the option names the WHOLE separated group ("one
        # Companion or one group of Companions", There and Back Again /
        # I Will Go Alone build singles + every size>=2 subset).  The
        # subset test let the 1-companion options match a 2-companion
        # separation too (2024-05-20 L320 Merry+Legolas -> Helm's Deep:
        # cardint_ambiguous_separation); the gesture set must be EXACT.
        if names != gone:
            return False
        dest = _opt_region(ch, 'destination')
        if dest is None:
            return True
        return _added_has(added, dest, lambda k: k[0] == 'NAMED')
    # multi-region muster (Many Kings / Pits / Hill-trolls list)
    for lk in ('settlements', 'strongholds', 'regions'):
        if isinstance(ch.get(lk), (list, tuple)) and ch[lk]:
            import rendered_replay as rr
            regs = [rr.sanitize(str(x)) for x in ch[lk]]
            if lk == 'regions':
                # Hill-trolls upgrades: COUNT-exact matching - [r,r]
                # (two upgrades in one army) and [r1,r2] are uniquely
                # identified.  An upgrade in r nets +c (SAURON,ELITE)
                # and -c (SAURON,REGULAR).
                need = Counter(regs)
                for r9, c9 in need.items():
                    if added.get(r9, Counter()).get(
                            ('SAURON', 'ELITE'), 0) != c9:
                        return False
                    if removed.get(r9, Counter()).get(
                            ('SAURON', 'REGULAR'), 0) < c9:
                        return False
                tot9 = sum(v for r9 in added
                           for k2, v in added[r9].items()
                           if k2 == ('SAURON', 'ELITE'))
                return tot9 == sum(need.values())
            return all(r in added for r in regs)
    # single region: muster (added) / casualty target (removed) / fel relocate
    reg = None
    for k in ('target_region', 'recruit_region', 'region', 'stronghold',
              'destination', 'eagles_dest', 'boromir_region'):
        if ch.get(k):
            reg = _opt_region(ch, k)
            break
    if reg is not None:
        if reg in added or reg in removed:
            return True
        if delta['fel_moved'] and delta['fel_to'] == reg:
            return True
        return False
    # political nation (Wisdom advance / Stormcrow retreat) â€” no region key hit
    if ch.get('nation') is not None:
        return _nat_norm(ch['nation']) in pol
    if 'action' in ch:
        return _action_matches(ch, delta, own_side, ctx)
    return False


def _card_family(ch):
    """Census bucket for a card_option by its param signature."""
    if '_cargo_index' in ch:
        return 'army_cargo'
    if 'gandalf_draw' in ch:
        return 'gandalf_draw'
    if ch.get('_dmd_recruit_only'):
        return 'dmd_recruit'
    if 'unit_type' in ch:
        return 'unit_type'
    if 'casualty_operation' in ch:
        return 'card_casualty'
    # W1-017: the three direct pendings a play can open AFTER its own
    # menu (Palantir deck grant, overstack return, Mithril redraw) -
    # each has its own evidence matcher in _match_interior
    if 'palantir_deck' in ch:
        return 'palantir_deck'
    if 'overstack_remove' in ch:
        return 'overstack'
    if 'mithril_redraw' in ch:
        return 'mithril_redraw'
    if '_naz_post' in ch or '_then_followup' in ch or (
            ch.get('from_region') and 'count' in ch):
        return 'nazgul_chain'
    if ch.get('from_region') and ch.get('to_region'):
        return 'teleport'
    if ch.get('strider_region'):
        return 'upgrade'
    if isinstance(ch.get('companions'), (list, tuple)):
        return 'separation'
    if ch.get('companion_action'):
        return 'companion_action'
    for lk in ('settlements', 'strongholds', 'regions'):
        if isinstance(ch.get(lk), (list, tuple)):
            return 'multi_muster'
    if ch.get('target_fp_army') is not None:
        return 'dreadful_spells'
    for k in ('target_region', 'recruit_region', 'region', 'stronghold'):
        if ch.get(k):
            return 'region'
    if ch.get('nation') is not None:
        return 'nation'
    if ch.get('dice_count') is not None:
        return 'dice_count'
    if 'action' in ch:
        return 'action_modal'
    return 'other'


def _card_siege_withdrawal_gesture(batch, region, side_str):
    """W1-032: True when the play batch holds the defender's withdrawal
    gesture - `side_str` units ARRIVING in region `region`'s own
    stronghold box from any other zone (the anchor _emit_siege_withdrawal
    uses for die-driven attacks is region -> own box; a VASSAL mis-drag
    corrected through another box, bowzo L315-316, is still the
    withdrawal).  Here the attack is the card."""
    import rendered_replay as rr
    if not region:
        return False
    for r in batch['records']:
        if r.get('op') != 'move':
            continue
        fk = rr.zone_kind(r.get('from', ''))
        tk = rr.zone_kind(r.get('to', ''))
        if tk != ('stronghold', region) or fk == tk:
            continue
        for t in _group_tokens(r.get('obj', '')):
            p = rr.parse_piece(t)
            if p and p[0] == 'unit' and _unit_side_str(p[1]) == side_str:
                return True
    return False


def _return_valinor_roll_sequence(state, batch):
    """Bind whole recorded rolls to regions before using engine roll order.

    The client records stronghold rolls in differing orders. Flattening groups
    moves a hit to another army (GAME12: Grey Havens' 6 hit Rivendell).
    Distinct public dice counts prove a unique binding. Equal-sized groups
    remain unproven here, even if their flattened total happens to match.
    """
    import event_effects as effects
    targets = effects._return_valinor_roll_targets(state)
    groups = [record for record in batch.get('rng', []) if record.get('kind') == 'roll_num']
    owner = next((record.get('player') for record in batch.get('records', [])
                  if record.get('die_stage') and record.get('line') == batch.get('start_line')), None)
    if (not owner or len(groups) != len(targets)
            or any(group.get('player') != owner for group in groups)):
        raise ValueError('return_valinor_roll_groups_unproven')
    sizes = [count for _, count in targets]
    if len(set(sizes)) != len(sizes):
        raise ValueError('return_valinor_region_order_ambiguous')
    by_size = {}
    for group in groups:
        faces = group.get('ints', [])
        if (len(faces) in by_size or len(faces) not in sizes
                or any(type(face) is not int or not 1 <= face <= 6 for face in faces)):
            raise ValueError('return_valinor_roll_groups_unproven')
        by_size[len(faces)] = faces
    hit_regions = [region for region, count in targets if 6 in by_size[count]]
    if len(hit_regions) > 1:
        # Region order can change the shared Elven replacement pool and
        # whether a Leader survives. Distinct dice counts bind regions, but
        # do not authorize changing their casualty-resolution order.
        regions_by_size = {count: region for region, count in targets}
        recorded_hits = [regions_by_size[len(group['ints'])] for group in groups
                         if 6 in group['ints']]
        if recorded_hits != hit_regions:
            raise ValueError('return_valinor_casualty_order_unproven')
        lines = [group.get('line', 0) for group in groups]
        if any(a >= b for a, b in zip(lines, lines[1:])):
            raise ValueError('return_valinor_casualty_order_unproven')
        for index, group in enumerate(groups):
            if 6 not in group['ints']:
                continue
            region = regions_by_size[len(group['ints'])]
            upper = lines[index + 1] if index + 1 < len(lines) else batch['end_line'] + 1
            try:
                operations = _casualty_gesture_operations(
                    batch, 'FREE_PEOPLES', {region}, batch['start_line'], batch['end_line'] + 1)
                scoped = _casualty_gesture_operations(
                    batch, 'FREE_PEOPLES', {region}, lines[index] + 1, upper)
            except _CasualtyGestureError as exc:
                raise ValueError('return_valinor_casualty_order_unproven') from exc
            if not operations or operations != scoped:
                raise ValueError('return_valinor_casualty_order_unproven')
    return [face for _, count in targets for face in by_size[count]]


def _card_movement_prefix(batch, title):
    """Movement after the Event play and before combat or Hunt resolution.

    A later victorious advance can move the same figures again, but is not
    another flight or army move supplied by this card.
    """
    play_line = next((r.get('line', 0) for r in batch['records']
                      if r.get('op') == 'move' and r.get('obj') == title), None)
    if play_line is None:
        return dict(batch, records=[])
    stops = [r['line'] for r in batch['records'] if r.get('line', 0) > play_line
             and (r.get('op') == 'choose'
                  or (r.get('op') == 'move'
                      and 'hand' in str(r.get('from', '')).lower()
                      and 'current' in str(r.get('to', '')).lower()
                      and 'card' in str(r.get('to', '')).lower()))]
    stops.extend(r['line'] for r in batch.get('rng', [])
                 if r.get('line', 0) > play_line
                 and r.get('kind') in ('roll_num', 'draw_tile'))
    boundary = min(stops) if stops else batch.get('end_line', 10 ** 9)
    return dict(batch, records=[r for r in batch['records']
                                if play_line < r.get('line', 0) < boundary])


def _card_followup_attack_index(state, side, legal, batch):
    """Match a granted attack only when a unique literal advance proves it.

    The destination must contain enemy units before combat, and the movement
    must follow a recorded combat choice or roll. Battles without an advance
    remain unlabelled here; adjacency alone cannot identify their attacker.
    """
    import rendered_replay as rr
    from game_state import Side
    owner = 'SHADOW' if side == Side.SHADOW else 'FREE_PEOPLES'
    boundaries = [r.get('line', 0) for r in batch['records']
                  if r.get('op') == 'choose' or r.get('kind') == 'roll_num']
    if not boundaries:
        return None
    first = min(boundaries)
    pairs = set()
    for record in batch['records']:
        if record.get('op') != 'move' or record.get('line', 0) <= first:
            continue
        source = rr.zone_kind(record.get('from', ''))
        target = rr.zone_kind(record.get('to', ''))
        if source[0] != 'region' or target[0] != 'region' or source[1] == target[1]:
            continue
        obj = str(record.get('obj', ''))
        tokens = rr.split_group(obj.replace('group (', '').rstrip(')')) if 'group' in obj else [obj]
        units = [piece for token in tokens if (piece := rr.parse_piece(token)) and piece[0] == 'unit']
        if not units or any(_unit_side_str(piece[1]) != owner for piece in units):
            continue
        army = state.armies.get(target[1])
        if not army or not any(_unit_side_str(unit.nation.name) != owner for unit in army.units):
            continue
        pairs.add((source[1], target[1]))
    if len(pairs) != 1:
        return None
    source, target = next(iter(pairs))
    matches = [i for i, action in enumerate(legal)
               if action.action_type == 'army_attack'
               and action.params.get('card_followup')
               and action.params.get('from') == source
               and action.params.get('to') == target]
    return matches[0] if len(matches) == 1 else None


def _post_separation_names(batch, remaining, before_line):
    """Names literally separated before this guide decision, or an empty set."""
    import rendered_replay as rr
    canonical = lambda name: rr.COMP_ALIAS.get(str(name).lower(), str(name))
    before = {canonical(name) for name in batch['truth_before']['fel'].get('comps', [])}
    remaining = {canonical(name) for name in remaining}
    departed = before - remaining
    if not departed or not remaining <= before:
        return set()
    observed = set()
    for record in batch['records']:
        line = int(record.get('line') or 0)
        if not int(batch['start_line']) <= line < before_line or record.get('op') != 'move':
            continue
        source, target = rr.zone_kind(record.get('from', '')), rr.zone_kind(record.get('to', ''))
        if source[0] != 'fellowship' or target[0] not in ('region', 'stronghold'):
            continue
        for token in _group_tokens(record.get('obj', '')):
            piece = rr.parse_piece(token)
            if piece and piece[0] == 'named':
                observed.add(canonical(piece[1]))
    return departed if observed == departed else set()


def _card_guide_index(state, batch, legal):
    """Resolve a separation's guide continuation without inferring a switch."""
    import rendered_replay as rr
    remaining = {c.name for c in state.fellowship.companions}
    final = batch['truth']['fel']
    if remaining != set(final.get('comps', [])):
        raise ValueError('guide_company_not_settled')
    incoming = [record for record in batch['records']
                if record.get('_guide_gesture')
                and str(record.get('to', '')).strip().lower() == 'guide of the fellowship'
                and int(record.get('line') or 0) >= int(batch['start_line'])]
    if incoming:
        if len(incoming) != 1:
            raise ValueError('multiple_guide_nominations')
        record = incoming[0]
        before, after = record.get('decision_truth'), record.get('guide_truth_after')
        piece = rr.parse_piece(str(record.get('obj', '')))
        chosen = piece[1] if piece and piece[0] == 'named' else None
        if (before is None or after is None or chosen is None
                or set(before['fel'].get('comps', [])) != remaining
                or after['fel'].get('guide_provenance') != 'declared'
                or not rr._named_eq(after['fel'].get('guide', ''), chosen)
                or not rr._named_eq(final.get('guide', ''), chosen)
                or not _post_separation_names(batch, remaining, int(record['line']))):
            raise ValueError('unproven_post_separation_guide')
        evidence = 'explicit_nomination'
    else:
        chosen = getattr(state.fellowship.guide, 'name', None)
        if (chosen not in remaining or final.get('guide_provenance') not in ('declared', 'forced')
                or not rr._named_eq(final.get('guide', ''), chosen)
                or not _post_separation_names(batch, remaining, int(batch['end_line']))):
            raise ValueError('unproven_post_separation_keep')
        evidence = 'retained_current'
    matches = [i for i, action in enumerate(legal)
               if action.action_type == 'choose_guide'
               and rr._named_eq(action.params.get('companion', ''), chosen)]
    if len(matches) != 1:
        raise ValueError('post_separation_guide_not_legal')
    return matches[0], evidence


def _emit_card_interiors(batch, stats):
    """Attribution-principle interior-decision emitter for a card-play batch.
    Additive + independent (its own reconstruction, drops on any failure), so
    it never touches the play_event sample or any gate."""
    import ai
    import rendered_replay as rr
    from game_state import Side
    title = _card_play_title(batch)
    if title is None:
        return []
    cid = _title_to_cid(title)
    if cid is None:
        return []
    movement = cid in {_title_to_cid(name) for name in _CARD_MOVEMENT_NAMES}
    prefix = _card_movement_prefix(batch, title) if movement else batch
    if batch['kind'] in ('combat', 'combat?') and not _card_strike(batch):
        # An Event may open this battle. Its prior movement is still a
        # card decision, but an actual combat commit has no Event interior.
        windows = _combat_round_windows(batch) or [
            {'first': 10 ** 9, 'last': 10 ** 9, 'next_first': 10 ** 9}]
        commits, _, _ = _round_cards(batch, windows)
        committed = {row.get(side) for row in commits for side in ('fp', 'sh')}
        target_event = str(title).split('/')[0].replace(',', '').strip().lower() in (
            'help unlooked for', 'the fighting uruk-hai', 'grond hammer of the underworld')
        if not ((movement or target_event) and _title_class(title) == 'open'
                and cid not in committed and prefix['records']):
            return []
    elif batch['kind'] == 'hunt':
        if not ((movement and _nazgul_hops(prefix))
                or str(title).split('/')[0].strip().lower() == 'there is another way'):
            return []
    elif _commit_in_missed_battle(batch):
        return []
    st, side = _mk_state_with_die(batch, install_hands=True)
    if st is None:
        return []
    if (_REACTIVE_TABLE is not None and str(title).split('/')[0].strip().lower()
            == 'the nazgul strike!'):
        _REACTIVE_TABLE.install(st, batch.get('start_line', 0), crebain=True)
    # The card's funding face controls bonus entitlements. A Ring can turn
    # the staged die into Event even when Character dice also remain.
    global _MM_FACE
    _funded9 = _effective_lead_letter(batch)
    if cid in ('SHADOW_Character_Nazgûl_Search', 'SHADOW_Character_Nazgûl_Strike!'):
        from rendered_nazgul_hunt import funding_face
        _funded9 = funding_face(batch, _funded9)
    face = _letter2face().get(_funded9)
    if face is not None:
        _set_effective_lead_face(st, side, face)
        _MM_FACE = face
    try:
        legal = ai.get_legal_actions(st, side)
    except Exception:
        return []
    play = next((a for a in legal if a.action_type == 'play_event'
                 and str(a.params.get('card_id')) == str(cid)
                 and (cid not in ('SHADOW_Character_Nazgûl_Search', 'SHADOW_Character_Nazgûl_Strike!')
                      or a.die_idx == 0)), None)
    if play is None:
        # play unmatched (effect-pruned class): the overstack trim can
        # still be gesture-proven (T9-14 Cirdan's) - probe on the raw
        # recon with the musters brute-applied
        return _trim_probe(batch, st, side, stats, brute=True)
    staged_plan = None
    from rendered_companion_chain import literal_plan as companion_literal_plan, Unproven as CompanionUnproven
    try:
        companion_plan = companion_literal_plan(batch, title)
    except CompanionUnproven as error:
        stats['drop_reasons']['cardint_companion_chain:' + str(error)] += 1
        return []
    from rendered_card_followup import literal_plan as followup_literal_plan, Unproven as FollowupUnproven
    try:
        followup_plan = followup_literal_plan(batch, title)
    except FollowupUnproven as error:
        stats['drop_reasons']['cardint_followup:' + str(error)] += 1
        return []
    if str(title).split('/')[0].strip().lower() in (
            'the shadow lengthens', 'the shadow is moving', 'shadows gather', 'corsairs of umbar'):
        from rendered_staged_movement import literal_plan, Unproven
        try:
            staged_plan = literal_plan(batch, st, title)
        except Unproven as error:
            stats['drop_reasons']['cardint_staged:' + str(error)] += 1
            return []
    # DET census P1-b root: a play's own dice (Dreadful Spells / Spirit
    # of Mordor / card strikes) rolled LIVE here, so the opened
    # pending's hit count was engine-random fiction while the client's
    # true rolls sit in the batch (engine-driven doctrine: the client
    # generates ALL randomness).  Force d6 rolls from the batch's
    # logged stream for the play and its chain; a play with no logged
    # rolls falls through to the real generator (behavior unchanged).
    import random as _rnd9
    if cid == 'SHADOW_Strategy_Return_to_Valinor':
        try:
            _seq9 = iter(_return_valinor_roll_sequence(st, batch))
        except ValueError as error:
            stats['drop_reasons']['cardint_rng:' + str(error)] += 1
            return []
    else:
        _seq9 = iter([v for c in batch['rng'] if c.get('kind') == 'roll_num'
                      for v in (c.get('ints') or [])])
    _ori9 = _rnd9.randint

    _rng_failed9 = []
    def _rint9(a, b, _s=_seq9, _f=_ori9):
        if a == 1 and b == 6:
            value = next(_s, None)
            if type(value) is not int or not 1 <= value <= 6:
                _rng_failed9.append('missing_or_invalid_card_die')
                raise ValueError('missing_or_invalid_card_die')
            return value
        return _f(a, b)
    from rendered_card_timing import pool_signature, pre_hunt_menu
    _pre_hunt_pool9 = pool_signature(st)
    from rendered_nazgul_hunt import RecordedNazgulHunt
    _naz_hunt9 = RecordedNazgulHunt.for_card(batch, cid, stats)
    if _naz_hunt9 is not None:
        try:
            _naz_hunt9.initialize(st)
            _pre_hunt_pool9 = pool_signature(st)
        except ValueError as error:
            stats['drop_reasons'][str(error)] += 1
            return []
    _rnd9.randint = _rint9
    try:
        s2 = _naz_hunt9.apply(st, play) if _naz_hunt9 is not None else ai.apply_action(st, play)
    except Exception as e:
        stats['drop_reasons']['cardint_apply_exc:%s' % str(e)[:14]] += 1
        return []
    finally:
        _rnd9.randint = _ori9
    if _rng_failed9:
        stats['drop_reasons']['cardint_rng_unproven'] += 1
        return []
    if str(title).split('/')[0].strip().lower() == 'there is another way':
        # The first hide/move menu precedes movement and its Hunt draw.
        from rendered_hunt_observation import mark
        mark(s2, int(batch['start_line']), 'before_draw')
    # interior CHAIN loop (spec 1.216 items 3/4/5): a card may open a
    # SEQUENCE of pending menus (per-figure nazgul flights, Ents free-chain
    # plays, a post-chain either/or).  Each link is evidence-matched and
    # applied; links with a real menu (>=2 options) emit a BC sample.
    delta = _snapshot_delta(batch['truth_before'], batch['truth'])
    ctx = {'batch': batch, 'delta': delta,
           'hops': _nazgul_hops(prefix), 'hop_i': 0,
           'plays': _card_play_titles(batch), 'play_i': 1,
           'chains': _collapsed_move_chains(prefix), 'tm_i': 0,
           'chats': _batch_chats(batch), 'play_title': title,
           # W1-017: per-link consumption of the overstack return
           # gestures (one unit per __overstack__ pending)
           'ovs_used': Counter()}
    if staged_plan is not None:
        ctx['_staged_army_plan'] = staged_plan
        ctx['_staged_army_i'] = 0
    if followup_plan is not None:
        ctx['hops'] = followup_plan['hops']
    followup_cursor = 0
    if companion_plan is not None:
        ctx['_companion_card_plan'] = companion_plan
        ctx['_companion_card_cursor'] = 0
    out = []
    first = True
    for _depth in range(16):
        if _naz_hunt9 is not None and _naz_hunt9.has_continuation(s2):
            _rnd9.randint = _rint9
            try:
                s2, continuation_rows = _naz_hunt9.resume(s2)
                out.extend(continuation_rows)
            finally:
                _rnd9.randint = _ori9
            break
        if getattr(s2, 'pending_guide_change', None):
            # The companion card has completed its separation; the guide
            # choice precedes deferred draw/other card continuations. Explicit
            # nominations are labelled by _emit_guides. Silence can retain a
            # still-legal current guide, but never invents a guide-change row.
            try:
                guide_menu = ai.get_legal_actions(s2, Side.FREE_PEOPLES)
                guide_index, guide_evidence = _card_guide_index(s2, batch, guide_menu)
                s2 = ai.apply_action(s2, guide_menu[guide_index])
            except (ValueError, RuntimeError) as error:
                stats['drop_reasons']['cardint_guide:' + str(error)] += 1
                break
            stats['detected']['cardint_guide_' + guide_evidence] += 1
            continue
        # Bonus acceptance is a fixed AI policy (user 2026-09-04). Human
        # alternatives stay legal, but do not produce a learned take/decline
        # target. Keep the entitlement for the later learned discard state.
        _stage_g, _context_g = ai._gandalf_draw_context(s2)
        _card_g = getattr(s2, 'pending_card_option', None)
        if (_context_g is not None
                and (_card_g is None or _card_g.get('gandalf_draw'))
                and not any(getattr(s2, key, None) for key in (
                    'pending_card_followup', 'pending_combat', 'pending_hunt',
                    'pending_guide_change', '_overstack_regions'))):
            if cid == 'FREE_PEOPLES_Character_House_of_the_Stewards':
                batch['_rx_stewards_gandalf'] = {
                    key: _context_g[key]
                    for key in ('source_card_id', 'deck', 'die_face')}
            stats['detected']['fixed_policy_bonus_draw'] += 1
            break
        # a transient hand-limit pending raised by the play's own draw
        # (Palantir / WK Sorcerer) belongs to the ledger-based discard
        # emitter, never to this interior probe â€” isolate the menu.
        if getattr(s2, 'pending_discard', None) is not None:
            s2.pending_discard = None
        # a card's muster can OVERSTACK a besieged garrison (truth row
        # T9-14, Cirdan's Ships into the DA siege): the engine opens a
        # siege_trim pending - WHO returns is the owner's decision,
        # gesture-matched via the killed-then-returned pieces
        _pct = getattr(s2, 'pending_combat', None)
        if _pct is not None and _pct.get('phase') == 'siege_trim':
            _treg = rr.sanitize(str(_pct.get('trim_region') or
                                    _pct.get('defender_region') or ''))
            _tsd = _pct.get('deciding_side')
            _tstr = ('SHADOW' if _tsd == Side.SHADOW
                     else 'FREE_PEOPLES')
            eb, _rb, _tc = _window_trim(batch, _treg, _tstr)
            try:
                la_t = ai.get_legal_actions(s2, _tsd)
            except Exception:
                stats['drop_reasons']['cardint_trim_legal_exc'] += 1
                break
            h_t = _match_index(la_t, lambda a: a.action_type ==
                               'siege_trim' and
                               int(a.params.get('elites_back', -1)) == eb)
            if h_t is None:
                stats['drop_reasons']['cardint_trim_unmatched'] += 1
                break
            if len(la_t) >= 2:
                stats['kinds_emit']['siege_trim'] += 1
                out.append(_emit_sample(s2, _tsd, la_t, h_t,
                                        'siege_trim', batch['turn']))
            try:
                s2 = ai.apply_action(s2, la_t[h_t])
            except Exception:
                break
            continue
        # W1-032 (P, sponsor 04:50): a card-opened ATTACK on an enemy-held
        # stronghold (Corsairs of Umbar landing on Dol Amroth) opens the
        # DEFENDER's field-vs-siege decision in the engine
        # (_effect_corsairs_of_umbar -> ai._post_rearguard_transition ->
        # 'defender_siege_choice', a learned decision) - the withdrawal
        # GESTURE (defender units region R -> R's stronghold box inside
        # the play batch) evidences retreat_into_siege.  No gesture ->
        # drop: a fought answer classifies the batch 'combat' and never
        # reaches this lane, and combat_no_retreat is never guessed.
        if _pct is not None and _pct.get('phase') == 'defender_siege_choice':
            _dsd = _pct.get('deciding_side')
            if not isinstance(_dsd, Side):
                stats['drop_reasons']['cardint_siege_no_side'] += 1
                break
            _dreg = rr.sanitize(str(_pct.get('defender_region') or ''))
            _dstr = 'SHADOW' if _dsd == Side.SHADOW else 'FREE_PEOPLES'
            if not _card_siege_withdrawal_gesture(batch, _dreg, _dstr):
                stats['drop_reasons']['cardint_siege_no_gesture'] += 1
                break
            try:
                la_s = ai.get_legal_actions(s2, _dsd)
            except Exception:
                stats['drop_reasons']['cardint_siege_legal_exc'] += 1
                break
            h_s = _match_index(la_s, lambda a: a.action_type ==
                               'retreat_into_siege')
            if h_s is None:
                stats['drop_reasons']['cardint_siege_unmatched'] += 1
                break
            if len(la_s) >= 2:
                stats['kinds_emit']['defender_siege_choice'] += 1
                out.append(_emit_sample(s2, _dsd, la_s, h_s,
                                        'defender_siege_choice',
                                        batch['turn']))
            try:
                s2 = ai.apply_action(s2, la_s[h_s])
            except Exception:
                stats['drop_reasons']['cardint_siege_apply_exc'] += 1
                break
            continue
        pco = getattr(s2, 'pending_card_option', None)
        if pco and any('crebain_discard' in choice for choice in pco.get('options', [])):
            from rendered_hunt_observation import mark as mark_hunt_observation
            mark_hunt_observation(s2, int(batch['start_line']), 'before_draw')
        if pco is None:
            if getattr(s2, 'pending_card_followup', None) is not None:
                stats['detected']['cardint_followup'] += 1
                menu = ai.get_legal_actions(s2, side)
                if followup_plan is not None:
                    from rendered_card_followup import match_move
                    try:
                        if followup_cursor < len(followup_plan['moves']):
                            chosen = match_move(s2, side, menu, followup_plan['moves'][followup_cursor])
                            followup_cursor += 1
                        else:
                            choices = [i for i, action in enumerate(menu) if action.action_type == 'followup_done']
                            if len(choices) != 1:
                                raise FollowupUnproven('no_unique_done')
                            chosen = choices[0]
                        sample = (_emit_sample(s2, side, menu, chosen, 'card_followup', batch['turn'])
                                  if len(menu) >= 2 else None)
                        s2 = ai.apply_action(s2, menu[chosen])
                    except (ValueError, RuntimeError) as error:
                        stats['drop_reasons']['cardint_followup:' + str(error)] += 1
                        return []
                    if sample is not None:
                        stats['kinds_emit']['card_followup'] += 1
                        stats['kinds_emit']['card_followup_' + menu[chosen].action_type] += 1
                        out.append(sample)
                    if s2.pending_card_followup is None and followup_cursor < len(followup_plan['moves']):
                        stats['drop_reasons']['cardint_followup:unconsumed_movement'] += 1
                        return []
                    continue
                chosen = _card_followup_attack_index(s2, side, menu, batch)
                if chosen is None:
                    stats['drop_reasons']['cardint_followup_unproven'] += 1
                else:
                    stats['kinds_emit']['card_followup'] += 1
                    out.append(_emit_sample(s2, side, menu, chosen,
                                            'card_followup', batch['turn']))
                # The combat emitter owns the subsequent battle choices.
            break
        if pco.get('card_casualty'):
            region = rr.sanitize(str(pco.get('cas_region') or ''))
            owner = pco.get('cas_side')
            owner_name = 'SHADOW' if owner == Side.SHADOW else 'FREE_PEOPLES'
            try:
                events = _casualty_gesture_operations(batch, owner_name,
                    {region.removesuffix(' stronghold')}, batch.get('start_line', 0),
                    batch.get('end_line', 10 ** 9) + 1)
                _rnd9.randint = _rint9
                after, allocations = _drive_card_casualty_gestures(s2, events)
            except (ValueError, RuntimeError) as error:
                stats['drop_reasons']['cardcas_operation:' + str(error)] += 1
                break
            finally:
                _rnd9.randint = _ori9
            if _rng_failed9:
                return []
            for before, owner, menu, chosen in allocations:
                stats['detected']['cardint_card_casualty'] += 1
                stats['kinds_emit']['card_option'] += 1
                stats['kinds_emit']['cardopt_card_casualty'] += 1
                _pend_note(stats, 'card_casualty', 'open')
                _pend_note(stats, 'card_casualty', 'emit')
                with pre_hunt_menu(before, batch, cid, _pre_hunt_pool9):
                    out.append(_emit_sample(before, owner, menu, chosen,
                                            'card_option', batch['turn']))
            if not allocations:
                stats['detected']['cardint_card_casualty_forced'] += 1
            s2, first = after, False
            continue
        # spec 1.216 NON-DECISIONS: never emit an interior choice for these
        # even if the (pre-fix) engine still defers one.
        if str(pco.get('card_name', '')).lower() in _NONDECISION_CARDS:
            stats['detected']['cardint_nondecision'] += 1
            break
        decider = pco.get('decider') or pco.get('side') or (
            'sh' if side == Side.SHADOW else 'fp')
        dec_side = Side.SHADOW if decider == 'sh' else Side.FREE_PEOPLES
        try:
            opts = ai.get_legal_actions(s2, dec_side)
        except Exception:
            stats['drop_reasons']['cardint_legal_exc'] += 1
            break
        copts = [(i, a) for i, a in enumerate(opts)
                 if a.action_type == 'card_option']
        if not copts:
            break
        _fam9i = ('card_casualty' if pco.get('card_casualty')
                  else ('table_kill_payment'
                        if pco.get('table_kill_payment')
                        else 'card_option'))
        from ai_policies import project_ai_actions
        fixed = project_ai_actions(s2, dec_side, opts).forced_reason
        if fixed:
            stats['detected']['fixed_policy:' + fixed] += 1
            _pend_note(stats, _fam9i, 'auto')
        elif len(copts) >= 2:
            _pend_note(stats, _fam9i, 'open')
        if followup_plan is not None and str(pco.get('card_name', '')).lower() == 'the black captain commands':
            from rendered_card_followup import match_bcc_mode
            try:
                midx, why = match_bcc_mode(s2, copts, followup_plan), None
            except FollowupUnproven as error:
                stats['drop_reasons']['cardint_followup:' + str(error)] += 1
                return []
        else:
            midx, why = _match_interior(pco, copts, ctx, batch, stats, first, state=s2)
        first = False
        if midx is None:
            if why:
                stats['drop_reasons'][why] += 1
            if companion_plan is not None:
                return []
            break
        if len(copts) >= 2 and not fixed:
            fam_e = _card_family(opts[midx].params.get('choices', {}) or {})
            stats['kinds_emit']['card_option'] += 1
            stats['kinds_emit']['cardopt_' + fam_e] += 1
            _pend_note(stats, _fam9i, 'emit')
            with pre_hunt_menu(s2, batch, cid, _pre_hunt_pool9):
                out.append(_emit_sample(s2, dec_side, opts, midx, 'card_option',
                                        batch['turn']))
        if str(pco.get('card_name', '')).lower() == 'there is another way':
            # This emitter owns the mode before the Hunt. Do not consume a
            # speculative tile to fabricate subsequent Hunt decisions.
            break
        _rnd9.randint = _rint9      # chained dispatch may roll (SoM/DS)
        try:
            s2 = (_naz_hunt9.apply(s2, opts[midx]) if _naz_hunt9 is not None
                  else ai.apply_action(s2, opts[midx]))
        except Exception as e:
            stats['drop_reasons']['cardint_apply2_exc:%s' % str(e)[:12]] += 1
            break
        finally:
            _rnd9.randint = _ori9
    if _rng_failed9:
        stats['drop_reasons']['cardint_rng_unproven'] += 1
        return []
    if companion_plan is not None and ctx['_companion_card_cursor'] != len(companion_plan):
        stats['drop_reasons']['cardint_companion_chain:unconsumed_movement'] += 1
        return []
    # T9-14 (Cirdan's Ships): a play that overstacks a BESIEGED garrison
    # opens a siege_trim in the engine - but the recon keeps the garrison
    # under the stronghold BOX key, so the engine counts it under 5 and
    # the pending never opens.  Detect the trim GESTURE (killed-then-
    # returned pieces after the play) and build the pending on the box
    # army key directly.
    out.extend(_trim_probe(batch, s2, side, stats, brute=False))
    return out


def _trim_probe(batch, s2, side, stats, brute):
    """T9-14 helper: emit the overstack siege_trim on state s2.  With
    brute=True (pruned-play path: the engine never applied the play) the
    play's reinf->besieged-region musters are brute-applied first so the
    trim menu holds the true garrison."""
    import ai
    import rendered_replay as rr
    from game_state import Side
    out = []
    try:
        _bes9 = sorted(set(rr.sanitize(x) for x in
                       (batch['truth_before'].get('besieged') or [])))
        for _reg9 in _bes9:
            # the trim gesture: own units KILLED in the besieged
            # region/box, then corrected to REINFORCEMENTS via a
            # casualties->reinf move ('those actually go to
            # reinforcements') - _window_trim's region-zone pattern
            # misses the box kill + off-board correction
            import game_state as _gs9t
            eb9 = _rb9 = 0
            _kill_ok = False
            for r9 in batch['records']:
                if r9.get('op') == 'kill':
                    zk9 = rr.zone_kind(str(r9.get('zone', '') or ''))
                    if zk9[0] in ('region', 'stronghold') \
                            and zk9[1] == _reg9:
                        _kill_ok = True
                elif r9.get('op') == 'move':
                    fk9 = rr.zone_kind(r9.get('from', ''))
                    tk9 = rr.zone_kind(r9.get('to', ''))
                    if fk9[0] != 'cas' or tk9[0] != 'reinf':
                        continue
                    for t9 in _group_tokens(r9.get('obj', '')):
                        p9 = rr.parse_piece(t9)
                        if p9 and p9[0] == 'unit' and \
                                ai._side_of_nation(
                                    _gs9t.Nation[p9[1]]) == side:
                            if p9[2] == 'ELITE':
                                eb9 += p9[3]
                            else:
                                _rb9 += p9[3]
            if not _kill_ok or (eb9 + _rb9) <= 0:
                continue
            _key9 = None
            for _k8 in s2.armies:
                if rr.sanitize(_k8).startswith(_reg9) and any(
                        ai._side_of_nation(u.nation) == side
                        for u in s2.armies[_k8].units):
                    _key9 = _k8
                    break
            if _key9 is None:
                continue
            if brute:
                # pruned-play path: the engine never executed the
                # muster - apply the reinf->region gestures so the trim
                # menu enumerates the TRUE (overstacked) garrison
                for r9 in batch['records']:
                    if r9.get('op') != 'move':
                        continue
                    fk9 = rr.zone_kind(r9.get('from', ''))
                    tk9 = rr.zone_kind(r9.get('to', ''))
                    if fk9[0] != 'reinf' or tk9[0] not in (
                            'region', 'stronghold') or tk9[1] != _reg9:
                        continue
                    for t9 in _group_tokens(r9.get('obj', '')):
                        p9 = rr.parse_piece(t9)
                        if p9 and p9[0] == 'unit' and \
                                ai._side_of_nation(
                                    _gs9t.Nation[p9[1]]) == side:
                            for _ in range(p9[3]):
                                s2.armies[_key9].units.append(
                                    _gs9t.Unit(
                                        _gs9t.Nation[p9[1]],
                                        _gs9t.UnitKind.ELITE
                                        if p9[2] == 'ELITE' else
                                        _gs9t.UnitKind.REGULAR))
            s2.pending_combat = {
                'phase': 'siege_trim', 'trim_region': _key9,
                'attacker_region': _reg9, 'defender_region': _key9,
                'deciding_side': side,
                'attacker_side': (Side.SHADOW if side ==
                                  Side.FREE_PEOPLES else
                                  Side.FREE_PEOPLES),
                'defender_side': side, 'is_siege': True}
            la_t = [a for a in ai.get_legal_actions(s2, side)
                    if a.action_type == 'siege_trim']
            h_t = _match_index(la_t, lambda a: int(
                a.params.get('elites_back', -1)) == eb9)
            if h_t is not None and len(la_t) >= 2:
                stats['kinds_emit']['siege_trim'] += 1
                out.append(_emit_sample(s2, side, la_t, h_t,
                                        'siege_trim', batch['turn']))
            else:
                stats['drop_reasons']['cardint_trim_unmatched'] += 1
            s2.pending_combat = None
            break
    except Exception:
        pass
    return out


_FUZZY_CARDS = {'the spirit of mordor', "faramir's rangers",
                'help unlooked for', 'dreadful spells'}


def _dead_men_group_evidence(ctx, batch, available_companions):
    """Closed literal Companion movement, including those deliberately left behind."""
    import rendered_replay as rr
    play_line = _play_line(ctx, batch)
    completed = [int(record.get('line', 0)) for record in batch['records']
                 if record.get('op') == 'kill'
                 and any(str(piece).split('/')[0].strip().lower() == 'dead men of dunharrow'
                         for piece in record.get('pieces', []))
                 and int(record.get('line', 0)) > play_line]
    if play_line < 0:
        return None, 'cardint_dmd_group_incomplete'
    end = min(completed) if completed else batch['end_line'] + 1
    names, route = [], None
    for record in batch['records']:
        if record.get('op') != 'move' or not play_line < int(record.get('line', 0)) < end:
            continue
        for token in _group_tokens(record.get('obj', '')):
            piece = rr.parse_piece(token)
            if not piece or piece[0] != 'named':
                continue
            name = _comp_norm(piece[1])
            if name not in ('Gandalf the Grey', 'Strider', 'Legolas', 'Gimli', 'Boromir',
                            'Meriadoc', 'Peregrin', 'Aragorn', 'Gandalf the White'):
                continue
            origin, target = rr.zone_kind(record.get('from', '')), rr.zone_kind(record.get('to', ''))
            if (origin[0] != 'region' or target[0] != 'region' or origin == target
                    or name in names or (route is not None and route != (origin[1], target[1]))):
                return None, 'cardint_dmd_group_movement_ambiguous'
            route = origin[1], target[1]
            names.append(name)
    if route is None or len(set(names) & {'Strider', 'Aragorn'}) != 1:
        return None, 'cardint_dmd_group_no_required_arrival'
    # Explicit movement of the whole eligible group proves the group itself.
    # Inferring a left-behind Companion from silence needs the card to close.
    if not completed and set(names) != set(available_companions):
        return None, 'cardint_dmd_group_incomplete'
    origin, target = route
    delta = ctx['delta']
    for name in names:
        for field, region in (('removed', origin), ('added', target)):
            found = sum(count for key, count in delta[field].get(region, {}).items()
                        if key[0] == 'NAMED' and _comp_norm(key[1]) == name)
            if found != 1:
                return None, 'cardint_dmd_group_delta_mismatch'
    return (target, frozenset(names)), None


def _dead_men_recruit_evidence(target, ctx, batch):
    """Exact completed-card recruitment evidence, including an explicit zero.

    A net board count alone cannot identify a choice: it can include a
    correction or an incomplete record. Require the named companion's arrival,
    reinforcement-to-target placements, and disposal of this card after those
    effects. Returns/corrections without an unambiguous count are dropped.
    """
    import rendered_replay as rr
    delta = ctx['delta']
    added = delta['added'].get(target, Counter())
    if not any(k[0] == 'NAMED' and _comp_norm(k[1]) in ('Strider', 'Aragorn')
               for k in added):
        return None, 'cardint_dmd_recruit_no_arrival'
    play_line = _play_line(ctx, batch)
    if play_line < 0:
        return None, 'cardint_dmd_recruit_no_play'
    count = 0
    last_effect = play_line
    completed = []
    for record in batch['records']:
        line = int(record.get('line', 0) or 0)
        if line <= play_line:
            continue
        if record.get('op') == 'kill':
            if any(str(piece).split('/')[0].strip().lower()
                   == 'dead men of dunharrow'
                   for piece in record.get('pieces', [])):
                completed.append(line)
        if record.get('op') != 'move':
            continue
        frm, to = rr.zone_kind(record.get('from', '')), rr.zone_kind(record.get('to', ''))
        for token in _group_tokens(record.get('obj', '')):
            piece = rr.parse_piece(token)
            if not piece:
                continue
            if (piece[0] == 'named' and piece[1] in ('Strider', 'Aragorn')
                    and to == ('region', target)):
                last_effect = max(last_effect, line)
            if piece[:3] != ('unit', 'GONDOR', 'REGULAR'):
                continue
            if frm == ('region', target):
                return None, 'cardint_dmd_recruit_return_ambiguous'
            if to == ('region', target):
                if frm[0] != 'reinf':
                    return None, 'cardint_dmd_recruit_not_reinforcement'
                count += int(piece[3])
                last_effect = max(last_effect, line)
    if not any(line > last_effect for line in completed):
        return None, 'cardint_dmd_recruit_incomplete'
    if not 0 <= count <= 3 or added.get(('GONDOR', 'REGULAR'), 0) != count:
        return None, 'cardint_dmd_recruit_delta_mismatch'
    return count, None


def _single_gondor_recruit_evidence(ctx, batch, *, name, target, family):
    """Identify one printed Gondor unit choice from a completed named play.

    Require an explicit FP reinforcement placement and matching board delta.
    Returns, overstack trims and ambiguous combined pools need their own
    attribution; they cannot silently become the historical Elite default.
    This identifies the unit selection, not every legality clause in a game.
    """
    import rendered_replay as rr
    def reject(reason):
        return None, 'cardint_' + family + '_' + reason
    plays = _card_play_titles(batch)
    if (not plays or str(plays[0][1]).split('/')[0].strip().lower() != name):
        return reject('no_play')
    if len(plays) != 1:
        return reject('multiple_plays')
    play_line = int(plays[0][0])
    zones = {('region', target), ('stronghold', target)}
    recruits = Counter()
    completed = []
    last_effect = play_line
    for record in batch['records']:
        line = int(record.get('line', 0) or 0)
        if line <= play_line:
            continue
        if record.get('op') == 'kill':
            if any(str(piece).split('/')[0].strip().lower() == name
                   for piece in record.get('pieces', [])):
                completed.append(line)
        if record.get('op') != 'move':
            continue
        frm = rr.zone_kind(record.get('from', ''))
        to = rr.zone_kind(record.get('to', ''))
        for token in _group_tokens(record.get('obj', '')):
            piece = rr.parse_piece(token)
            if not piece or piece[:2] not in (('unit', 'GONDOR'), ('leader', 'GONDOR')):
                continue
            if frm in zones and to not in zones:
                return reject('return_ambiguous')
            if to not in zones or frm in zones:
                continue
            if record.get('from', '').strip().lower() != 'fp reinforcements':
                return reject('not_fp_reinforcement')
            last_effect = max(last_effect, line)
            if piece[0] == 'unit':
                recruits[piece[2]] += int(piece[3])
    if not any(line > last_effect for line in completed):
        return reject('incomplete')
    if sum(recruits.values()) != 1 or set(recruits) - {'REGULAR', 'ELITE'}:
        return reject('recruit_count')
    observed = Counter()
    for region in (target, target + ' stronghold'):
        for kind in ('REGULAR', 'ELITE'):
            key = ('GONDOR', kind)
            observed[kind] += ctx['delta']['added'].get(region, {}).get(key, 0)
            observed[kind] -= ctx['delta']['removed'].get(region, {}).get(key, 0)
    if observed != recruits:
        return reject('delta_mismatch')
    return next(iter(recruits)).lower(), None


def _guards_recruit_evidence(ctx, batch):
    return _single_gondor_recruit_evidence(
        ctx, batch, name='guards of the citadel', target='minas tirith', family='guards')


def _recruit_allocation_match(pco, copts, ctx, batch, state):
    """Match scarce recruitment by exact completed reinforcement placements."""
    import ai
    import rendered_replay as rr
    from card_options import _free_for_side
    from game_state import Nation, Side
    from shadow_recruit_allocations import allocation_spec
    name = str(pco.get('card_name', ''))
    label = name.lower()

    def reject(reason):
        return None, 'cardint_recruit_allocation_' + reason

    if state is None or allocation_spec(state, name) is None:
        return reject('no_state')
    plays = _card_play_titles(batch)
    if len(plays) != 1 or str(plays[0][1]).split('/')[0].strip().lower() != label:
        return reject('no_single_play')
    play_line = int(plays[0][0])
    placed, completed = Counter(), []
    last_effect = play_line
    for record in batch['records']:
        line = int(record.get('line', 0) or 0)
        if line <= play_line:
            continue
        if record.get('op') == 'kill':
            if any(str(piece).split('/')[0].strip().lower() == label
                   for piece in record.get('pieces', [])):
                completed.append(line)
        if record.get('op') != 'move':
            continue
        frm = rr.zone_kind(record.get('from', ''))
        to = rr.zone_kind(record.get('to', ''))
        for token in _group_tokens(record.get('obj', '')):
            piece = rr.parse_piece(token)
            if not piece or piece[0] != 'unit':
                continue
            if frm[0] in ('region', 'stronghold'):
                return reject('return_or_movement')
            if to[0] not in ('region', 'stronghold'):
                continue
            if frm[0] != 'reinf' or str(record.get('from', '')).lower().startswith('fp '):
                return reject('not_shadow_reinforcement')
            placed[(to[1], piece[1], piece[2])] += int(piece[3])
            last_effect = max(last_effect, line)
    if not any(line > last_effect for line in completed):
        return reject('incomplete')
    added = Counter({(region, nation, kind): count
                     for region, units in ctx['delta']['added'].items()
                     for (nation, kind), count in units.items() if kind in ('REGULAR', 'ELITE')})
    removed = any(kind in ('REGULAR', 'ELITE')
                  for units in ctx['delta']['removed'].values() for nation, kind in units)
    if placed != added or removed:
        return reject('delta_mismatch')
    matches = []
    for index, action in copts:
        choice = action.params.get('choices') or {}
        if 'recruit_counts' not in choice:
            continue
        key, nation = allocation_spec(state, name, choice)[:2]
        expected = Counter({(rr.sanitize(str(region)), nation.name, 'REGULAR'): count
                            for region, count in zip(choice[key], choice['recruit_counts'])})
        extra_region, extra_nation, elites = None, nation, 0
        if name == 'Monsters Roused':
            extra_region, extra_nation = 'trollshaws', Nation.SAURON
            elites = min(1, ai.figures_available(state, extra_nation, 'elite'))
        elif name == 'A New Power is Rising':
            extra_region = 'orthanc'
            elites = min(2, ai.figures_available(state, nation, 'elite'))
            requested = {'2regular': 0, 'mixed': 1, '2elite': 2}.get(choice.get('unit_combo'))
            if requested is not None:
                elites = min(requested, ai.figures_available(state, nation, 'elite'))
        if extra_region and elites and _free_for_side(state, extra_region, Side.SHADOW):
            expected[(extra_region, extra_nation.name, 'ELITE')] = elites
        if expected == placed:
            matches.append(index)
    return (matches[0], None) if len(matches) == 1 else reject('unmatched' if not matches else 'ambiguous')


def _match_interior(pco, copts, ctx, batch, stats, first, *, state=None):
    """Match ONE pending_card_option menu against the batch evidence.
    copts = [(opts_index, action)].  Returns (opts_index or None,
    drop_reason or None)."""
    import rendered_replay as rr
    delta = ctx['delta']
    card_lname = str(pco.get('card_name', '')).lower()
    choices0 = copts[0][1].params.get('choices', {}) or {}
    if 'recruit_counts' in choices0:
        return _recruit_allocation_match(pco, copts, ctx, batch, state)
    if card_lname == 'there is another way':
        from rendered_gollum_mode import match
        return match(copts, batch)
    if card_lname == "cirdan's ships":
        from rendered_event_selections import match_cirdan
        return match_cirdan(copts, batch)
    if card_lname == 'stormcrow' and 'eliminate' in choices0:
        from rendered_event_selections import match_stormcrow_eliminate
        return match_stormcrow_eliminate(copts, batch)
    if card_lname == 'the lidless eye':
        from rendered_event_selections import match_lidless
        return match_lidless(copts, batch)
    if card_lname == "denethor's folly" and 'nation' in choices0:
        from rendered_event_selections import match_denethor
        return match_denethor(copts, batch)
    if '_cargo_index' in choices0:
        from rendered_staged_movement import match_cargo
        return match_cargo(copts, ctx)
    if card_lname in ('corsairs of umbar', 'shadows gather', 'the shadow lengthens', 'the shadow is moving') and ctx.get('_staged_army_plan') is not None and 'overstack_remove' not in choices0:
        from rendered_staged_movement import match_plan
        return match_plan(copts, ctx)
    if 'crebain_discard' in choices0:
        from rendered_reactive_cards import crebain_choice, Unproven
        try:
            use, _line = crebain_choice(batch)
        except Unproven as error:
            return None, 'cardint_' + str(error)
        matches = [i for i, a in copts if
                   (a.params.get('choices') or {}).get('crebain_discard') is use]
        return (matches[0], None) if len(matches) == 1 else (None, 'cardint_crebain_unmatched')
    if card_lname.replace(',', '') in ('the fighting uruk-hai', 'grond hammer of the underworld'):
        from rendered_siege_event import match
        return match(copts, batch)
    if card_lname == 'help unlooked for':
        from rendered_event_battle import match
        return match(copts, batch)
    if card_lname == 'guards of the citadel':
        unit_type, why = _guards_recruit_evidence(ctx, batch)
        if unit_type is None:
            return None, why
        matches = [i for i, a in copts
                   if (a.params.get('choices') or {}).get('unit_type') == unit_type]
        return (matches[0], None) if len(matches) == 1 else (
            None, 'cardint_guards_unmatched')
    if card_lname == 'house of the stewards':
        target = _opt_region(choices0, 'boromir_region')
        if target is None:
            return None, 'cardint_stewards_no_region'
        unit_type, why = _single_gondor_recruit_evidence(
            ctx, batch, name=card_lname, target=target, family='stewards')
        if unit_type is None:
            return None, why
        matches = [i for i, a in copts
                   if (a.params.get('choices') or {}).get('unit_type') == unit_type
                   and _opt_region(a.params.get('choices') or {}, 'boromir_region') == target]
        return (matches[0], None) if len(matches) == 1 else (
            None, 'cardint_stewards_unmatched')
    if pco.get('card_casualty'):
        # Aggregate board deltas cannot distinguish whole-Elite loss from
        # replacement-and-removal, nor establish intermediate nation choices.
        # The interior driver validates the entire ordered sequence instead.
        return None, 'cardcas_requires_ordered_replay'
    # Dead Men's target, opponent retreat and final optional recruitment are
    # distinct decisions. Generic "some figure changed in this region"
    # matching confuses its target with the Shadow's retreat destination.
    if card_lname == 'dead men of dunharrow':
        if choices0.get('_dmd_recruit_only'):
            target = _opt_region(choices0, 'target_region')
            count, why = _dead_men_recruit_evidence(target, ctx, batch)
            if count is None:
                return None, why
            matches = [i for i, a in copts if
                       (a.params.get('choices') or {}).get('recruit_count') == count]
            return (matches[0], None) if len(matches) == 1 else (
                None, 'cardint_dmd_recruit_unmatched')
        if choices0.get('_dmd_target'):
            target = _opt_region(choices0, '_dmd_target')
            destinations = set()
            for record in batch['records']:
                if record.get('op') != 'move':
                    continue
                frm = rr.zone_kind(record.get('from', ''))
                to = rr.zone_kind(record.get('to', ''))
                if frm != ('region', target) or to[0] != 'region':
                    continue
                if any((p := rr.parse_piece(token)) and p[0] == 'unit'
                       and p[1] in _SHADOW_NATS
                       for token in _group_tokens(record.get('obj', ''))):
                    destinations.add(to[1])
            matches = [i for i, a in copts if
                       _opt_region(a.params.get('choices') or {}, 'destination')
                       in destinations]
        else:
            available = {_comp_norm(name) for _, action in copts for name in
                         (action.params.get('choices') or {}).get('companions', [])}
            evidence, why = _dead_men_group_evidence(ctx, batch, available)
            if evidence is None:
                return None, why
            target, group = evidence
            matches = [i for i, a in copts
                       if _opt_region(a.params.get('choices') or {}, 'target_region') == target
                       and frozenset(_comp_norm(name) for name in
                                     (a.params.get('choices') or {}).get('companions', [])) == group]
        return (matches[0], None) if len(matches) == 1 else (
            None, 'cardint_dmd_movement_unmatched')
    # ---- per-figure nazgul flight chain (spec 1.216 item 3) -----------
    # BCC / Ringwraiths Abroad / Nazgul Search / Nazgul Strike: the menu is
    # one option per (from,to) single-figure hop (+WK variant, always
    # distinct) + the '_naz_done' terminator.  Rendered evidence: the ordered
    # per-figure flight gestures; the terminator matches when they exhaust.
    if card_lname == '__nazgul_moves__' or \
            _card_family(choices0) == 'nazgul_chain':
        stats['detected']['cardint_nazgul_chain' if first
                          else 'nazchain_link'] += 1
        hops, hi = ctx['hops'], ctx['hop_i']
        if hi < len(hops):
            frm, to, wk = hops[hi]
            for i, a in copts:
                ch = a.params.get('choices', {}) or {}
                if ch.get('_naz_done'):
                    continue
                if rr.sanitize(str(ch.get('from_region', ''))) != frm or \
                        rr.sanitize(str(ch.get('to_region', ''))) != to:
                    continue
                if bool(ch.get('include_wk')) != bool(wk):
                    continue
                ctx['hop_i'] = hi + 1
                return i, None
            return None, 'nazchain_hop_unmatched'
        for i, a in copts:
            if (a.params.get('choices', {}) or {}).get('_naz_done'):
                return i, None
        return None, 'nazchain_no_term'
    # ---- Ents Awake free chain (spec 1.216 META) -----------------------
    # 'cards played in a chain WITHOUT extra dice': the chained play is the
    # batch's NEXT hand-exit to a play area; no further play = chain done.
    if any(('ents_chain_play' in (a.params.get('choices', {}) or {})
            or '_ents_chain_done' in (a.params.get('choices', {}) or {}))
           for _i, a in copts):
        stats['detected']['cardint_ents_chain'] += 1
        plays, pi = ctx['plays'], ctx['play_i']
        cid2 = _title_to_cid(plays[pi][1]) if pi < len(plays) else None
        if cid2 is not None:
            for i, a in copts:
                ch = a.params.get('choices', {}) or {}
                if str(ch.get('ents_chain_play')) == str(cid2):
                    ctx['play_i'] = pi + 1
                    return i, None
            return None, 'ents_chain_unmatched'
        for i, a in copts:
            if (a.params.get('choices', {}) or {}).get('_ents_chain_done'):
                return i, None
        return None, 'ents_chain_no_term'
    # Book/FFF transfer whole named groups. An unmatched group cannot be
    # relabelled as completion; the literal plan was closure-checked first.
    if any('move_group' in (action.params.get('choices') or {})
           or '_ffm_done' in (action.params.get('choices') or {})
           for _, action in copts):
        if '_companion_card_plan' not in ctx:
            return None, 'cardint_companion_chain:missing_literal_plan'
        from rendered_companion_chain import match
        return match(copts, ctx)
    # Match the full source/composition choice, including a closed zero move.
    if any('rage_split' in (a.params.get('choices') or {}) for _, a in copts):
        from rendered_rage import match
        return match(copts, batch)
    fam = _card_family(choices0)
    if first:
        stats['detected']['cardint_' + fam] += 1
    # ---- W1-017: the Palantir of Orthanc deck grant (ruling 1.208m: a
    # learned WHICH-DECK decision, "the parser derives the label from
    # which deck the human drew").  Evidence = the Shadow deck draw the
    # human made after the play; no Shadow event card draws Event cards
    # by its own text (card_list.csv), so the ONE post-play deck draw
    # of the batch is the grant.  0 draws / 2+ draws = named drops.
    if fam == 'palantir_deck':
        deck9, why9 = _palantir_deck_evidence(ctx, batch)
        if deck9 is None:
            return None, why9
        m9 = [i for i, a in copts
              if str((a.params.get('choices') or {}).get(
                  'palantir_deck', '')).lower() == deck9]
        return (m9[0], None) if len(m9) == 1 else (
            None, 'cardint_palantir_unmatched')
    # ---- W1-017: __overstack__ return (event_effects.resolve_overstacks:
    # one (nation, kind) unit of the overstacked garrison goes back to
    # reinforcements per pending, chained).  Evidence = the human's
    # '<unit> from <stronghold> to <Side> Reinforcements' move records;
    # each link consumes one unit of the gesture (2023-09-18 L974
    # '[2]Gondor Regular' = two links).
    if fam == 'overstack':
        overstack_batch = batch
        if '_corsairs_landing_pc' in choices0:
            from rendered_staged_movement import corsairs_trim_window, Unproven
            try:
                overstack_batch = corsairs_trim_window(batch, choices0['overstack_remove']['region'])
            except Unproven as error:
                return None, 'cardint_' + str(error)
        gest9 = _overstack_returns(overstack_batch)
        used9 = ctx.setdefault('ovs_used', Counter())
        m9 = []
        for i, a in copts:
            spec9 = (a.params.get('choices') or {}).get(
                'overstack_remove') or {}
            key9 = (rr.sanitize(str(spec9.get('region', ''))),
                    _nat_key(spec9.get('nation')),
                    str(spec9.get('kind', '')).lower())
            if gest9.get(key9, 0) - used9.get(key9, 0) >= 1:
                m9.append((i, key9))
        if not gest9:
            return None, 'cardint_overstack_no_gesture'
        if len(m9) != 1:
            return None, ('cardint_overstack_unmatched' if not m9
                          else 'cardint_overstack_ambiguous')
        used9[m9[0][1]] += 1
        return m9[0][0], None
    # ---- W1-017: Mithril Coat and Sting redraw (FP table card, card
    # text: "After the Shadow player draws a Hunt tile, you may discard
    # Mithril Coat and Sting to force him to replace the tile").
    # Evidence = the hunt-tile draws of the batch after the play: ONE
    # tile and the table card still on the table = kept (False); a
    # SECOND tile draw or the table card leaving BoardN = redraw (True).
    if fam == 'mithril_redraw':
        red9, why9 = _mithril_redraw_evidence(ctx, batch)
        if red9 is None:
            return None, why9
        m9 = [i for i, a in copts
              if bool((a.params.get('choices') or {}).get(
                  'mithril_redraw')) == red9]
        return (m9[0], None) if len(m9) == 1 else (
            None, 'cardint_mithril_unmatched')
    # ---- recruit-region gesture primary: the reinf->region move NAMES
    # the recruit region; the raw PRE/POST delta alone is ambiguous when
    # a join-move changed a second region (Rage T5-22) -----------------
    if fam == 'region':
        _rds9 = set()
        for r9 in batch['records']:
            if r9.get('op') != 'move':
                continue
            if rr.zone_kind(r9.get('from', ''))[0] == 'reinf' and \
                    rr.zone_kind(r9.get('to', ''))[0] in (
                        'region', 'stronghold'):
                _rds9.add(rr.zone_kind(r9.get('to', ''))[1])
        if len(_rds9) == 1:
            _rd9 = next(iter(_rds9))
            m9 = [i for i, a in copts
                  if rr.sanitize(str(
                      (a.params.get('choices') or {}).get(
                          'recruit_region',
                          (a.params.get('choices') or {}).get(
                              'region', '')))) == _rd9]
            if len(m9) == 1:
                return m9[0], None
    # ---- fuzzy-region battle cards (spec 1.216 item 4) -----------------
    if card_lname in _FUZZY_CARDS or fam == 'dreadful_spells':
        return _fuzzy_interior_match(copts, ctx, card_lname, batch, stats)
    if fam == 'companion_action':
        # Gwaihir / We Prove Swifter ("move or separate from the
        # Fellowship one Companion or one group of Companions"): the
        # gesture is read BY NAME after the play line (W1-008, replaces
        # the _sep_moves-only read that saw neither a group token leaving
        # the Fellowship nor any board move - 7 opened / 0 emitted):
        #   sep arm : 1 name -> sep_<name>; 2+ names to ONE region ->
        #             the 1.149c 'group' option (GAME19 T2)
        #   move arm: 1 name -> move_<name>; 2+ co-located names to ONE
        #             region -> 'move_group'
        # Two destinations or both arms in one play = not one option:
        # drop, never a guess.
        _pl9 = [int(ln or 0) for ln, tt in (ctx.get('plays') or [])
                if tt == ctx.get('play_title')]
        arm9, moves9 = _companion_card_gesture(
            batch, min(_pl9) if _pl9 else 0)
        if arm9 is None:
            return None, 'cardint_deferred_' + fam
        _dests9 = set(moves9.values())
        if arm9 == 'mixed' or len(_dests9) != 1:
            return None, 'cardint_compact_ambiguous'
        _dst9 = next(iter(_dests9))
        _want9 = sorted(moves9)
        if len(_want9) >= 2:
            _act9 = 'group' if arm9 == 'sep' else 'move_group'
        else:
            _act9 = ('sep_' if arm9 == 'sep' else 'move_') + _want9[0]
        m9 = [i for i, a in copts
              if str((a.params.get('choices') or {}).get(
                  'companion_action', '')) == _act9
              and (len(_want9) < 2
                   or sorted((a.params.get('choices') or {})
                             .get('companions') or []) == _want9)
              and rr.sanitize(str((a.params.get('choices') or {})
                                  .get('destination', '')))
              == _dst9]
        if len(m9) == 1:
            return m9[0], None
        return None, 'cardint_compact_unmatched'
    # ---- dice_count (The Lidless Eye: dice added to the hunt) ----------
    if fam == 'dice_count':
        n = _lidless_dice_evidence(batch)
        if n is None:
            return None, 'cardint_dice_no_evidence'
        m = [i for i, a in copts if int((a.params.get('choices') or {})
                                        .get('dice_count', -1)) == n]
        return (m[0], None) if len(m) == 1 else (None,
                                                 'cardint_dice_unmatched')
    # ---- multi-army movement chain (TSiM / TSL / Shadows Gather): match
    # the NEXT collapsed drag + its exact moved composition against the
    # single-move menu; the terminator matches when the drags exhaust
    # (spec 1.216: teleport-chain hop attribution).
    if fam == 'teleport':
        return _teleport_chain_match(copts, ctx, batch, stats)
    # ---- generic attribution-principle matching (PRE/POST delta) -------
    matches = [i for i, a in copts
               if _option_matches(a.params.get('choices', {}) or {}, delta,
                                  pco.get('decider'), ctx)]
    if len(matches) == 1:
        return matches[0], None
    tag = 'unmatched' if not matches else 'ambiguous'
    return None, 'cardint_%s_%s' % (tag, fam)


_TELEPORT_TERM_KEYS = ('_tsim_done', '_tsl_done')


def _teleport_chain_match(copts, ctx, batch, stats):
    """One chained army-move menu: the next un-consumed drag (from,to) picks
    the option; among same-(from,to) split variants the EXACT moved
    composition decides (the bare option IS the whole-army move â€” the split
    grid excludes the full stack, so bare matches iff no split does)."""
    if ctx.get('_staged_army_plan') is not None:
        from rendered_staged_movement import match_plan
        return match_plan(copts, ctx)
    chains, ti = ctx['chains'], ctx['tm_i']
    if ti >= len(chains):
        for i, a in copts:
            ch = a.params.get('choices', {}) or {}
            if any(ch.get(k) for k in _TELEPORT_TERM_KEYS):
                return i, None
        return None, 'cardint_teleport_no_term'
    frm, to, leg1 = chains[ti]
    comp = _moved_group_composition(batch, frm, leg1)
    exact = bare = None
    for i, a in copts:
        ch = a.params.get('choices', {}) or {}
        if any(ch.get(k) for k in _TELEPORT_TERM_KEYS):
            continue
        if _opt_region(ch, 'from_region') != frm or \
                _opt_region(ch, 'to_region') != to:
            continue
        if ch.get('n_regular') is not None or ch.get('n_elite') is not None:
            if int(ch.get('n_regular') or 0) == comp['reg'] and \
                    int(ch.get('n_elite') or 0) == comp['elite'] and \
                    exact is None:
                exact = i
        elif bare is None:
            bare = i
    pick = exact if exact is not None else bare
    if pick is None:
        return None, 'cardint_teleport_hop_unmatched'
    ctx['tm_i'] = ti + 1
    return pick, None


def _fp_units_removed_at(delta, reg):
    c = delta['removed'].get(reg)
    return bool(c) and any(k[0] not in _SHADOW_NATS and
                           k[1] in ('REGULAR', 'ELITE') for k in c)


def _sh_units_removed_at(delta, reg):
    c = delta['removed'].get(reg)
    return bool(c) and any(k[0] in _SHADOW_NATS and
                           k[1] in ('REGULAR', 'ELITE') for k in c)


def _snap_nazgul_at(snap, reg):
    """Nazgul count at region INCLUDING the Witch-king (WK counts as 1
    nazgul for Dreadful Spells dice, spec 1.217/1.216)."""
    import rendered_replay as rr
    n = 0
    for k, (_units, leaders) in snap['armies'].items():
        if rr.sanitize(k) != reg:
            continue
        for ld in leaders:
            if ld[0] == 'NAZGUL':
                n += 1
            elif ld[0] == 'NAMED' and str(ld[1]).lower().startswith('witch'):
                n += 1
    return n


def _snap_fp_units_at(snap, reg):
    import rendered_replay as rr
    n = 0
    for k, (units, _l) in snap['armies'].items():
        if rr.sanitize(k) == reg:
            n += sum(1 for (nat, _ki) in units if nat not in _SHADOW_NATS)
    return n


def _lidless_dice_evidence(batch):
    """Dice ADDED to the hunt box by The Lidless Eye: the 'adds N eyes'
    rendered line, else the count of action-die moves into the hunt box."""
    import rendered_replay as rr
    for it in batch['rng']:
        if it.get('kind') == 'add_eye':
            return int(it.get('n', 0)) or None
    n = 0
    for r in batch['records']:
        if r.get('op') != 'move':
            continue
        if 'hunt' not in str(r.get('to', '')).lower():
            continue
        # W1-017: the dice may be dragged as ONE group record
        # ('group ([2]Shadow Action Die (H), Shadow Action Die (A))',
        # 2024-05-20 L803) - parse_piece on the whole obj saw no die
        for t in _group_tokens(str(r.get('obj', ''))):
            p = rr.parse_piece(t)
            if p and p[0] == 'die':
                n += p[3]
    return n or None


def _nat_key(v):
    """Engine nation KEY ('GONDOR', 'THE_NORTH', 'SOUTHRON_EASTERLINGS')
    for a Nation member, a Nation.value ('The North') or a rendered nation
    word ('S&E') - the one alphabet the overstack matcher compares in."""
    import rendered_replay as rr
    if v is None:
        return None
    if hasattr(v, 'name'):
        return v.name
    s = str(v).strip().lower()
    if s in rr.NATION_NAME:
        return rr.NATION_NAME[s]
    if 'southron' in s or 'easterling' in s:
        return 'SOUTHRON_EASTERLINGS'
    return s.upper().replace(' ', '_')


def _play_line(ctx, batch):
    """Line of the batch's die-bought play (the hand exit), -1 if none."""
    plays = ctx.get('plays') or _card_play_titles(batch)
    return int(plays[0][0]) if plays else -1


def _palantir_deck_evidence(ctx, batch):
    """('character' | 'strategy', None) for the Palantir of Orthanc grant
    = the ONE Shadow deck draw the human made after the play, inside the
    batch and before any turn marker; (None, drop_reason) otherwise."""
    pl = _play_line(ctx, batch)
    decks = []
    for r in batch['records']:
        if r.get('op') == 'turn':
            break                       # phase-1 draws are not the grant
        if r.get('op') != 'draw' or int(r.get('line', 0) or 0) <= pl:
            continue
        code = _DECK_CODE.get(str(r.get('what', '')).strip().lower())
        if code == '2006':
            decks.append('character')
        elif code == '2007':
            decks.append('strategy')
    if not decks:
        return None, 'cardint_palantir_no_draw'
    if len(decks) > 1:
        return None, 'cardint_palantir_ambiguous'
    return decks[0], None


def _overstack_returns(batch):
    """Counter{(region_san, NATION_KEY, 'regular'|'elite'): n} of the
    units the human sent from a region / stronghold box back to a
    reinforcements zone this batch (the __overstack__ return gesture)."""
    import rendered_replay as rr
    out = Counter()
    for r in batch['records']:
        if r.get('op') != 'move':
            continue
        fk = rr.zone_kind(r.get('from', ''))
        tk = rr.zone_kind(r.get('to', ''))
        if fk[0] not in ('region', 'stronghold') or tk[0] != 'reinf':
            continue
        for t in _group_tokens(str(r.get('obj', ''))):
            p = rr.parse_piece(t)
            if p and p[0] == 'unit':
                out[(fk[1], p[1], str(p[2]).lower())] += p[3]
    return out


def _mithril_redraw_evidence(ctx, batch):
    """(True | False, None) for the FP Mithril Coat and Sting decision on a
    card-drawn hunt tile: hunt-tile draws after the play (deck draws are
    not tiles) and the table card's exit from BoardN; (None, reason)
    when the batch shows no tile at all."""
    import rendered_replay as rr
    pl = _play_line(ctx, batch)
    tiles = 0
    left_table = False
    for r in batch['records']:
        if r.get('op') == 'turn':
            break
        ln = int(r.get('line', 0) or 0)
        if ln <= pl:
            continue
        if r.get('op') == 'draw':
            what = str(r.get('what', '')).strip()
            if what.lower() in _DECK_CODE:
                continue
            p = rr.parse_piece(what)
            if p and p[0] == 'tile':
                tiles += 1
            continue
        pieces = []
        if r.get('op') == 'move':
            if _BOARD_DEST_RE.match(_zone_norm(r.get('from', ''))):
                pieces = _group_tokens(str(r.get('obj', '')))
        elif r.get('op') == 'kill':
            if _BOARD_DEST_RE.match(_zone_norm(r.get('zone', ''))):
                pieces = list(r.get('pieces') or [])
        if any('mithril coat' in str(t).lower() for t in pieces):
            left_table = True
    if tiles == 0:
        return None, 'cardint_mithril_no_tile'
    if tiles >= 2 or left_table:
        return True, None
    return False, None


def _fuzzy_interior_match(copts, ctx, card_lname, batch, stats):
    """Confidence-scored region attribution for the fuzzy battle cards
    (spec 1.216 item 4): chat fuzzy-match + casualty location + dice count.
    casualty-location = +2 (strongest), chat naming = +1 per region, dice
    count = +1.  Emit iff the top option is unique with score >= 2 (or is
    the only option with score >= 1); low confidence = NAMED drop, never a
    guess."""
    delta = ctx['delta']
    chats = ctx['chats']
    tb = batch['truth_before']
    chunks = [r for r in batch['rng'] if r['kind'] == 'roll_num']
    dice_n = len(chunks[0]['ints']) if chunks else None
    tag = re.sub(r'[^a-z]+', '_', card_lname).strip('_')
    stats['detected']['fuzzy_' + tag] += 1
    scored = []
    for i, a in copts:
        ch = a.params.get('choices', {}) or {}
        sc = 0
        if card_lname == 'dreadful spells':
            treg = _opt_region(ch, 'target_region')
            farm = _opt_region(ch, 'target_fp_army')
            if farm and _fp_units_removed_at(delta, farm):
                sc += 2
            if treg and _chat_names_region(treg, chats):
                sc += 1
            if farm and farm != treg and _chat_names_region(farm, chats):
                sc += 1
            if dice_n is not None and treg and \
                    min(5, _snap_nazgul_at(tb, treg)) == dice_n:
                sc += 1
        elif card_lname == 'help unlooked for':
            shr = _opt_region(ch, 'stronghold')
            af = _opt_region(ch, 'attack_from')
            if shr and _sh_units_removed_at(delta, shr):
                sc += 2
            if shr and _chat_names_region(shr, chats):
                sc += 1
            if af and _chat_names_region(af, chats):
                sc += 1
            if dice_n is not None and af and \
                    min(5, _snap_fp_units_at(tb, af)) == dice_n:
                sc += 1
        else:                   # Spirit of Mordor / Faramir's Rangers
            treg = _opt_region(ch, 'target_region')
            if treg and _sh_units_removed_at(delta, treg):
                sc += 2
            if treg and _chat_names_region(treg, chats):
                sc += 1
        scored.append((sc, i))
    scored.sort(key=lambda x: -x[0])
    if len(scored) == 1 and scored[0][0] >= 1:
        return scored[0][1], None
    if scored and scored[0][0] >= 2 and scored[0][0] > scored[1][0]:
        stats['kinds_emit']['fuzzy_conf_' + tag] += 1
        return scored[0][1], None
    return None, 'fuzzy_lowconf_' + tag


# ==========================================================================
# UNIFIED COMBAT DECISION FAMILY  (emission spec 1.216, final wave)
# --------------------------------------------------------------------------
# One engine step-through per combat batch (replaces the round-1-only
# combat-card emitter + the single-round casualty emitter):
#   * per-ROUND combat-card commit/decline (attacker before defender), the
#     round's cards segmented by their line position among the round roll
#     chunks â€” a later-round card can never mislabel round 1;
#   * per-ROUND casualty allocation (k = elite downgrades, counted from the
#     round's replacement-Regular gestures in the settled per-round window;
#     undos are already net-resolved before segmentation);
#   * attacker-continue (pressing = the next round occurs) / cease;
#   * defender stand (no-retreat) / retreat destination / retreat-into-siege;
#   * siege press (elite downgrade buys another siege round) / no-press;
#   * siege trim (WHO returns to reinforcements when the garrison > 5;
#     mistakes to casualties tolerated + flagged);
#   * victor advance (WHICH units move in â€” partial advances legal) / stay.
# The batch's real dice force the engine in per-round engine order
# (rendered_census._combat_rounds_and_order); every decision is legality-
# matched against get_legal_actions on the live driven state and emitted
# only when the menu holds a real choice (>=2 options).  A completed drive
# must reproduce the client's settled per-side unit totals over the battle
# regions (final gate) or EVERY sample from the batch drops; an early break
# keeps only the samples emitted on states forced faithful up to that point.
# ==========================================================================
def _combat_round_windows(batch, leadership=None):
    """Round line-windows from the roll chunks, segmented exactly like
    rendered_census._combat_rounds_and_order (W2-008: dice counts plus the
    casualty / card gestures between the chunks - a one-sided card strike,
    a split leader re-roll or a defender-first round no longer opens a
    phantom round).  `leadership` = optional {'A': n, 'D': n} re-roll
    upper bound from the engine state.  Returns [{'first','last',
    'next_first'}]: 'first' = the round's first roll chunk line (a
    pre-combat strike included), 'last' = its last chunk line."""
    import rendered_census as rc
    rounds = rc._combat_chunk_rounds(batch, leadership)
    wins = []
    for items in rounds:
        lines = [c.get('line', 0) for c, _s, _cls in items]
        wins.append({'first': min(lines), 'last': max(lines)})
    for j, w in enumerate(wins):
        w['next_first'] = (wins[j + 1]['first'] if j + 1 < len(wins)
                           else 10 ** 9)
    return wins


def _battle_leadership(st, atk, dfn, atk_str):
    """UPPER BOUND {'A': n, 'D': n} of each side's re-roll Leadership at
    the battle start, from the engine state: every leader of the army in
    the region (Companion rating, Nazgul 1, Witch-king / Mouth 2, Saruman
    and plain leaders 1) + every elite of the army (Saruman's Isengard
    bonus) + the Fellowship's companions when it sits in the region,
    capped at 5 like the engine.  Deliberately an over-estimate: the
    round segmenter only uses it to rule a re-roll IMPOSSIBLE (W2-008).
    A side whose army is not on the state gets None (no cap)."""
    import rendered_replay as rr
    from game_state import Companion, Nazgul, UnitKind
    dfn_str = 'FREE_PEOPLES' if atk_str == 'SHADOW' else 'SHADOW'
    out = {}
    for tag, reg in (('A', atk), ('D', dfn)):
        army = st.armies.get(reg)
        if army is None:
            for k, aa in st.armies.items():
                if rr.sanitize(str(k)) == rr.sanitize(str(reg)):
                    army = aa
                    break
        if army is None:
            out[tag] = None
            continue
        n = 0
        for ld in getattr(army, 'leaders', []) or []:
            if isinstance(ld, Companion):
                n += int(getattr(ld, 'leadership', 1) or 0)
            elif isinstance(ld, Nazgul):
                n += 1
            elif type(ld).__name__ == 'Minion':
                nm = str(getattr(ld, 'name', '')).lower()
                n += 2 if ('witch' in nm or 'mouth' in nm) else 1
            else:
                n += 1
        n += sum(1 for u in getattr(army, 'units', []) or []
                 if getattr(u, 'kind', None) == UnitKind.ELITE)
        fs = getattr(st, 'fellowship', None)
        freg = getattr(fs, 'region', None) if fs is not None else None
        if freg is not None and (rr.sanitize(str(getattr(freg, 'name', freg)))
                                 == rr.sanitize(str(reg))):
            n += sum(int(getattr(c, 'leadership', 1) or 0)
                     for c in (getattr(fs, 'companions', []) or []))
        out[tag] = min(5, n)
    return out


def _round_cards(batch, wins):
    """Per-round combat-card commits keyed by CURRENT-slot side: cards[r] =
    {'fp': cid|None, 'sh': cid|None}.  A play is round r's commit iff its
    line sits after round r-1's last chunk and before round r's first chunk
    (pre-roll commitment order).  Returns (cards, n_unassigned)."""
    import rendered_census as rc
    from rendered_event_battle import funded_event_line
    help_event_line = funded_event_line(batch)
    lk = rc._card_lookup()
    cards = [{'fp': None, 'sh': None} for _ in wins]
    anom = 0
    anom_slots = set()
    commits = {}
    # A siege Event may use the same chosen-hand -> Current Card grammar
    # as a combat card. Bind only its initiating action, never every copy
    # of that title or every card killed before a roll. The native event
    # protocol commits the staged die before the battle's first roll.
    siege_event_line = uncertain_siege_event_line = None
    event_faces = {
        'SHADOW_Character_Grond_Hammer_of_the_Underworld': {'C', 'P'},
        'SHADOW_Strategy_The_Fighting_Uruk-hai': {'A', 'H', 'P'},
    }
    if wins and batch.get('die_side') == 'SHADOW':
        first_roll = min(w['first'] for w in wins)
        prefix = [r for r in batch['records'] if r.get('line', 0) < first_roll]
        stages = [r for r in prefix if r.get('die_stage')]
        if len(stages) == 1:
            staged = stages[0]
            face = batch.get('die_face')
            actor = staged.get('player')
            if (actor and actor != 'auto'
                    and tuple(staged['die_stage']) == ('SHADOW', face)
                    and _zone_norm(staged.get('from')) == 'sa dice'
                    and _zone_norm(staged.get('to')) == 'current action'):
                die = 'Shadow Action Die (%s)' % face
                closes = [r for r in prefix if r.get('op') == 'kill'
                          and r.get('player') == actor
                          and _zone_norm(r.get('zone')) == 'current action'
                          and sum(str(p).removeprefix('[1]').strip() == die
                                  for p in r.get('pieces', ())) == 1]
                if len(closes) == 1:
                    closed = closes[0]['line']
                    plays = [r for r in prefix if r.get('op') == 'move'
                             and staged['line'] < r.get('line', 0) < closed
                             and '/' in str(r.get('obj', ''))
                             and _zone_norm(r.get('to')) in _PLAY_AREA_DEST]
                    if len(plays) == 1:
                        play = plays[0]
                        cid = rc._lookup_cid(lk, play.get('obj', ''))
                        if (face in event_faces.get(cid, ())
                                and play.get('player') in (actor, 'auto')
                                and _zone_norm(play.get('from')) == 'shadow hand'
                                and _zone_norm(play.get('to')) == 'current sa card'):
                            discarded = [r for r in prefix if r.get('op') == 'kill'
                                and play['line'] < r.get('line', 0) <= closed
                                and r.get('player') == actor
                                and _zone_norm(r.get('zone')) in ('current action', 'current sa card')
                                and any(rc._lookup_cid(lk, str(p).removeprefix('[1]').strip()) == cid
                                        for p in r.get('pieces', ()))]
                            returned = any(r.get('op') in ('undo', 'redo') or (r.get('op') == 'move'
                                and play['line'] < r.get('line', 0) <= closed
                                and rc._lookup_cid(lk, r.get('obj', '')) == cid)
                                for r in prefix)
                            if not returned:
                                # P cannot buy an ordinary army attack. A
                                # dual-use C/A/H die additionally needs the
                                # event's exact discard within its paid window.
                                if face == 'P' or len(discarded) == 1:
                                    siege_event_line = play['line']
                                else:
                                    uncertain_siege_event_line = play['line']
    # a TABLE event play transits the Current slot en route to its Board
    # slot (T7 Fighting Uruk-hai) - it is a play_event, never this
    # round's combat-card commit (it stole Desperate Battle's slot)
    _table_titles = set()
    for r in batch['records']:
        if r.get('op') == 'move' and '/' in str(r.get('obj', '')) \
                and str(r.get('to', '')).startswith('Board'):
            _table_titles.add(str(r.get('obj', '')))
    for r in batch['records']:
        if r.get('op') != 'move':
            continue
        to = str(r.get('to', '')).lower()
        if 'card' not in to or 'current' not in to:
            continue
        title = r.get('obj', '')
        if '/' not in str(title):
            continue
        if str(title) in _table_titles:
            continue
        line = r.get('line', 0)
        if line == help_event_line or line == siege_event_line:
            continue
        if line == uncertain_siege_event_line:
            anom += 1
            anom_slots.add('sh')
            cards[0]['sh_ambiguous'] = True
            continue
        ridx = None
        for j, w in enumerate(wins):
            prev_last = wins[j - 1]['last'] if j else -1
            if prev_last < line < w['first']:
                ridx = j
                break
        slot = 'fp' if ('fp' in to or 'free' in to) else 'sh'
        if ridx is None:
            # pass 2 (T9 Grond): a leading attacker-only pseudo chunk
            # (WCtK post roll) can open a window EARLY, displacing the
            # next round's pre-roll commit INSIDE it.  Assign to the
            # containing window iff a roll chunk still FOLLOWS the
            # commit there (pre-roll in substance); a play after all of
            # a window's chunks stays anomalous.
            _chks = [c for c in batch['rng'] if c.get('kind') == 'roll_num']
            for j, w in enumerate(wins):
                if w['first'] <= line <= w['last'] and any(
                        w['first'] <= c.get('line', 0) <= w['last']
                        and c.get('line', 0) > line for c in _chks):
                    ridx = j
                    break
        if ridx is None:
            anom += 1                      # post-roll play: not a commit
            anom_slots.add(slot)
            continue
        cid = rc._lookup_cid(lk, title)
        if cid is None:
            anom += 1
            anom_slots.add(slot)
        else:
            commits.setdefault((ridx, slot), []).append(
                (cid, line, '(chosen)' in str(r.get('from', '')).lower()))
    for (ridx, slot), observed in commits.items():
        if cards[ridx].get(slot + '_ambiguous'):
            continue
        # The client's revealed chosen card identifies its combat commit.
        # A prior Event can occupy the same Current slot before the battle.
        # Multiple distinct unqualified plays remain ambiguous.
        chosen = [entry for entry in observed if entry[2]]
        candidates = chosen or observed
        if len({entry[0] for entry in candidates}) != 1:
            anom += 1
            anom_slots.add(slot)
            cards[ridx][slot + '_ambiguous'] = True
            continue
        cid, line, _ = candidates[0]
        cards[ridx][slot] = cid
        cards[ridx][slot + '_ln'] = line
    return cards, anom, anom_slots


def _combat_commit_board_matches(state, batch, line):
    """Require the figures at a recorded commit, including prior Event moves.

    The counterfactual battle starts at the die's staging. A card can move
    figures before that battle, so reaching a legal combat menu alone does
    not establish the expert's predecision position.
    """
    from board import sanitize
    record = next((r for r in batch['records'] if r.get('line') == line), None)
    truth = (record or {}).get('decision_truth')
    if truth is None:
        return not batch.get('record_truth_version')
    observed = _recon_state(truth, batch.get('turn'))

    def figures(st):
        result = {}
        for region, army in st.armies.items():
            units = sorted((u.nation.name, u.kind.name) for u in army.units)
            leaders = sorted((type(l).__name__, str(getattr(l, 'name', '')),
                              str(getattr(l, 'nation', ''))) for l in army.leaders)
            if units or leaders:
                result[sanitize(str(region))] = (units, leaders)
        return result

    return figures(state) == figures(observed)


def _confirmed_current_casualty(record, batch):
    """Verify a staged physical loss before using it in a recovery prefix."""
    import rendered_replay as rr
    origin = rr.zone_kind(record.get('from', ''))
    if origin[0] not in ('region', 'stronghold'):
        return False
    for token in _group_tokens(record.get('obj', '')):
        piece = rr.parse_piece(token)
        if not piece or piece[0] != 'unit':
            continue
        owner = _unit_side_str(piece[1])
        try:
            events = _casualty_gesture_operations(batch, owner, {origin[1]},
                record.get('line', 0), batch.get('end_line', 10 ** 9) + 1)
        except _CasualtyGestureError:
            return False
        if not any(e[0] == record.get('line', 0) for e in events):
            return False
    return True


def _apply_window_casualties(st, batch, regs, upto_line):
    """Brute-apply casualty gestures (kills / parked-to-cas moves /
    downgrade replacement returns) with line < upto_line to the recon
    state's battle-region armies - a mirror-derived MID-BATTLE state for
    the gesture-driven round emitter (menus only need honest unit
    counts)."""
    import rendered_replay as rr
    import combat as _cmb
    import game_state as gs
    for r in batch['records']:
        ln = r.get('line', 0)
        if ln >= upto_line:
            continue
        toks, reg = [], None
        if r.get('op') == 'kill':
            zone = rr.zone_kind(str(r.get('zone', '') or ''))
            if zone[0] in ('region', 'stronghold') and zone[1] in regs:
                reg = zone[1]
                for pc in (r.get('pieces') or []):
                    toks.extend(_group_tokens(str(pc)))
        elif r.get('op') == 'move':
            fk = rr.zone_kind(r.get('from', ''))
            tk = rr.zone_kind(r.get('to', ''))
            if fk[0] in ('region', 'stronghold') and fk[1] in regs \
                    and (tk[0] == 'cas'
                         or (tk[0] == 'reinf' and upto_line > 0)
                         or ('current' in str(r.get('to', '')).lower()
                             and _confirmed_current_casualty(r, batch))):
                # kills-as-returns dialect: region->REINF inside the
                # battle = a casualty (GAME19 MM garrison)
                reg = fk[1]
                toks = _group_tokens(r.get('obj', ''))
            elif fk[0] in ('cas', 'reinf') and \
                    tk[0] in ('region', 'stronghold') and tk[1] in regs:
                # replacement Regular return (downgrade completion)
                _k2, _a2 = _cmb._army_entry(st, tk[1])
                for t in _group_tokens(r.get('obj', '')):
                    p = rr.parse_piece(t)
                    if p and p[0] == 'unit' and p[2] == 'REGULAR' \
                            and _a2 is not None:
                        for _ in range(p[3]):
                            nation = gs.Nation[p[1]]
                            if fk[0] == 'cas' and _cmb.side_of_nation(nation) == gs.Side.FREE_PEOPLES:
                                available = (st.fp_casualties.get(nation) or {}).get('regular', 0)
                                if available <= 0:
                                    raise ValueError('recovery replacement missing FP casualty figure')
                                st.fp_casualties[nation]['regular'] -= 1
                            _a2.units.append(gs.Unit(nation, gs.UnitKind.REGULAR))
                continue
            elif fk[0] == 'region' and fk[1] not in regs \
                    and tk[0] in ('region', 'stronghold') \
                    and tk[1] in regs:
                # mid-batch MOVE-IN (Corsairs landing / assault advance):
                # without it the attacker army is absent from the battle
                # region and every attacker-side menu starves (T9 DA)
                import game_logic as _gl
                _src = _cmb._army_entry(st, fk[1])[1]
                _dst = _gl.get_or_create_army(st, tk[1])
                for t in _group_tokens(r.get('obj', '')):
                    p = rr.parse_piece(t)
                    if p and p[0] == 'unit':
                        kind = (gs.UnitKind.ELITE if p[2] == 'ELITE'
                                else gs.UnitKind.REGULAR)
                        for _ in range(p[3]):
                            moved = None
                            if _src is not None:
                                moved = next(
                                    (u for u in _src.units
                                     if u.kind == kind
                                     and u.nation.name == p[1]), None)
                            if moved is not None:
                                _src.units.remove(moved)
                                _dst.units.append(moved)
                            else:
                                _dst.units.append(gs.Unit(
                                    gs.Nation[p[1]], kind))
                    elif p and p[0] in ('nazgul', 'named', 'leader'):
                        # LEADERS ride the move-in too - without them
                        # the Corsairs' Nazgul stay at Umbar and every
                        # 'Nazgul in battle' combat-card condition
                        # false-fails (T9 Grond/Breaking)
                        for _ in range(p[3] if p[0] == 'nazgul' else 1):
                            moved = None
                            if _src is not None:
                                if p[0] == 'nazgul':
                                    moved = next(
                                        (l for l in _src.leaders
                                         if isinstance(l, gs.Nazgul)),
                                        None)
                                else:
                                    moved = next(
                                        (l for l in _src.leaders
                                         if not isinstance(l, gs.Nazgul)
                                         and rr._named_eq(
                                             str(getattr(l, 'name', '')),
                                             str(p[1]))), None)
                            if moved is not None:
                                _src.leaders.remove(moved)
                                _dst.leaders.append(moved)
                            elif p[0] == 'nazgul':
                                _dst.leaders.append(
                                    gs.Nazgul(nation=gs.Nation.SAURON))
                continue
        if not toks or reg is None:
            continue
        # the recon may keep a besieged garrison under the box key -
        # remove each fallen unit from WHICHEVER matching-key army
        # actually holds a unit of that nation+kind
        cand = [a for k, a in st.armies.items()
                if rr.sanitize(k).startswith(reg)]
        for t in toks:
            p = rr.parse_piece(t)
            if not p or p[0] != 'unit':
                continue
            kind = (gs.UnitKind.ELITE if p[2] == 'ELITE'
                    else gs.UnitKind.REGULAR)
            for _ in range(p[3]):
                for _a1 in cand:
                    hit = next((u for u in _a1.units
                                if u.kind == kind
                                and u.nation.name == p[1]), None)
                    if hit is not None:
                        _a1.units.remove(hit)
                        _cmb._track_fp_casualties(st, [hit], _cmb.side_of_nation(hit.nation))
                        break


class _CasualtyGestureError(ValueError):
    """The rendered record cannot establish an exact casualty decision."""


def _casualty_gesture_operations(batch, owner, regs, lo, hi, *, with_evidence_end=False,
                                 cleanup_hi=None):
    """Ordered physical loss evidence; no nation or within-group order guesses.

    A Current-card parking is a loss only with a later matching cleanup.
    Its original departure line determines decision order, not cleanup time.
    """
    import rendered_replay as rr
    events = []
    records = sorted(batch['records'], key=lambda r: r.get('line', 0))
    cleanup_used = Counter()
    for record in records:
        line = record.get('line', 0)
        if not lo <= line < hi:
            continue
        tokens, direction, staged = [], None, False
        if record.get('op') == 'kill':
            zone = rr.zone_kind(str(record.get('zone', '') or ''))
            if zone[0] in ('region', 'stronghold') and zone[1] in regs:
                direction = 'loss'
                for piece in record.get('pieces') or []:
                    tokens.extend(_group_tokens(str(piece)))
        elif record.get('op') == 'move':
            origin = rr.zone_kind(record.get('from', ''))
            dest = rr.zone_kind(record.get('to', ''))
            if origin[0] in ('region', 'stronghold') and origin[1] in regs:
                if dest[0] in ('cas', 'reinf'):
                    direction = 'loss'
                elif 'current' in str(record.get('to', '')).lower():
                    direction, staged = 'loss', True
            elif origin[0] in ('cas', 'reinf') and dest[0] in ('region', 'stronghold') and dest[1] in regs:
                direction = 'replacement'
            if direction:
                tokens = _group_tokens(record.get('obj', ''))
        if direction is None:
            continue
        pieces = []
        for token in tokens:
            piece = rr.parse_piece(token)
            if piece and piece[0] == 'unit' and _unit_side_str(piece[1]) == owner:
                pieces.extend([(piece[1], piece[2])] * int(piece[3]))
        if len(set(pieces)) > 1 and not (direction == 'loss' and all(kind == 'REGULAR' for _, kind in pieces)):
            raise _CasualtyGestureError('group_order_ambiguous')
        if staged and pieces:
            # A return or correction before the eventual cleanup invalidates
            # this parked loss; do not silently reinterpret the player's undo.
            outstanding = Counter(pieces)
            for later in records:
                if not line < later.get('line', 0) < (hi if cleanup_hi is None else cleanup_hi):
                    continue
                if later.get('op') == 'move' and 'current' in str(later.get('from', '')).lower():
                    target = rr.zone_kind(later.get('to', ''))
                    moved = [rr.parse_piece(t) for t in _group_tokens(later.get('obj', ''))]
                    if any(p and p[0] == 'unit' and (p[1], p[2]) in outstanding for p in moved):
                        if target[0] in ('region', 'stronghold'):
                            raise _CasualtyGestureError('parked_loss_reversed')
                if later.get('op') != 'kill' or 'current' not in str(later.get('zone', '')).lower():
                    continue
                cleanup = Counter()
                for raw in later.get('pieces') or []:
                    for token in _group_tokens(str(raw)):
                        piece = rr.parse_piece(token)
                        if piece and piece[0] == 'unit':
                            cleanup[(piece[1], piece[2])] += int(piece[3])
                for figure, count in cleanup.items():
                    if figure in outstanding:
                        key = (later.get('line', 0), *figure)
                        available = max(0, count - cleanup_used[key])
                        used = min(outstanding[figure], available)
                        outstanding[figure] -= used
                        cleanup_used[key] += used
                if all(count <= 0 for count in outstanding.values()):
                    break
            if any(count > 0 for count in outstanding.values()):
                raise _CasualtyGestureError('parked_loss_unconfirmed')
        for nation, kind in pieces:
            if direction == 'replacement' and kind != 'REGULAR':
                raise _CasualtyGestureError('replacement_not_regular')
            events.append((line, nation, 'replacement' if direction == 'replacement'
                           else 'loss_regular' if kind == 'REGULAR' else 'loss_elite'))
    # Swaps may be rendered in either order, but must be adjacent owner
    # operations of the same nation. Never borrow another nation's Regular.
    result, index = [], 0
    while index < len(events):
        event = events[index]
        if index + 1 < len(events):
            after = events[index + 1]
            if event[1] == after[1] and {event[2], after[2]} == {'loss_elite', 'replacement'}:
                result.append((event[0], event[1], 'reduce_elite') +
                              ((after[0],) if with_evidence_end else ()))
                index += 2
                continue
        if event[2] == 'replacement':
            raise _CasualtyGestureError('replacement_unpaired')
        result.append(event + ((event[0],) if with_evidence_end else ()))
        index += 1
    return result


def _casualty_event_matches(event, operation):
    nation = operation.get('nation')
    nation = getattr(nation, 'name', nation)
    if event[1] != nation:
        return False
    kind = operation.get('operation')
    return ((event[2] == 'loss_regular' and kind == 'remove_regular')
            or (event[2] == 'reduce_elite' and kind == 'reduce_elite'
                and operation.get('replacement_source') is not None)
            or (event[2] == 'loss_elite' and (kind == 'remove_elite'
                or (kind == 'reduce_elite' and operation.get('replacement_source') is None))))


def _casualty_event_operation(state, event, hits, *, known_hits=True):
    """Map a literal loss to an operation, retaining shortage ambiguity."""
    import ai
    from game_state import Nation
    nation = Nation[event[1]]
    operation = {'loss_regular': 'remove_regular', 'reduce_elite': 'reduce_elite'}.get(event[2])
    if operation is None:
        from combat import _cas_box_regulars
        available = (_cas_box_regulars(state, nation) > 0
                     or ai.figures_available(state, nation, 'regular') > 0)
        if not known_hits and not available:
            raise _CasualtyGestureError('shortage_hit_cost_unknown')
        if hits == 1 and not available:
            operation = 'reduce_elite'
        elif hits >= 2:
            operation = 'remove_elite'
        else:
            raise _CasualtyGestureError('elite_loss_hit_cost_mismatch')
    return nation, operation


def _drive_card_casualty_gestures(state, events):
    """Replay one card-damage owner sequence before publishing training rows."""
    import ai
    from rendered_casualty_groups import consume as consume_group, annotate
    events = list(events)
    import rendered_replay as rr
    pco = state.pending_card_option or {}
    side, region = pco.get('cas_side'), pco.get('cas_region')
    card_id = pco.get('card_id')
    current, cursor, rows = state, 0, []

    def consume(witnesses):
        nonlocal cursor
        for witness in witnesses:
            if (witness.get('stage') != 'card' or witness.get('side') != side
                    or rr.sanitize(str(witness.get('region'))) != rr.sanitize(str(region))):
                continue
            if not consume_group(events, cursor, witness, _casualty_event_matches):
                raise _CasualtyGestureError('forced_operation_mismatch')
            cursor += 1

    consume(getattr(current, '_casualty_operations', ()) or ())
    while cursor < len(events):
        pco = current.pending_card_option or {}
        if (not pco.get('card_casualty') or pco.get('cas_side') != side
                or pco.get('cas_region') != region or pco.get('card_id') != card_id):
            raise _CasualtyGestureError('extra_owner_losses')
        nation, operation = _casualty_event_operation(current, events[cursor], int(pco['cas_hits']))
        legal = ai.get_legal_actions(current, side)
        matches = []
        for index, action in enumerate(legal):
            choices = action.params.get('choices') or {}
            if (action.action_type == 'card_option' and choices.get('nation') == nation
                    and choices.get('casualty_operation') == operation
                    and rr.sanitize(str(choices.get('target_region'))) == rr.sanitize(str(region))):
                matches.append(index)
        if len(matches) != 1:
            raise _CasualtyGestureError('operation_unmatched')
        chosen = matches[0]
        if len(legal) >= 2:
            before = ai.clone_state(current)
            annotate(before, legal, chosen, events, cursor, card=True)
            rows.append((before, side, legal, chosen))
        successor = ai.apply_action(current, legal[chosen])
        witnesses = getattr(successor, '_casualty_operations', ()) or ()
        if not witnesses:
            raise _CasualtyGestureError('operation_witness_missing')
        consume(witnesses)
        current = successor
    pco = current.pending_card_option or {}
    if (pco.get('card_casualty') and pco.get('cas_side') == side
            and pco.get('cas_region') == region and pco.get('card_id') == card_id):
        raise _CasualtyGestureError('owner_losses_incomplete')
    return current, rows


def _drive_casualty_gestures(state, side, events, *, known_hits=True):
    """Follow literal operations and validate engine-forced tails atomically.

    Returns final state and predecision (state, menu, chosen index) rows.
    Callers emit rows only after the complete owner sequence validates.
    """
    import ai
    from rendered_casualty_groups import consume as consume_group, annotate
    events = list(events)
    from game_state import Nation
    pc = state.pending_combat or {}
    stage = pc.get('alloc_stage')
    initial_hits = int(pc.get('atk_hits' if stage == 'attacker' else 'def_hits', 0))
    region = pc.get('attacker_region' if stage == 'attacker' else 'defender_region')
    capacity = sum(2 if unit.kind == ai.UnitKind.ELITE else 1
                   for unit in ai._alloc_side_units(state, region, side))
    current, cursor, rows = state, 0, []

    def consume(witnesses):
        nonlocal cursor
        for witness in witnesses:
            if witness.get('side') != side or witness.get('stage') != stage:
                continue
            if not consume_group(events, cursor, witness, _casualty_event_matches):
                raise _CasualtyGestureError('forced_operation_mismatch')
            cursor += 1

    consume(getattr(current, '_casualty_operations', ()) or ())
    while cursor < len(events):
        pc = current.pending_combat or {}
        if pc.get('phase') != 'casualty_allocation' or pc.get('deciding_side') != side or pc.get('alloc_stage') != stage:
            raise _CasualtyGestureError('extra_owner_losses')
        hits = int(pc.get('atk_hits' if stage == 'attacker' else 'def_hits', 0))
        event = events[cursor]
        nation, operation = _casualty_event_operation(current, event, hits, known_hits=known_hits)
        legal = ai.get_legal_actions(current, side)
        matches = [i for i, action in enumerate(legal)
                   if action.action_type == 'casualty_allocate'
                   and action.params.get('nation') == nation
                   and action.params.get('operation') == operation]
        if len(matches) != 1:
            raise _CasualtyGestureError('operation_unmatched')
        chosen = matches[0]
        if sum(a.action_type == 'casualty_allocate' for a in legal) >= 2:
            before = ai.clone_state(current)
            annotate(before, legal, chosen, events, cursor, card=False)
            rows.append((before, legal, chosen))
        successor = ai.apply_action(current, legal[chosen])
        witnesses = getattr(successor, '_casualty_operations', ()) or ()
        if not witnesses:
            raise _CasualtyGestureError('operation_witness_missing')
        consume(witnesses)
        current = successor
    pc = current.pending_combat or {}
    if pc.get('phase') == 'casualty_allocation' and pc.get('deciding_side') == side and pc.get('alloc_stage') == stage:
        raise _CasualtyGestureError('owner_losses_incomplete')
    if rows and not known_hits and initial_hits >= capacity:
        # A removed army proves absorbed damage, not incoming overkill.
        # FP Elite wipes can still offer pool-distinct choices; do not
        # train those choices on a guessed remaining-hit observation.
        raise _CasualtyGestureError('incoming_hits_unknown_at_wipe')
    return current, rows


def _drive_onslaught_ordinary_prefix(state, batch, regs, lo, post_line):
    """Prove both ordinary loss obligations before a recorded Onslaught roll.

    Try finite literal prefixes on independent states. The first complete
    owner sequence is accepted only when the public executor opens the other
    owner's ordinary allocation or the actual deferred Onslaught menu. Later
    sacrifices are not ordinary losses, and cannot repair an earlier mismatch.
    Returns the paused successor, allocation rows and last ordinary evidence
    line. The input state is unchanged on success and refusal.
    """
    import ai
    from game_state import Side
    current = ai.clone_state(state)
    rows, spans = [], {}
    cutoffs = sorted({int(r.get('line', 0)) + 1 for r in batch['records']
                      if lo <= int(r.get('line', 0)) < post_line})
    for _ in range(2):
        pc = current.pending_combat or {}
        if pc.get('phase') != 'casualty_allocation':
            break
        actor = pc.get('deciding_side')
        owner = 'SHADOW' if actor == Side.SHADOW else 'FREE_PEOPLES'
        role = pc.get('alloc_stage')
        if any(w.get('stage') != role and w.get('stage') not in spans
               for w in (getattr(current, '_casualty_operations', ()) or ())):
            raise _CasualtyGestureError('onslaught_skipped_owner_witness_unverified')
        matched = None
        for cutoff in cutoffs:
            try:
                events = _casualty_gesture_operations(batch, owner, regs, lo, cutoff)
                candidate, choices = _drive_casualty_gestures(
                    ai.clone_state(current), actor, events)
            except _CasualtyGestureError:
                continue
            next_pc = candidate.pending_combat or {}
            effect = next_pc.get('effect_choice') or {}
            if (next_pc.get('phase') != 'casualty_allocation'
                    and any(w.get('stage') != role and w.get('stage') not in spans
                            for w in (getattr(candidate, '_casualty_operations', ()) or ()))):
                raise _CasualtyGestureError('onslaught_skipped_owner_witness_unverified')
            if not (next_pc.get('phase') == 'casualty_allocation'
                    and next_pc.get('alloc_stage') != role) and not (
                    next_pc.get('phase') == 'combat_effect_choice'
                    and effect.get('effect') == 'onslaught'
                    and effect.get('stage') == 'post'):
                raise _CasualtyGestureError('onslaught_prefix_wrong_successor')
            matched = candidate, choices, events, cutoff
            break
        if matched is None:
            raise _CasualtyGestureError('onslaught_ordinary_prefix_unproven')
        current, choices, events, cutoff = matched
        rows.extend(choices)
        if events:
            spans[role] = (min(int(e[0]) for e in events), cutoff - 1)
    pc = current.pending_combat or {}
    effect = pc.get('effect_choice') or {}
    if pc.get('phase') != 'combat_effect_choice' or effect.get('effect') != 'onslaught':
        raise _CasualtyGestureError('onslaught_ordinary_incomplete')
    if ('attacker' in spans and 'defender' in spans
            and spans['attacker'][1] >= spans['defender'][0]):
        raise _CasualtyGestureError('onslaught_ordinary_owner_order')
    boundary = max((span[1] for span in spans.values()), default=lo - 1)
    pc['_rx_onslaught_ordinary_end'] = boundary
    pc['_rx_onslaught_ordinary_round'] = int(pc.get('rounds_fought', 0)) - 1
    return current, rows, boundary


def _drive_commuting_ordinary_cleanup(state, batch, regs, lo, hi):
    """Recover a later decision after reversed, independent direct removals.

    Both Armies must survive. Each side pays its exact known ordinary hits
    with complete direct removals (kills or board-to-pool moves), using
    disjoint nations. No replacement, intervening random event or other
    unit move is allowed in that prefix.
    Replay uses public attacker-first actions, but emits NO ordinary labels:
    their original information order cannot be certified. Only the resulting
    shared position can support subsequent actual decisions.
    """
    import ai
    import rendered_replay as rr
    pc = state.pending_combat or {}
    if (pc.get('phase') != 'casualty_allocation' or pc.get('alloc_stage') != 'attacker'
            or not pc.get('_observed_round_completed')
            or getattr(state, '_casualty_operations', None)):
        raise _CasualtyGestureError('commuting_cleanup_not_initial_ordinary')
    sides = {role: pc[role + '_side'] for role in ('attacker', 'defender')}
    needs = {'attacker': int(pc.get('atk_hits', 0)), 'defender': int(pc.get('def_hits', 0))}
    if min(needs.values()) <= 0:
        raise _CasualtyGestureError('commuting_cleanup_requires_two_hit_owners')
    groups = {role: [] for role in sides}
    remaining = dict(needs)
    first_line, boundary = None, None
    for record in sorted(batch.get('records', ()), key=lambda r: int(r.get('line', 0))):
        line = int(record.get('line', 0))
        if not lo <= line < hi:
            continue
        op = record.get('op')
        zone = rr.zone_kind(record.get('zone', ''))
        parsed = None
        if op == 'move':
            origin, target = rr.zone_kind(record.get('from', '')), rr.zone_kind(record.get('to', ''))
            pieces = [rr.parse_piece(t) for t in _group_tokens(record.get('obj', ''))]
            if (origin[0] in ('region', 'stronghold') and origin[1] in regs
                    and target[0] in ('cas', 'reinf')):
                parsed = pieces
            elif any(p and p[0] == 'unit' for p in pieces) and (
                    origin[1] in regs or target[1] in regs):
                raise _CasualtyGestureError('commuting_cleanup_intervening_unit_move')
            if str(record.get('to', '')).lower().strip() in ('current sa card', 'current fp card'):
                raise _CasualtyGestureError('commuting_cleanup_intervening_card_commit')
            if parsed is None:
                continue
        else:
            if op != 'kill' or zone[0] not in ('region', 'stronghold') or zone[1] not in regs:
                continue
            parsed = [rr.parse_piece(token) for text in record.get('pieces', ())
                      for token in _group_tokens(text)]
        if not parsed or any(not p or p[0] != 'unit' for p in parsed):
            raise _CasualtyGestureError('commuting_cleanup_nonunit_loss')
        by_role = {role: [] for role in sides}
        for piece in parsed:
            _, nation, kind, count = piece
            owner = next((role for role, side in sides.items()
                          if side.name == _unit_side_str(nation)), None)
            if owner is None or kind not in ('REGULAR', 'ELITE'):
                raise _CasualtyGestureError('commuting_cleanup_unknown_owner')
            by_role[owner].extend((line, nation, 'loss_elite' if kind == 'ELITE' else 'loss_regular')
                                 for _ in range(int(count)))
        for owner, events in by_role.items():
            if not events:
                continue
            cost = sum(2 if e[2] == 'loss_elite' else 1 for e in events)
            if cost > remaining[owner]:
                raise _CasualtyGestureError('commuting_cleanup_not_exact_prefix')
            remaining[owner] -= cost
            # This is an implementation order within the recorded group,
            # never a claimed human sequence or an intermediate label.
            groups[owner].extend(sorted(events, key=lambda e: (e[2] != 'loss_elite', e[1])))
        first_line = line if first_line is None else first_line
        if not any(remaining.values()):
            boundary = line
            break
    if boundary is None:
        raise _CasualtyGestureError('commuting_cleanup_incomplete')
    if any(lo <= int(chunk.get('line', 0)) <= boundary for chunk in batch.get('rng', ())):
        raise _CasualtyGestureError('commuting_cleanup_intervening_random_event')
    nations = [{e[1] for e in groups[role]} for role in ('attacker', 'defender')]
    if nations[0] & nations[1]:
        raise _CasualtyGestureError('commuting_cleanup_shared_nation')
    for role, side in sides.items():
        units = ai._alloc_side_units(state, pc[role + '_region'], side)
        if len(groups[role]) >= len(units):
            raise _CasualtyGestureError('commuting_cleanup_army_wipe')
    current = ai.clone_state(state)
    suppressed = 0
    for role in ('attacker', 'defender'):
        current_pc = current.pending_combat or {}
        if current_pc.get('phase') != 'casualty_allocation' or current_pc.get('alloc_stage') != role:
            raise _CasualtyGestureError('commuting_cleanup_skipped_owner')
        current, rows = _drive_casualty_gestures(current, sides[role], groups[role])
        suppressed += len(rows)
    final_pc = current.pending_combat or {}
    if final_pc.get('phase') not in ('combat_round_result', 'attacker_continue', 'defender_retreat'):
        raise _CasualtyGestureError('commuting_cleanup_unexpected_tail')
    if not all(ai._alloc_side_units(current, final_pc[role + '_region'], side)
               for role, side in sides.items()):
        raise _CasualtyGestureError('commuting_cleanup_army_wipe')
    return current, {'schema': 'commuting_ordinary_cleanup_v1',
        'round': int(pc.get('rounds_fought', 0)) - 1, 'first_line': first_line,
        'last_line': boundary, 'suppressed_ordinary_rows': suppressed,
        'groups': groups, 'emits_ordinary_rows': False}


def _recovery_allocation_lines(batch, regs, lo, hi, attacker_str):
    """Prove ordinary attacker-then-defender casualty gesture boundaries.

    Each returned line is BEFORE that owner's first removal/replacement.
    Reversed, interleaved, or simultaneous records cannot establish the
    rule-ordered observations; do not read later choices to repair them.
    """
    import rendered_replay as rr
    by_owner = {'SHADOW': [], 'FREE_PEOPLES': []}
    for record in batch['records']:
        line = record.get('line', 0)
        if not lo <= line < hi:
            continue
        tokens = []
        if record.get('op') == 'kill':
            zone = rr.zone_kind(str(record.get('zone', '') or ''))
            if zone[0] in ('region', 'stronghold') and zone[1] in regs:
                for piece in record.get('pieces') or []:
                    tokens.extend(_group_tokens(str(piece)))
        elif record.get('op') == 'move':
            origin = rr.zone_kind(record.get('from', ''))
            dest = rr.zone_kind(record.get('to', ''))
            outward = (origin[0] in ('region', 'stronghold') and origin[1] in regs
                       and (dest[0] in ('cas', 'reinf') or 'current' in str(record.get('to', '')).lower()))
            replacement = (origin[0] in ('cas', 'reinf')
                           and dest[0] in ('region', 'stronghold') and dest[1] in regs)
            if outward or replacement:
                tokens = _group_tokens(record.get('obj', ''))
        for token in tokens:
            piece = rr.parse_piece(token)
            if piece and piece[0] == 'unit':
                by_owner[_unit_side_str(piece[1])].append(line)
    defender_str = 'FREE_PEOPLES' if attacker_str == 'SHADOW' else 'SHADOW'
    attacker = by_owner[attacker_str]
    defender = by_owner[defender_str]
    if attacker and defender and max(attacker) >= min(defender):
        return None
    return {owner: min(lines) for owner, lines in by_owner.items() if lines}


def _window_replacements(batch, side_str, regs, lo=0, hi=None):
    """Elite-DOWNGRADE evidence for `side_str` within line window [lo,hi):
    replacement Regulars moved from casualties/reinforcements into a battle
    region (a mid-combat cas/reinf->region move is never a muster)."""
    import rendered_replay as rr
    n = 0
    for r in batch['records']:
        if r.get('op') != 'move':
            continue
        ln = r.get('line', 0)
        if ln < lo or (hi is not None and ln >= hi):
            continue
        fk = rr.zone_kind(r.get('from', ''))
        tk = rr.zone_kind(r.get('to', ''))
        if fk[0] not in ('cas', 'reinf'):
            continue
        if tk[0] not in ('region', 'stronghold') or tk[1] not in regs:
            continue
        for t in _group_tokens(r.get('obj', '')):
            p = rr.parse_piece(t)
            if p and p[0] == 'unit' and p[2] == 'REGULAR':
                s = 'SHADOW' if p[1] in _SHADOW_NATS else 'FREE_PEOPLES'
                if s == side_str:
                    n += p[3]
    return n


def _unit_side_str(nat):
    return 'SHADOW' if nat in _SHADOW_NATS else 'FREE_PEOPLES'


def _window_owner_hits(batch, owner_str, regs, lo, hi):
    """HITS `owner_str` absorbed in the battle regions within [lo, hi),
    reconstructed from the rendered casualty gestures.  A regular removed = 1
    hit, an elite removed outright = 2, an elite downgraded (replacement
    Regular appears) = 1.  = regs_to_cas + 2*elites_to_cas - downgrades
    (each downgraded elite goes to casualties AND yields one replacement
    Regular, so it double-counts as 2 in elites_to_cas and must net to 1).
    This is the back-calc denominator for Onslaught/Relentless self-hits
    (spec 1.216 item 5: self-hits = owner deaths - engine roll casualties)."""
    import rendered_replay as rr
    regs_to_cas = elites_to_cas = 0
    for r in batch['records']:
        ln = r.get('line', 0)
        if ln < lo or (hi is not None and ln >= hi):
            continue
        if r.get('op') == 'move':
            fk = rr.zone_kind(r.get('from', ''))
            tk = rr.zone_kind(r.get('to', ''))
            if fk[0] not in ('region', 'stronghold') or fk[1] not in regs:
                continue
            # 'cas' = normal casualty park; GAME19 dialect also renders
            # mid-battle unit deaths as region->REINFORCEMENTS returns
            # ('those actually go to reinforcements' habit) - inside a
            # ROLL window those are casualties, not trims
            if tk[0] != 'cas' and not (tk[0] == 'reinf' and lo > 0):
                continue
            toks = _group_tokens(r.get('obj', ''))
        elif r.get('op') == 'kill':
            # USER truth-table #9 (game01_verification.md): the client
            # renders most unit deaths as 'kills X in R' (op=kill), not
            # moves-to-casualties - scan both, or whole battles read 0
            # (T5 DA L532/535, T6 Lorien).
            zone = rr.zone_kind(str(r.get('zone', '') or ''))
            if zone[0] not in ('region', 'stronghold') \
                    or zone[1] not in regs:
                continue
            toks = []
            for pc in (r.get('pieces') or []):
                toks.extend(_group_tokens(str(pc)))
        else:
            continue
        for t in toks:
            p = rr.parse_piece(t)
            if p and p[0] == 'unit' and _unit_side_str(p[1]) == owner_str:
                if p[2] == 'ELITE':
                    elites_to_cas += p[3]
                else:
                    regs_to_cas += p[3]
    downgrades = _window_replacements(batch, owner_str, regs, lo, hi)
    return max(0, regs_to_cas + 2 * elites_to_cas - downgrades)


def _window_retreat(batch, region, side_str, lo, hi):
    """Defender-family retreat evidence in [lo,hi): units of `side_str`
    leaving `region` â€” ('siege', region) for a withdrawal into the region's
    own stronghold box, ('region', dest) for a field retreat, (None, None)."""
    import rendered_replay as rr
    from collections import Counter
    direct = []
    def pieces(record):
        counts = Counter()
        for token in _group_tokens(record.get('obj', '')):
            piece = rr.parse_piece(token)
            if piece and piece[0] == 'unit' and _unit_side_str(piece[1]) == side_str:
                counts[(piece[1], piece[2])] += piece[3]
        return counts
    records = sorted((r for r in batch['records'] if r.get('op') == 'move'
                      and lo <= r.get('line', 0) < hi), key=lambda r: r['line'])
    for r in records:
        if r.get('op') != 'move':
            continue
        ln = r.get('line', 0)
        if ln < lo or ln >= hi:
            continue
        fk = rr.zone_kind(r.get('from', ''))
        tk = rr.zone_kind(r.get('to', ''))
        if fk[0] not in ('region', 'stronghold') or fk[1] != region:
            continue
        has_side_unit = False
        for t in _group_tokens(r.get('obj', '')):
            p = rr.parse_piece(t)
            if p and p[0] == 'unit' and _unit_side_str(p[1]) == side_str:
                has_side_unit = True
        if not has_side_unit:
            continue
        if tk[0] == 'stronghold' and tk[1] == region and fk[0] == 'region':
            return ('siege', region)
        if tk[0] in ('region', 'stronghold') and tk[1] != region:
            direct.append((r, tk[1], pieces(r)))
    if direct:
        destinations = {dest for _r, dest, _pieces in direct}
        if len(destinations) != 1:
            return ('ambiguous', None)
        dest = direct[0][1]
        counts = sum((item[2] for item in direct), Counter())
        last_line = max(item[0]['line'] for item in direct)
        visited = {region}
        while True:
            onward = [r for r in records if r['line'] > last_line
                       and rr.zone_kind(r.get('from', '')) == ('region', dest)
                       and pieces(r)]
            if not onward:
                return ('region', dest)
            # Identical pre-existing figures make the second drag's identity
            # unprovable. A split, partial transfer or return is also not a
            # uniquely recorded retreat destination.
            old = Counter((nat, kind) for key, (units, _leaders) in
                          batch.get('truth_before', {}).get('armies', {}).items()
                          if rr.sanitize(key) == dest for nat, kind in units
                          if _unit_side_str(nat) == side_str)
            transfers = [r for r in records if r['line'] <= last_line
                         and rr.zone_kind(r.get('to', '')) == ('region', dest)
                         and r not in [item[0] for item in direct] and pieces(r)]
            if any(old[piece] for piece in counts) or transfers or len(onward) != 1:
                return ('ambiguous', None)
            record = onward[0]
            target = rr.zone_kind(record.get('to', ''))
            if pieces(record) != counts or target[0] != 'region' or target[1] in visited:
                return ('ambiguous', None)
            visited.add(dest)
            dest, last_line = target[1], record['line']
    return (None, None)


def _correct_regular_retreat(batch, state, region, side_str, lo, hi):
    """Recognize a narrowly proven remove-all/restore-survivor gesture.

    The live completed round supplies the survivor count. This is only a
    physical correction of uniform Regular defenders, never an inferred
    incoming damage budget or a reconstruction of an Elite choice.
    """
    import copy
    import re
    import rendered_replay as rr
    import rendered_census as rc
    from collections import Counter
    returns = []
    for record in batch['records']:
        if record.get('op') != 'move' or not lo <= record.get('line', 0) < hi:
            continue
        origin = rr.zone_kind(record.get('from', ''))
        target = rr.zone_kind(record.get('to', ''))
        if origin[0] not in ('cas', 'reinf') or target[0] != 'region' or target[1] == region:
            continue
        pieces = [rr.parse_piece(t) for t in _group_tokens(record.get('obj', ''))]
        if any(p and p[0] == 'unit' and _unit_side_str(p[1]) == side_str for p in pieces):
            returns.append((record, target[1], pieces))
    if not returns:
        return batch, None
    pc = state.pending_combat or {}
    if (len(returns) != 1 or pc.get('is_siege')
            or any((pc.get('_p9') or {}).get(k) for k in ('atk_card', 'def_card'))):
        return batch, 'unproven'
    record, dest, pieces = returns[0]
    if len(pieces) != 1 or not pieces[0] or pieces[0][0] != 'unit' or pieces[0][2] != 'REGULAR':
        return batch, 'unproven'
    _kind, nation, _unit_kind, count = pieces[0]
    survivors = rc._eng_side_units(state, region, side_str)
    initial = rc._snap_side_units(batch['truth_before'], region, side_str)
    if survivors != [(nation, 'REGULAR')] * count or not initial or any(
            unit != (nation, 'REGULAR') for unit in initial):
        return batch, 'unproven'
    if any(leaders for key, (_units, leaders) in batch['truth_before']['armies'].items()
           if rr.sanitize(key) == region):
        return batch, 'unproven'
    kills = [r for r in batch['records'] if r.get('op') == 'kill'
             and lo <= r.get('line', 0) < record['line']
             and rr.zone_kind(r.get('zone', '')) == ('region', region)]
    if len(kills) != 1 or len(kills[0].get('pieces') or []) != 1:
        return batch, 'unproven'
    token = kills[0]['pieces'][0]
    killed = rr.parse_piece(token)
    if killed != ('unit', nation, 'REGULAR', len(initial)) or len(initial) <= count:
        return batch, 'unproven'
    old_dest = Counter(rc._snap_side_units(batch['truth_before'], dest, side_str))
    new_dest = Counter(rc._snap_side_units(batch['truth'], dest, side_str))
    if new_dest != old_dest + Counter({(nation, 'REGULAR'): count}) or rc._snap_side_units(batch['truth'], region, side_str):
        return batch, 'unproven'
    # Other defender movements/losses could make this an unrelated casualty
    # restoration. The original region's single kill and the restoration
    # must be the only two defender-unit records in this round window.
    for other in batch['records']:
        if other is record or other is kills[0] or not lo <= other.get('line', 0) < hi:
            continue
        tokens = other.get('pieces', []) if other.get('op') == 'kill' else _group_tokens(other.get('obj', ''))
        if any(p and p[0] == 'unit' and _unit_side_str(p[1]) == side_str
               for p in (rr.parse_piece(t) for t in tokens)):
            return batch, 'unproven'
    fixed = dict(batch)
    fixed['records'] = [copy.deepcopy(r) for r in batch['records']]
    for item in fixed['records']:
        if item['line'] == kills[0]['line']:
            item['pieces'] = ['[%d]%s' % (len(initial) - count, re.sub(r'^\[\d+\]', '', token))]
        elif item['line'] == record['line']:
            item['from'] = kills[0]['zone']
            item['_rx_casualty_retreat_correction'] = True
    return fixed, None


def _window_trim(batch, region, side_str, lo=0, hi=None):
    """Siege-trim evidence: units of `side_str` sent from `region` (or its
    stronghold box) back to REINFORCEMENTS â€” or, mistakes, to CASUALTIES
    (spec: tolerate + flag).  Returns (elites_back, regulars_back, to_cas)."""
    import rendered_replay as rr
    eb = rb = 0
    to_cas = False
    for r in batch['records']:
        if r.get('op') != 'move':
            continue
        ln = r.get('line', 0)
        if ln < lo or (hi is not None and ln >= hi):
            continue
        fk = rr.zone_kind(r.get('from', ''))
        tk = rr.zone_kind(r.get('to', ''))
        if fk[0] not in ('region', 'stronghold') or fk[1] != region:
            continue
        if tk[0] not in ('reinf', 'cas'):
            continue
        for t in _group_tokens(r.get('obj', '')):
            p = rr.parse_piece(t)
            if p and p[0] == 'unit' and _unit_side_str(p[1]) == side_str:
                if p[2] == 'ELITE':
                    eb += p[3]
                else:
                    rb += p[3]
                if tk[0] == 'cas':
                    to_cas = True
    return eb, rb, to_cas


def _combat_final_ok(s2, truth, regions):
    """END-TO-END gate: the fully-driven engine must reproduce the client's
    settled per-side unit TOTALS summed over the battle-involved regions
    (advance/retreat move units BETWEEN these regions, so the per-side sum
    is advance-policy-agnostic while still validating every casualty)."""
    import rendered_census as rc
    for side_str in ('FREE_PEOPLES', 'SHADOW'):
        eng = sum(len(rc._eng_side_units(s2, r, side_str)) for r in regions)
        tru = sum(len(rc._snap_side_units(truth, r, side_str))
                  for r in regions)
        if eng != tru:
            return False
    return True


# ---- combat-effect magnitude back-calculation (spec 1.216 item 3) ---------
# Dread and Despair chats an explicit 'ff N' (forfeit N Nazgul leadership);
# Heroic Death cancels hit(s) by a specific figure death (0 = decline, 1 = a
# plain leader dies, level = a companion dies).  Onslaught / Relentless Assault
# self-hits are NOT back-calculated here (the pre-roll self-kills are not
# cleanly separable from the round's roll casualties in the rendered stream) â€”
# those two kinds keep the censused break.
_FF_RE = re.compile(r'\bff\s*(\d+)\b', re.I)


def _ff_chat_count(batch):
    """The 'ff N' forfeit count chatted inside the batch window, or None."""
    for t in _batch_chats(batch):
        m = _FF_RE.search(t)
        if m:
            return int(m.group(1))
    return None


def _owner_figure_deaths(batch, owner_str, lo, hi):
    """(n_plain_leaders, [companion_names]) of `owner_str` KILLED in a battle
    region within line window [lo, hi) â€” the Heroic Death evidence."""
    import rendered_replay as rr
    n_leader = 0
    comps = []
    for r in batch['records']:
        if r.get('op') != 'kill':
            continue
        ln = r.get('line', 0)
        if ln < lo or ln >= hi:
            continue
        if rr.zone_kind(r.get('zone', ''))[0] not in ('region', 'stronghold'):
            continue
        for p in r.get('pieces', []):
            pp = rr.parse_piece(p)
            if not pp:
                continue
            if pp[0] == 'leader' and _unit_side_str(pp[1]) == owner_str:
                n_leader += 1
            elif pp[0] == 'named' and owner_str == 'FREE_PEOPLES' \
                    and not _is_minion_name(pp[1]):
                comps.append(pp[1])
    return n_leader, comps


def _combat_effect_count(batch, ec, owner_str, counts, lo, hi):
    """Best-effort magnitude back-calc for a combat_effect_choice.  Returns a
    legal count (from `counts`) uniquely supported by the batch evidence, else
    None (caller keeps the censused break -> no mislabel)."""
    import rendered_census as rc
    kind = str(ec.get('effect', '')).lower()
    stage = ec.get('stage')
    if kind == 'dread_despair':
        n = _ff_chat_count(batch)
        if n is not None and n in counts:
            return n
        return None
    if kind == 'heroic_death' and stage != 'post':
        n_leader, comps = _owner_figure_deaths(batch, owner_str, lo, hi)
        if comps:
            lv = max(min(3, rc._COMP_LEVEL.get(c, 1)) for c in comps)
            return lv if lv in counts else None
        if n_leader >= 1:
            return 1 if 1 in counts else None
        return 0 if 0 in counts else None
    if kind == 'relentless_assault':
        # DET census P1-a: the RA self-hit sacrifice renders as the
        # owner's units PARKED from a battle region onto the committed
        # card (the same gesture the Onslaught magnitude trusts);
        # K = parked unit count.  No park = no evidence (censused).
        k = _parked_units_count(batch, owner_str, lo, hi)
        if k is not None and k in counts:
            return k
        return None
    return None                            # onslaught: censused here


def _parked_units_count(batch, owner_str, lo, hi):
    """Units owner_str moved from a region onto the committed combat card
    (Current SA/FP Card) in [lo, hi) - the Onslaught / Relentless
    Assault self-sacrifice gesture.  None when no park gesture exists
    (absence of the gesture is NOT evidence of K=0)."""
    import rendered_replay as rr
    n = 0
    seen = False
    for r in batch['records']:
        if r.get('op') != 'move':
            continue
        ln = int(r.get('line', 0) or 0)
        if ln < lo or ln >= hi:
            continue
        fk = rr.zone_kind(r.get('from', ''))
        if fk[0] not in ('region', 'stronghold'):
            continue
        to = str(r.get('to', '')).lower()
        if 'current sa card' not in to and 'current fp card' not in to:
            continue
        for t in _group_tokens(r.get('obj', '')):
            p = rr.parse_piece(t)
            if p and p[0] == 'unit' and _unit_side_str(p[1]) == owner_str:
                n += p[3]
                seen = True
    return n if seen else None


def _onslaught_owner_commit(batch, side_str, lo, hi):
    """W2-008: True when `side_str` committed an ONSLAUGHT combat card in
    the line window [lo, hi) (a '/Onslaught'-titled card moved into a
    Current FP/SA Card slot at a line < hi whose discard, if any, is not
    before lo).  We Come to Kill is a post-roll card too but never a
    self-hit sacrifice."""
    import rendered_census as rc
    for r in batch['records']:
        if r.get('op') != 'move':
            continue
        to = str(r.get('to', '') or '').strip().lower()
        if to not in rc._CUR_CARD_ZONES:
            continue
        obj = str(r.get('obj', '') or '')
        if not obj.strip().lower().endswith('/onslaught'):
            continue
        ln = int(r.get('line', 0) or 0)
        if ln >= hi:
            continue
        dl = rc._card_discard_line(batch, obj, ln)
        if dl is not None and dl < lo:
            continue
        if rc._card_title_side(obj, r.get('to', '')) != side_str:
            continue
        return True
    return False


def _onslaught_post_chunk(batch, owner_str, lo, hi):
    """W2-008: the owner's Onslaught strike roll in [lo, hi) - the chunk
    _combat_chunk_rounds classes 'post' for the owner's side, when the
    owner committed an Onslaught (not We Come to Kill) before it.  None
    unless exactly one such chunk exists."""
    import rendered_census as rc
    atk = batch.get('die_side') or 'SHADOW'
    tag = 'A' if owner_str == atk else 'D'
    found = []
    try:
        rounds = rc._combat_chunk_rounds(batch)
    except Exception:
        return None
    for items in rounds:
        for c, s, cls in items:
            if cls != 'post' or s != tag:
                continue
            ln = int(c.get('line', 0) or 0)
            if ln < lo or ln >= hi:
                continue
            if _onslaught_owner_commit(batch, owner_str, lo, ln):
                found.append(c)
    return found[0] if len(found) == 1 else None


def _onslaught_park_k(batch, owner_str, regs, lo, hi):
    """W2-008: Onslaught magnitude K = the number of Onslaught DICE the
    owner rolled (combat.py rolls one die per self-inflicted hit), cross-
    checked against the PARK gesture - the owner moves the sacrificed
    units from a battle region onto the committed card (Current SA/FP
    Card; units moved back off the card are un-parked): parked Regulars
    absorb 1 hit each, a parked Elite 1 or 2, so the dice count must lie
    in [regs + elites, regs + 2*elites].  Returns (k, reason): reason
    'park' (park + matching dice), 'dice' (no park: the sacrifice was
    rendered as kills and the owner's post-casualty Onslaught roll gives
    K), or None with the counted reason 'no_park' (no gesture and no
    Onslaught roll in [lo, hi)), 'no_roll' (a park with no roll after
    it), 'dice_mismatch' (the roll after the park is not by the owner or
    not a legal hit count for the parked units)."""
    import rendered_replay as rr
    import rendered_census as rc
    n_reg = 0
    n_elite = 0
    seen = False
    park_line = None
    for r in batch['records']:
        if r.get('op') != 'move':
            continue
        ln = int(r.get('line', 0) or 0)
        if ln < lo or ln >= hi:
            continue
        fk = rr.zone_kind(r.get('from', ''))
        tk = rr.zone_kind(r.get('to', ''))
        to = str(r.get('to', '')).lower()
        frm = str(r.get('from', '')).lower()
        sign = 0
        if (fk[0] in ('region', 'stronghold') and fk[1] in regs
                and ('current sa card' in to or 'current fp card' in to)):
            sign = 1
        elif (tk[0] in ('region', 'stronghold') and tk[1] in regs
              and ('current sa card' in frm or 'current fp card' in frm)
              and park_line is not None):
            sign = -1
        if not sign:
            continue
        for t in _group_tokens(r.get('obj', '')):
            p = rr.parse_piece(t)
            if p and p[0] == 'unit' and _unit_side_str(p[1]) == owner_str:
                seen = True
                if p[2] == 'REGULAR':
                    n_reg += sign * p[3]
                else:
                    n_elite += sign * p[3]
                if park_line is None:
                    park_line = ln
    if not seen:
        c = _onslaught_post_chunk(batch, owner_str, lo, hi)
        if c is None:
            return None, 'no_park'
        return len(c.get('ints') or []), 'dice'
    n_reg = max(0, n_reg)
    n_elite = max(0, n_elite)
    ap = rc._combat_atk_player(batch)
    atk = batch.get('die_side') or 'SHADOW'
    own_is_atk = (owner_str == atk)
    for r in (batch.get('rng') or []):
        if r.get('kind') != 'roll_num':
            continue
        ln = int(r.get('line', 0) or 0)
        if ln <= park_line or ln >= hi:
            continue
        if (r.get('player') == ap) != own_is_atk:
            return None, 'dice_mismatch'
        k = len(r.get('ints') or [])
        if not (n_reg + n_elite <= k <= n_reg + 2 * n_elite):
            return None, 'dice_mismatch'
        return k, 'park'
    return None, 'no_roll'


def _alloc_downgrades_ordered(batch, side_str, regs, lo, hi, hits):
    """W2-008: the allocation `side_str` chose for the round's roll HITS
    when the same window also holds its Onslaught self-sacrifice (a
    second, later loss event rendered with the same kill / cas-return
    gestures, so a whole-window replacement count over-labels the
    allocation).  Walk the side's unit losses in the battle regions in
    log order - a Regular removed = 1 hit, an Elite removed = 1 hit when
    a replacement Regular is paired with it (downgrade) else 2 - and stop
    once `hits` are absorbed.  Returns (downgrades_in_allocation,
    hits_seen): hits_seen < hits means the window shows fewer losses than
    the roll inflicted (caller must not label)."""
    import rendered_replay as rr
    ev = []
    seq = 0
    for r in batch['records']:
        ln = int(r.get('line', 0) or 0)
        if ln < lo or (hi is not None and ln >= hi):
            continue
        op = r.get('op')
        toks = []
        kind = None
        if op == 'kill':
            zone = rr.zone_kind(str(r.get('zone', '') or ''))
            if zone[0] not in ('region', 'stronghold') or zone[1] not in regs:
                continue
            for pc in (r.get('pieces') or []):
                toks.extend(_group_tokens(str(pc)))
            kind = 'loss'
        elif op == 'move':
            fk = rr.zone_kind(r.get('from', ''))
            tk = rr.zone_kind(r.get('to', ''))
            if (fk[0] in ('region', 'stronghold') and fk[1] in regs
                    and (tk[0] == 'cas' or (tk[0] == 'reinf' and lo > 0))):
                kind = 'loss'
            elif (fk[0] in ('cas', 'reinf')
                  and tk[0] in ('region', 'stronghold') and tk[1] in regs):
                kind = 'rep'
            else:
                continue
            toks = _group_tokens(r.get('obj', ''))
        else:
            continue
        for t in toks:
            p = rr.parse_piece(t)
            if not (p and p[0] == 'unit' and _unit_side_str(p[1]) == side_str):
                continue
            for _ in range(int(p[3] or 1)):
                seq += 1
                if kind == 'rep':
                    if p[2] == 'REGULAR':
                        ev.append((ln, seq, 'rep'))
                else:
                    ev.append((ln, seq, 'R' if p[2] == 'REGULAR' else 'E'))
    ev.sort()
    paired = set()
    used_rep = set()
    for i, (ln, sq, kd) in enumerate(ev):
        if kd != 'rep':
            continue
        j = None
        for k2 in range(i - 1, -1, -1):
            if ev[k2][2] == 'E' and k2 not in paired:
                j = k2
                break
        if j is None:
            for k2 in range(i + 1, len(ev)):
                if ev[k2][2] == 'E' and k2 not in paired:
                    j = k2
                    break
        if j is not None:
            paired.add(j)
            used_rep.add(i)
    seen = 0
    downs = 0
    for i, (ln, sq, kd) in enumerate(ev):
        if seen >= hits:
            break
        if kd == 'R':
            seen += 1
        elif kd == 'E':
            if i in paired:
                seen += 1
                downs += 1
            else:
                seen += 2
    return downs, seen


def _effect_target_pick(batch, ec, la, owner_str, lo, hi, stats):
    """1.254 target-bearing combat_effect_choice matcher (DET census
    P1-a).  ec['targets'] (WoP/BB/BoW: which enemy figure) or
    ec['pairs'] (Heroic Death: (count, own figure)).  The victim is
    NAMED in the batch's kill lines (any zone - region, guide box,
    cleanup group); 'leader' = a plain-Leader kill.  Unique evidence ->
    the la index of the target-carrying action; None (counted)
    otherwise - never a guess."""
    import rendered_census as rc
    import rendered_replay as rr

    def _norm(n):
        return str(n or '').strip().lower().replace('-', ' ')

    if ec.get('pairs'):
        # Heroic Death: the sacrificed OWN figure (leaders + companions
        # of owner_str, in-window)
        n_leader, comps = _owner_figure_deaths(batch, owner_str, lo, hi)
        if comps:
            if len(set(comps)) > 1:
                stats['drop_reasons']['cf_effect_target_unresolved'] += 1
                return None
            nm = comps[0]
            want = (min(3, rc._COMP_LEVEL.get(nm, 1)), _norm(nm))
        elif n_leader >= 1:
            want = (1, 'leader')
        else:
            want = (0, None)
        return _match_index(
            la, lambda a: a.action_type == 'combat_effect_choice'
            and int(a.params.get('count', -2)) == want[0]
            and ((_norm(a.params.get('target')) == want[1])
                 if want[1] is not None
                 else a.params.get('target') is None))
    # targets menu (WoP/BB/BoW): the named victim among the offered
    offered = {_norm(a.params.get('target'))
               for a in la
               if a.action_type == 'combat_effect_choice'
               and a.params.get('target') is not None}
    found = set()
    for r in batch['records']:
        if r.get('op') != 'kill':
            continue
        ln = int(r.get('line', 0) or 0)
        if ln < lo or ln >= hi:
            continue
        for p in (r.get('pieces') or ()):
            pp = rr.parse_piece(p)
            if not pp:
                continue
            if pp[0] == 'named' and _norm(pp[1]) in offered:
                found.add(_norm(pp[1]))
            elif pp[0] == 'leader' and 'leader' in offered:
                found.add('leader')
    if len(found) != 1:
        stats['drop_reasons']['cf_effect_target_unresolved'] += 1
        return None
    vict = found.pop()
    return _match_index(
        la, lambda a: a.action_type == 'combat_effect_choice'
        and _norm(a.params.get('target')) == vict)


def _inject_named_cc(st, dside, cid, la, stats):
    """DET census P2-a: the combat-card line NAMES the committed card but
    the reconstructed hand lacks it (mid-battle draw / ledger gap).  The
    log is the hand's ground truth at that instant - inject the card
    into the deciding side's hand and re-enumerate the pending menu.
    Side mismatch or enumeration failure returns the menu unchanged (the
    caller's drop then fires as before)."""
    import ai
    from game_state import Side
    want_pfx = 'SHADOW' if dside == Side.SHADOW else 'FREE_PEOPLES'
    if not str(cid).startswith(want_pfx):
        return la
    hand = st.hands[dside].hand
    if cid in hand:
        return la
    hand.append(cid)
    try:
        la2 = ai.get_legal_actions(st, dside)
    except Exception:
        hand.remove(cid)
        return la
    stats['detected']['cc_card_injected'] += 1
    return la2


# pending-combat phase -> keep-kind family (EMISSION COMPLETENESS table)
_PEND_FAM = {
    'attacker_combat_card': 'combat_card',
    'defender_combat_card': 'combat_card',
    'casualty_allocation': 'casualty_alloc',
    'attacker_continue': 'attacker_continue',
    'defender_retreat': 'defender_retreat',
    'defender_siege_choice': 'defender_siege_choice',
    'combat_round_result': 'siege_press',
    'siege_trim': 'siege_trim',
    'victor_advance': 'victor_advance',
    'fp_shine': 'fp_shine',
    'combat_effect_choice': 'combat_effect',
}


class _RecordedCombatDiceError(RuntimeError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


class _RecordedCombatDice:
    """Consume captured main/re-roll dice without crossing owners or rounds.

    RB30: a Leader re-roll can contain fewer dice than the available misses.
    An omitted re-roll is an empty recorded choice, never another player's
    dice or invented misses. Main rolls must match the requested strength.
    Special card rolls require a separately identified source; this bounded
    consumer refuses those windows instead of guessing from the flat stream.
    """
    def __init__(self, rounds, attacker_side, stats, *, allow_special=False):
        from game_state import Side
        self.owners = {attacker_side: 'A',
                       Side.FREE_PEOPLES if attacker_side == Side.SHADOW
                       else Side.SHADOW: 'D'}
        self.rounds = []
        for items in rounds:
            values = {}
            for chunk, owner, phase in items:
                values.setdefault((owner, phase), []).extend(chunk['ints'])
            self.rounds.append(values)
        self.index, self.seen, self.stats = 0, set(), stats
        self.special_actor = None
        self.special_phase = None
        self.allow_special = allow_special

    def _close_round(self, *, allow_pending_post=False):
        if self.index >= len(self.rounds):
            raise _RecordedCombatDiceError('missing_round')
        if any(key not in self.seen and values and not (allow_pending_post and key[1] == 'post')
               for key, values in self.rounds[self.index].items()):
            raise _RecordedCombatDiceError('unused_recorded_dice')
        has_main = any(phase == 'main' for _owner, phase in self.rounds[self.index])
        if (has_main or not self.allow_special) and not {('A', 'main'), ('D', 'main')} <= self.seen:
            raise _RecordedCombatDiceError('missing_main_request')

    def draw(self, n, ctx='', side=None):
        if self.special_phase is not None:
            side = self.special_actor
        if side not in self.owners:
            raise _RecordedCombatDiceError('unknown_roll_owner')
        phase = self.special_phase or {'combat roll': 'main', 'leader reroll': 're'}.get(ctx)
        if phase is None:
            raise _RecordedCombatDiceError('unknown_roll_phase')
        key = (self.owners[side], phase)
        if ((phase == 'main' and key in self.seen) or
                (phase == 'pre' and any(p == 'main' for _, p in self.seen))):
            self._close_round()
            self.index += 1
            self.seen = set()
        if self.index >= len(self.rounds):
            raise _RecordedCombatDiceError('missing_round')
        values = self.rounds[self.index]
        supported = ('main', 're', 'pre', 'post') if self.allow_special else ('main', 're')
        if any(p not in supported for _owner, p in values):
            raise _RecordedCombatDiceError('special_roll_window')
        if key in self.seen:
            raise _RecordedCombatDiceError('duplicate_roll_request')
        if phase == 're' and (key[0], 'main') not in self.seen:
            raise _RecordedCombatDiceError('reroll_before_main')
        recorded = values.get(key, [])
        if phase in ('main', 'pre', 'post') and len(recorded) != n:
            raise _RecordedCombatDiceError('main_count')
        if phase == 're' and len(recorded) > n:
            raise _RecordedCombatDiceError('reroll_count')
        self.seen.add(key)
        if phase == 're' and len(recorded) < n:
            self.stats['detected']['cf_rng_short_reroll'] += 1
        return list(recorded)

    def finish(self):
        if not self.rounds:
            raise _RecordedCombatDiceError('missing_round')
        self._close_round()
        if self.index != len(self.rounds) - 1:
            raise _RecordedCombatDiceError('unused_recorded_round')

    def validate_completed(self, rounds_fought, *, allow_pending_post=False):
        # Check before a completed round exposes casualty/continue menus.
        # Finding an unconsumed re-roll only at final completion would
        # leave already-emitted post-roll labels on an unverified state.
        if rounds_fought > 0:
            if self.index < rounds_fought - 1:
                raise _RecordedCombatDiceError('missing_recorded_round')
            if self.index == rounds_fought - 1:
                self._close_round(allow_pending_post=allow_pending_post)


def _shine_evidence_refusal(state, evidence, action):
    """An actual White Rider choice must be bound before cards and rolls."""
    import ai
    from game_state import Side
    pc = getattr(state, 'pending_combat', None) or {}
    if (not isinstance(evidence, dict) or evidence.get('schema') != 'observed_shine_v1'
            or action.action_type != 'shine' or pc.get('phase') != 'fp_shine'
            or pc.get('deciding_side') != Side.FREE_PEOPLES
            or int(pc.get('rounds_fought', 0)) != 0
            or evidence.get('round') != 0
            or evidence.get('side') != Side.FREE_PEOPLES.name
            or evidence.get('owner_binding') not in ('player', 'fp_owned_figure')):
        return 'shine_observed_choice_missing'
    try:
        if not (type(evidence['line']) is int and
                evidence['battle_start'] < evidence['line'] < evidence['before_card_or_roll']):
            return 'shine_observed_timing_invalid'
        role = 'attacker' if pc.get('attacker_side') == Side.FREE_PEOPLES else 'defender'
        import rendered_replay as rr
        if evidence.get('region') != rr.sanitize(pc[role + '_region']):
            return 'shine_observed_region_invalid'
        proof = ai.clone_state(state)
        proof.pending_combat.pop('shine_asked', None)
        import combat
        old_hook = combat.FINAL_HITS_HOOK
        try:
            combat.FINAL_HITS_HOOK = None
            if not ai._maybe_shine_phase(proof, proof.pending_combat):
                return 'shine_participants_unproven'
        finally:
            combat.FINAL_HITS_HOOK = old_hook
    except (KeyError, TypeError, ValueError):
        return 'shine_observed_choice_invalid'
    return None


def _match_shine_gesture(state, batch, legal, cards, recorded_rounds):
    """Only a unique participating Gandalf flip before the first commitment."""
    import rendered_replay as rr
    from game_state import Side
    pc = state.pending_combat or {}
    start = int(batch.get('start_line', 0))
    rolls = [int(chunk['line']) for items in recorded_rounds for chunk, _, _ in items]
    commits = [int(cards[0].get(slot + '_ln')) for slot in ('fp', 'sh')
               if cards and cards[0].get(slot + '_ln') is not None]
    if not rolls:
        return None, None, 'shine_battle_boundary_unproven'
    first_roll = min(rolls)
    upper = min([first_roll] + commits)
    flips = [r for r in batch.get('records', ()) if r.get('op') == 'flip'
             and re.sub('[^a-z]', '', str(r.get('obj', '')).lower()) == 'gandalfthewhite'
             and start < int(r.get('line', 0)) < first_roll]
    if len(flips) != 1:
        return None, None, 'shine_missing_or_ambiguous_flip'
    record = flips[0]
    line = int(record['line'])
    if line >= upper:
        return None, None, 'shine_flip_after_card_commit'
    role = 'attacker' if pc.get('attacker_side') == Side.FREE_PEOPLES else 'defender'
    region = rr.sanitize(pc.get(role + '_region', ''))
    kind, observed_region = rr.zone_kind(record.get('region') or record.get('zone') or '')
    if kind not in ('region', 'stronghold') or observed_region != region:
        return None, None, 'shine_flip_region_unproven'
    tag = 'A' if role == 'attacker' else 'D'
    players = {str(chunk.get('player', '')).strip().casefold()
               for items in recorded_rounds for chunk, owner, _ in items
               if owner == tag and chunk.get('player')}
    actor = record.get('player')
    if actor is not None and (len(players) != 1 or str(actor).strip().casefold() not in players):
        return None, None, 'shine_flip_actor_unproven'
    found = [i for i, action in enumerate(legal) if action.action_type == 'shine']
    if len(found) != 1:
        return None, None, 'shine_action_unmatched'
    evidence = {'schema': 'observed_shine_v1', 'round': 0, 'line': line,
                'battle_start': start, 'before_card_or_roll': upper,
                'region': region, 'side': Side.FREE_PEOPLES.name,
                'owner_binding': 'player' if actor is not None else 'fp_owned_figure'}
    refusal = _shine_evidence_refusal(state, evidence, legal[found[0]])
    return (None, None, refusal) if refusal else (found[0], evidence, 'observed_gandalf_flip')


def _emit_shine_sample(state, side, legal, index, turn, evidence):
    missing = object()
    previous = getattr(state, '_rx_shine_evidence', missing)
    state._rx_shine_evidence = dict(evidence)
    try:
        return _emit_sample(state, side, legal, index, 'fp_shine', turn)
    finally:
        if previous is missing:
            del state._rx_shine_evidence
        else:
            state._rx_shine_evidence = previous


def _observed_round_choice_lower(batch, wins, cards, round_index):
    """Effects follow this round's revealed commitments, not earlier losses."""
    previous = wins[round_index - 1]['last'] if round_index else (batch.get('start_line', 0) - 1)
    current = cards[round_index] if round_index < len(cards) else {}
    return max([previous] + [int(current.get(slot + '_ln') or 0) for slot in ('fp', 'sh')])


def _emit_combat_family(batch, stats):
    global _RX_ALT_HIDX
    import ai
    import random
    import game_state as gs
    import rendered_census as rc
    import rendered_replay as rr
    from game_state import Side
    if batch['kind'] not in ('combat', 'combat?'):
        return []
    chunks = [r for r in batch['rng'] if r['kind'] == 'roll_num']
    _strike9f = _card_strike(batch)   # W1-001: shared strike predicate
    wins = _combat_round_windows(batch)
    nrounds = len(wins)
    no_dice = not chunks
    if no_dice:
        # card-commit-only fallback (a combat? batch whose roll rendered
        # outside the batch): round-1 card commits are still pre-roll
        # faithful; everything after the (fallback-dice) roll is not.
        wins = [{'first': 10 ** 9, 'last': 10 ** 9, 'next_first': 10 ** 9}]
        nrounds = 1
    carded = False
    try:
        carded = rc._batch_has_combat_card(batch)
    except Exception:
        pass
    try:
        # Combat uses the same typed figures, turn and public ledgers as
        # other extraction paths. Raw census reconstruction leaves named
        # Companions as political Leaders and omits these decision fields.
        st = _recon_state(batch['truth_before'], batch.get('turn'))
        plan = rc._plan_combat(
            st, batch, expected_dice=(None if (carded or no_dice)
                                      else len(chunks[0]['ints'])),
            chats=_batch_chats(batch))
    except Exception:
        stats['drop_reasons']['cf_recon'] += 1
        return []
    if not plan:
        stats['drop_reasons']['cc_no_plan'] += 1
        return []
    atk, dfn, _sieg = plan
    regs = set(rr._battle_regions(batch)) | {atk, dfn}
    side = Side.SHADOW if batch['die_side'] == 'SHADOW' else Side.FREE_PEOPLES
    atk_str = batch['die_side']
    dfn_str = 'FREE_PEOPLES' if atk_str == 'SHADOW' else 'SHADOW'
    atk_tag = 'sh' if side == Side.SHADOW else 'fp'
    dfn_tag = 'fp' if atk_tag == 'sh' else 'sh'
    _lead = None
    if not no_dice:
        # W2-008: re-segment the rounds with the engine-state
        # leadership bound (a chunk longer than a side's Leadership
        # can never be its re-roll)
        try:
            _lead = _battle_leadership(st, atk, dfn, atk_str)
            wins = _combat_round_windows(batch, _lead)
            nrounds = len(wins)
        except Exception:
            pass
    cards, _card_anom, _anom_slots = _round_cards(batch, wins)
    # 1.267: Grond / The Fighting Uruk-hai EVENT -> 3 mandatory rounds
    _mand9 = _mandatory_siege_rounds(batch, cards) if not _strike9f else 0
    if _mand9:
        stats['detected']['mandatory_siege_rounds'] += 1
    st.dice = {Side.FREE_PEOPLES: [], Side.SHADOW: []}
    d = gs.ActionDie(gs.SH_FACES if side == Side.SHADOW else gs.FP_FACES, side)
    d.current = gs.ActionDieFace.ARMY
    d.used = False
    st.dice[side] = [d]
    _install_ledger_hands(st, batch)
    try:
        legal = ai.get_legal_actions(st, side)
    except Exception:
        stats['drop_reasons']['cc_legal_exc'] += 1
        return []
    if _strike9f:
        # CARD STRIKE (Dreadful Spells class): no battle happened - no
        # drive, no attack, no cc rounds; the victim ALLOCATION is the
        # only combat-family decision, via the gesture path
        emitted = []
        broke = None
        retreat_dests = set()
        _drive_fx9 = set()
        round_roll_hits = {}
        s2 = st
        keep = emitted   # ALIAS: gesture-path keep.append lands in emitted
        # (telemetry: count real card strikes only â€” the bytecode
        #  reconstruction had mis-hoisted this outside the if)
        stats['detected']['cf_card_strike'] += 1

    atk_act = None
    if not _strike9f:
        atk_act = next(
            (a for a in legal
             if a.action_type == 'army_attack'
             and rr.sanitize(str(a.params.get('from', ''))) == atk
             and rr.sanitize(str(a.params.get(
                 'to', a.params.get('to_region', '')))) == dfn),
            None)
    if atk_act is None and not _strike9f:
        atk_act = next(
            (a for a in legal
             if a.action_type == 'army_attack'
             and rr.sanitize(str(a.params.get(
                 'to', a.params.get('to_region', '')))) == dfn),
            None)
    if atk_act is None and not _strike9f:
        stats['drop_reasons']['cc_no_attack'] += 1
        return []
    emitted = []
    broke = 'cf_card_strike' if _strike9f else None
    retreat_dests = set()
    _drive_fx9 = set()
    round_roll_hits = {}
    import combat as _combat_rng
    _recorded_rounds = rc._combat_chunk_rounds(batch, _lead) if not no_dice else []
    _recorded_rng = _RecordedCombatDice(_recorded_rounds, side, stats,
        allow_special=bool(getattr(ai, '_COMBAT_ROUND_CONTEXT_ON', False)))
    _round_choice_lines = {}
    ori, orc = random.randint, random.choice
    _original_combat_d6 = _combat_rng.roll_d6_batch
    _original_pre_attack = _combat_rng._apply_pre_combat_for_side
    import onslaught_choices as _onslaught_rng
    _original_onslaught_advance = _onslaught_rng._advance
    _original_final_hits_hook = _combat_rng.FINAL_HITS_HOOK
    def _unbound_combat_roll(*_args, **_kwargs):
        raise _RecordedCombatDiceError('unbound_die_request')
    random.randint = _unbound_combat_roll
    _combat_rng.roll_d6_batch = _recorded_rng.draw
    def _recorded_pre_attack(combat, is_attacker):
        previous = (_recorded_rng.special_actor, _recorded_rng.special_phase)
        _recorded_rng.special_actor = combat.attacker_side if is_attacker else combat.defender_side
        _recorded_rng.special_phase = 'pre'
        try:
            return _original_pre_attack(combat, is_attacker)
        finally:
            _recorded_rng.special_actor, _recorded_rng.special_phase = previous
    if getattr(ai, '_COMBAT_ROUND_CONTEXT_ON', False):
        _combat_rng._apply_pre_combat_for_side = _recorded_pre_attack
    def _recorded_onslaught_advance(state, pc):
        payment = pc.get('_onslaught_payment') or {}
        owner = payment.get('owner')
        previous = (_recorded_rng.special_actor, _recorded_rng.special_phase)
        _recorded_rng.special_actor = pc.get('attacker_side' if owner == 'atk' else 'defender_side')
        _recorded_rng.special_phase = 'post'
        try:
            return _original_onslaught_advance(state, pc)
        finally:
            _recorded_rng.special_actor, _recorded_rng.special_phase = previous
    if getattr(ai, '_COMBAT_ROUND_CONTEXT_ON', False):
        _onslaught_rng._advance = _recorded_onslaught_advance
    # This local public-action drive has authenticated physical dice. Keep
    # outcome-dependent choices at their real observed stage even when a
    # surrounding teacher uses a final-hit override. Restore that hook below.
    if getattr(ai, '_COMBAT_ROUND_CONTEXT_ON', False):
        _combat_rng.FINAL_HITS_HOOK = None
    random.choice = lambda s2: s2[-1]
    _declared_rear = None
    try:
        s2 = st if _strike9f else ai.apply_action(st, atk_act)
        if not _strike9f:
            # The executor captures this BEFORE attack-induced politics.
            # Keep that immutable declaration through every recovery round;
            # never select a new guard from the post-casualty Army.
            _declared_rear = (getattr(s2, 'pending_combat', None) or {}).get(
                'attacker_rearguard')
        for _ in range(80):
            if _strike9f:
                break
            pc = getattr(s2, 'pending_combat', None)
            if pc is None:
                break
            ph = pc.get('phase')
            ds = pc.get('deciding_side')
            dside = ds if isinstance(ds, Side) else side
            r_i = int(pc.get('rounds_fought', 0) or 0)
            _recorded_rng.validate_completed(r_i, allow_pending_post=(
                bool(getattr(ai, '_COMBAT_ROUND_CONTEXT_ON', False))
                and ph in ('casualty_allocation', 'combat_effect_choice')
                and not pc.get('_onslaught_done')))
            if ph == 'victor_advance' and not no_dice:
                _recorded_rng.finish()
            # no_dice fallback: the ROUND-1 card commits are the whole
            # point (state is still pre-roll faithful) â€” let the r_i==0
            # card phases run the commit-match/emit block below and
            # break at anything after (the fallback-dice roll and
            # everything past it are not faithful)
            if no_dice and not (r_i == 0
                                and ph in ('attacker_combat_card',
                                           'defender_combat_card')):
                broke = 'cf_no_dice'
                break
            try:
                la = ai.get_legal_actions(s2, dside)
            except Exception:
                broke = 'cf_legal_exc'
                break
            if not la:
                broke = 'cf_no_actions'
                break
            if len(la) >= 2 and ph in _PEND_FAM:
                _pend_note(stats, _PEND_FAM[ph], 'open')
            if ph in ('attacker_combat_card', 'defender_combat_card'):
                if r_i >= len(wins):
                    broke = 'cf_round_overrun'
                    break
                slot = atk_tag if ph == 'attacker_combat_card' else dfn_tag
                human = cards[r_i][slot]
                if cards[r_i].get(slot + '_ambiguous'):
                    broke = 'cc_commit_ambiguous'
                    break
                if human is not None and not _combat_commit_board_matches(
                        s2, batch, cards[r_i].get(slot + '_ln')):
                    broke = 'cc_commit_board_mismatch'
                    break
                cc = [a for a in la if a.action_type in
                      ('combat_card_play', 'combat_card_none')]
                if human is not None:
                    hidx = _match_index(la, lambda a: a.action_type ==
                                        'combat_card_play' and
                                        str(a.params.get('card_id')) ==
                                        str(human))
                    if hidx is None:
                        # DET census P2-a (log = truth): the rendered
                        # line NAMES the committed card - a hand-ledger
                        # gap must not drop the decision.  Inject the
                        # named card, re-enumerate; still no offer ->
                        # drop as before.
                        la = _inject_named_cc(s2, dside, human, la,
                                              stats)
                        cc = [a for a in la if a.action_type in
                              ('combat_card_play', 'combat_card_none')]
                        hidx = _match_index(
                            la, lambda a: a.action_type ==
                            'combat_card_play' and
                            str(a.params.get('card_id')) == str(human))
                    if hidx is None:        # not offerable even injected
                        broke = 'cc_card_absent'
                        break
                else:
                    hidx = _match_index(
                        la, lambda a: a.action_type == 'combat_card_none')
                    if hidx is None:
                        broke = 'cf_none_missing'
                        break
                if len(cc) >= 2:            # a real choice -> emit
                    stats['detected']['combat_card'] += 1
                    emitted.append(('combat_card', r_i, _emit_sample(
                        s2, dside, la, hidx, 'combat_card', batch['turn'])))
                s2 = ai.apply_action(s2, la[hidx])
                continue

            if ph == 'casualty_allocation':
                w = wins[r_i - 1] if 0 < r_i <= len(wins) else wins[-1]
                loser = 'SHADOW' if dside == Side.SHADOW else 'FREE_PEOPLES'
                if pc.get('_rx_onslaught_ordinary_round') != r_i - 1:
                    pc.pop('_rx_onslaught_ordinary_end', None)
                    pc.pop('_rx_onslaught_ordinary_round', None)
                if (getattr(ai, '_COMBAT_ROUND_CONTEXT_ON', False)
                        and pc.get('_observed_round_completed') and pc.get('_onslaught_done')
                        and 0 < r_i <= len(_recorded_rounds)):
                    # Only the additional attack's victim pays here. Earlier
                    # ordinary deaths, self-payments and a later attacker
                    # siege-press sacrifice are separate obligations.
                    owner_tag = 'D' if dside == pc['attacker_side'] else 'A'
                    posts = [chunk for chunk, owner, stage in _recorded_rounds[r_i - 1]
                             if owner == owner_tag and stage == 'post']
                    if len(posts) != 1:
                        broke = 'cf_onslaught_missing_unique_post_allocation'
                        break
                    other_hits = pc.get('def_hits' if dside == pc['attacker_side'] else 'atk_hits', 0)
                    if other_hits:
                        broke = 'cf_onslaught_multiple_post_victims'
                        break
                    try:
                        events = _casualty_gesture_operations(batch, loser, regs,
                            int(posts[0]['line']) + 1, w['next_first'])
                        successor, allocations = _drive_casualty_gestures(s2, dside, events)
                    except _CasualtyGestureError as error:
                        broke = 'cas_operation:' + str(error)
                        break
                    for before, menu, chosen in allocations:
                        stats['detected']['casualty_alloc'] += 1
                        emitted.append(('casualty_alloc', r_i, _emit_sample(
                            before, dside, menu, chosen, 'casualty_alloc', batch['turn'])))
                    stats['detected']['cf_onslaught_post_allocation'] += 1
                    s2 = successor
                    continue
                if (getattr(ai, '_COMBAT_ROUND_CONTEXT_ON', False)
                        and pc.get('_observed_round_completed')
                        and not pc.get('_onslaught_done')
                        and not pc.get('_rx_onslaught_ordinary_end')
                        and 0 < r_i <= len(_recorded_rounds)):
                    from cards import get_card as _get_round_card
                    p9 = pc.get('_p9') or {}
                    owners = []
                    for owner, tag in (('atk', 'A'), ('def', 'D')):
                        card = _get_round_card(p9.get(owner + '_card'))
                        if (card is not None and str(card.combat_name).lower() == 'onslaught'
                                and not p9.get(owner + '_cancelled')):
                            owners.append(tag)
                    items = _recorded_rounds[r_i - 1]
                    post = [chunk for chunk, owner, stage in items
                            if owner in owners and stage == 'post']
                    ordinary = [int(chunk['line']) for chunk, owner, stage in items
                                if stage in ('main', 're')]
                    if len(owners) == 1 and len(post) == 1 and ordinary:
                        try:
                            successor, allocations, _ = _drive_onslaught_ordinary_prefix(
                                s2, batch, regs, max(ordinary) + 1, int(post[0]['line']))
                        except _CasualtyGestureError as error:
                            broke = 'cas_operation:' + str(error)
                            break
                        for before, menu, chosen in allocations:
                            stats['detected']['casualty_alloc'] += 1
                            actor = before.pending_combat['deciding_side']
                            emitted.append(('casualty_alloc', r_i, _emit_sample(
                                before, actor, menu, chosen, 'casualty_alloc', batch['turn'])))
                        stats['detected']['cf_onslaught_ordinary_prefix'] += 1
                        s2 = successor
                        continue
                if _recovery_allocation_lines(batch, regs, w['first'], w['next_first'], atk_str) is None:
                    if (getattr(ai, '_COMBAT_ROUND_CONTEXT_ON', False)
                            and pc.get('_observed_round_completed')
                            and 0 < r_i <= len(_recorded_rounds)):
                        ordinary_lines = [int(chunk['line']) for chunk, owner, stage
                            in _recorded_rounds[r_i - 1] if stage in ('main', 're')]
                        if ordinary_lines:
                            try:
                                recovered, cleanup_proof = _drive_commuting_ordinary_cleanup(
                                    s2, batch, regs, max(ordinary_lines) + 1, w['next_first'])
                            except _CasualtyGestureError as error:
                                stats['drop_reasons']['cf_' + str(error)] += 1
                            else:
                                stats['detected']['cf_commuting_ordinary_cleanup'] += 1
                                stats['detected']['cf_suppressed_reordered_ordinary_rows'] += \
                                    cleanup_proof['suppressed_ordinary_rows']
                                s2 = recovered
                                continue
                    # A defender may physically clean up its forced losses
                    # early. Prove that there was no defender choice before
                    # accepting this transcript; real choices retain RB30 order.
                    forced_cleanup = False
                    if (pc.get('alloc_stage') == 'attacker'
                            and not any((pc.get('_p9') or {}).get(key)
                                        for key in ('atk_card', 'def_card'))):
                        try:
                            proof = ai.clone_state(s2)
                            proof.pending_combat.update(alloc_stage='defender',
                                deciding_side=pc['defender_side'])
                            proof._casualty_operations = []
                            defender_events = _casualty_gesture_operations(
                                batch, dfn_str, regs, w['first'], w['next_first'])
                            _, proof_rows = _drive_casualty_gestures(
                                proof, pc['defender_side'], defender_events)
                            forced_cleanup = not proof_rows
                        except (ValueError, RuntimeError):
                            pass
                    if not forced_cleanup:
                        broke = 'cas_owner_order_ambiguous'
                        break
                    stats['detected']['cf_forced_defender_cleanup_early'] += 1
                role = 'atk' if pc.get('alloc_stage') == 'attacker' else 'def'
                round_roll_hits[(r_i, role)] = int(pc.get(role + '_hits', 0) or 0)
                # These windows contain a separate loss obligation; a total
                # replacement count cannot establish which decision it paid.
                if (_onslaught_owner_commit(batch, loser, w['first'], w['next_first'])
                        or (pc.get('is_siege') and loser == atk_str and nrounds > r_i
                            and not (_mand9 and r_i < _mand9))):
                    broke = 'cas_separate_loss_window'
                    break
                try:
                    events = _casualty_gesture_operations(
                        batch, loser, regs, w['first'], w['next_first'])
                    s2_next, allocations = _drive_casualty_gestures(s2, dside, events)
                except _CasualtyGestureError as error:
                    broke = 'cas_operation:' + str(error)
                    break
                for before, menu, chosen in allocations:
                    stats['detected']['casualty_alloc'] += 1
                    emitted.append(('casualty_alloc', r_i, _emit_sample(
                        before, dside, menu, chosen, 'casualty_alloc', batch['turn'])))
                if not allocations:
                    stats['detected']['cf_alloc_forced'] += 1
                s2 = s2_next
                continue

            if ph == 'attacker_continue':
                w = wins[min(max(r_i, 1), len(wins)) - 1]
                if nrounds == 1:
                    batch, correction_error = _correct_regular_retreat(
                        batch, s2, dfn, dfn_str, w['first'], w['next_first'])
                    if correction_error:
                        broke = 'cf_retreat_correction_unproven'
                        break
                rkind, _rd = _window_retreat(batch, dfn, dfn_str,
                                             w['first'], w['next_first'])
                if rkind == 'ambiguous':
                    broke = 'cf_retreat_chain_ambiguous'
                    break
                more = nrounds > r_i
                want = ('combat_continue' if (more or rkind)
                        else 'combat_cease')
                hidx = _match_index(la, lambda a: a.action_type == want)
                if hidx is None:
                    broke = 'cf_continue_unmatched'
                    break
                if len(la) >= 2:
                    emitted.append(('attacker_continue', r_i, _emit_sample(
                        s2, dside, la, hidx, 'attacker_continue',
                        batch['turn'])))
                s2 = ai.apply_action(s2, la[hidx])
                continue

            if ph in ('defender_retreat', 'defender_siege_choice'):
                if ph == 'defender_siege_choice':
                    lo, hi = 0, wins[0]['first']     # pre-round-1 window
                else:
                    w = wins[min(max(r_i, 1), len(wins)) - 1]
                    lo, hi = w['first'], w['next_first']
                rkind, rdest = _window_retreat(batch, dfn, dfn_str, lo, hi)
                if rkind == 'ambiguous':
                    broke = 'cf_retreat_chain_ambiguous'
                    break
                if rkind == 'siege':
                    hidx = _match_index(la, lambda a: a.action_type ==
                                        'retreat_into_siege')
                elif rkind == 'region' and ph == 'defender_retreat':
                    hidx = _match_index(la, lambda a: a.action_type ==
                                        'combat_retreat' and
                                        rr.sanitize(str(a.params.get(
                                            'to_region', ''))) == rdest)
                    if hidx is not None:
                        retreat_dests.add(rdest)
                elif nrounds > r_i or (ph == 'defender_siege_choice'
                                       and not no_dice):
                    hidx = _match_index(la, lambda a: a.action_type ==
                                        'combat_no_retreat')
                else:
                    broke = 'cf_retreat_ambiguous'
                    break
                if hidx is None:
                    broke = 'cf_retreat_unmatched'
                    break
                if len(la) >= 2:
                    emitted.append((ph, r_i, _emit_sample(
                        s2, dside, la, hidx, ph, batch['turn'])))
                s2 = ai.apply_action(s2, la[hidx])
                continue

            if ph == 'combat_round_result':      # siege: press or stop
                more = nrounds > r_i
                want = 'siege_press' if more else 'siege_no_press'
                if _mand9 and r_i < _mand9:
                    # 1.267: siege round 1..2 under Grond / The Fighting
                    # Uruk-hai is MANDATORY - no press decision, no Elite
                    # cost.  The drive opened the battle by army_attack
                    # (the engine pending carries no mandatory_rounds),
                    # so continue it with the executor's own no-cost
                    # press (no_reduce) when the log shows a further
                    # round, else end it (only an elimination stops a
                    # mandatory round early).  No row either way.
                    stats['detected']['cf_press_mandatory'] += 1
                    if more:
                        _act9 = ai.Action(die_idx=-1, action_type='siege_press',
                                          params={'no_reduce': True})
                    else:
                        _act9 = next((a for a in la
                                      if a.action_type == 'siege_no_press'),
                                     None)
                        if _act9 is None:
                            broke = 'cf_press_unmatched'
                            break
                    s2 = ai.apply_action(s2, _act9)
                    continue
                hidx = _match_index(la, lambda a: a.action_type == want)
                if hidx is None:
                    broke = 'cf_press_unmatched'
                    break
                if len(la) >= 2:
                    emitted.append(('siege_press', r_i, _emit_sample(
                        s2, dside, la, hidx, 'siege_press', batch['turn'])))
                s2 = ai.apply_action(s2, la[hidx])
                continue

            if ph == 'siege_trim':
                tr = rr.sanitize(str(pc.get('trim_region') or dfn))
                eb, _rb, to_cas = _window_trim(batch, tr, dfn_str)
                if to_cas:
                    stats['detected']['siege_trim_to_casualties'] += 1
                hidx = _match_index(la, lambda a: a.action_type ==
                                    'siege_trim' and
                                    int(a.params.get('elites_back', -1))
                                    == eb)
                if hidx is None:
                    broke = 'cf_trim_unmatched'
                    break
                if len(la) >= 2:
                    emitted.append(('siege_trim', r_i, _emit_sample(
                        s2, dside, la, hidx, 'siege_trim', batch['turn'])))
                s2 = ai.apply_action(s2, la[hidx])
                continue

            if ph == 'victor_advance':
                # W2-025: the post-siege-retreat advance (defender
                # withdrew BEFORE round 1, rounds_fought == 0) precedes
                # every roll of the batch - a roll-anchored window
                # would miss the move-in drag and label a real advance
                # as victor_stay (a WRONG row); the whole batch is the
                # window, exactly as _emit_siege_withdrawal uses it.
                if r_i <= 0:
                    lo = 0
                else:
                    lo = wins[min(r_i, len(wins)) - 1]['first']
                comp = _moved_group_composition(batch, atk, dfn, lo=lo)
                moved = comp['reg'] + comp['elite'] + comp['leaders'] + \
                    comp['nazgul']
                if moved == 0:
                    hidx = _match_index(la, lambda a: a.action_type ==
                                        'victor_stay')
                else:
                    hidx = _match_army_move(la, s2, side, atk, dfn, comp,
                                            atype='victor_advance')
                if hidx is None:
                    broke = 'cf_advance_unmatched'
                    break
                if len(la) >= 2:
                    emitted.append(('victor_advance', r_i, _emit_sample(
                        s2, dside, la, hidx, 'victor_advance',
                        batch['turn'])))
                s2 = ai.apply_action(s2, la[hidx])
                continue

            if ph == 'attacker_siege_decision':  # self-play: assault only
                nx = next((a for a in la
                           if a.action_type == 'siege_assault'), None)
                if nx is None:
                    broke = 'cf_no_assault'
                    break
                s2 = ai.apply_action(s2, nx)
                continue

            if ph == 'fp_shine':
                stats['detected']['fp_shine'] += 1
                hidx, evidence, reason = _match_shine_gesture(
                    s2, batch, la, cards, _recorded_rounds)
                if hidx is None:
                    broke = 'cf_shine_choice_unproven:' + reason
                    break
                stats['detected']['fp_shine_gesture'] += 1
                if len(la) >= 2:
                    emitted.append(('fp_shine', r_i, _emit_shine_sample(
                        s2, dside, la, hidx, batch['turn'], evidence)))
                s2 = ai.apply_action(s2, la[hidx])
                continue

            if ph == 'combat_effect_choice':
                stats['detected']['cc_effect_choice'] += 1
                ec = pc.get('effect_choice') or {}
                if ec.get('effect') == 'words_of_power' and ec.get('stage') == 'pre':
                    from combat_choice_evidence import match_words_of_power
                    if r_i >= len(wins) or r_i >= len(cards) or r_i >= len(_recorded_rounds):
                        broke = 'cf_words_of_power_target_unproven:missing_round'
                        break
                    commits = [cards[r_i].get(k) for k in ('fp_ln', 'sh_ln')]
                    lower = max([int(batch.get('start_line') or 0),
                                 wins[r_i - 1]['last'] if r_i else 0] +
                                [int(line) for line in commits if line is not None])
                    hidx, evidence, reason = match_words_of_power(
                        s2, batch, la, lower, wins[r_i]['first'],
                        _recorded_rounds[r_i], chats=_COMBAT_CHAT_INDEX)
                    if hidx is None:
                        broke = 'cf_words_of_power_target_unproven:' + reason
                        break
                    # This corpus-only user priority is explicitly inferred;
                    # an authenticated cancellation supersedes the fallback.
                    # Apply the exact current legal action returned by the
                    # matcher, without changing self-play's learned choice.
                    stats['detected']['cc_words_of_power_' + evidence['provenance']] += 1
                    if len(la) >= 2:
                        emitted.append(('combat_effect', r_i, _emit_observed_combat_sample(
                            s2, dside, la, hidx, batch['turn'], evidence)))
                    _drive_fx9.add('words_of_power')
                    s2 = ai.apply_action(s2, la[hidx])
                    continue
                if (getattr(ai, '_COMBAT_ROUND_CONTEXT_ON', False)
                        and ec.get('effect') in ('onslaught', 'onslaught_payment')
                        and ec.get('stage') in ('post', 'onslaught_payment')):
                    from combat_choice_evidence import match_onslaught_payment
                    ri = max(0, r_i - 1)
                    if ri >= len(_recorded_rounds):
                        broke = 'cf_onslaught_missing_round'
                        break
                    w = wins[ri]
                    ordinary_boundary = (pc.get('_rx_onslaught_ordinary_end')
                        if pc.get('_rx_onslaught_ordinary_round') == ri else None)
                    hidx, line, reason = match_onslaught_payment(
                        s2, batch, la, w['first'], w['next_first'], _recorded_rounds[ri],
                        self_loss_after_line=ordinary_boundary, chats=_COMBAT_CHAT_INDEX)
                    unordered = None
                    if hidx is None and ec.get('effect') == 'onslaught_payment':
                        from combat_choice_evidence import match_unordered_onslaught_removal_group
                        unordered = match_unordered_onslaught_removal_group(
                            s2, batch, la, w['first'], w['next_first'], _recorded_rounds[ri],
                            self_loss_after_line=ordinary_boundary)
                        if unordered is not None:
                            first = unordered['canonical_options'][0]
                            hidx = next(i for i in unordered['legal_indices']
                                if {k: v for k, v in la[i].params.items() if k != 'card_id'} == first)
                            line, reason = unordered['line'], unordered['reason']
                    if hidx is None:
                        broke = 'cf_onslaught_' + reason
                        break
                    stats['detected']['cc_round_effect_' + ec['effect']] += 1
                    if len(la) >= 2:
                        evidence = {'schema': 'observed_round_v1',
                            'effect': ec['effect'], 'round': ri, 'line': line, 'reason': reason,
                            'observed_rolls': ['combat', 'reroll', 'ordinary_casualties']}
                        if unordered is not None:
                            evidence['physical_regular_losses'] = unordered['physical_regular_losses']
                            evidence['physical_unit_losses'] = unordered['physical_unit_losses']
                            evidence['equivalent_first_indices'] = unordered['legal_indices']
                        previous_alt = _RX_ALT_HIDX
                        try:
                            if unordered is not None:
                                _RX_ALT_HIDX = unordered['legal_indices']
                            sample = _emit_observed_combat_sample(s2, dside, la, hidx,
                                                                  batch['turn'], evidence)
                        finally:
                            _RX_ALT_HIDX = previous_alt
                        emitted.append(('combat_effect', ri, sample))
                    _drive_fx9.add(ec['effect'])
                    s2 = ai.apply_action(s2, la[hidx])
                    if unordered is not None:
                        # The record specifies a simultaneous loss set. Its
                        # commuting implementation order supplies no further
                        # human decisions and must not produce extra rows.
                        for option in unordered['canonical_options'][1:]:
                            next_pc = getattr(s2, 'pending_combat', None) or {}
                            if (next_pc.get('effect_choice') or {}).get('effect') != 'onslaught_payment':
                                raise _CasualtyGestureError('unordered_payment_ended_early')
                            menu = ai.get_legal_actions(s2, next_pc['deciding_side'])
                            matches = [a for a in menu if a.action_type == 'combat_effect_choice'
                                and {k: v for k, v in a.params.items() if k != 'card_id'} == option]
                            if len(matches) != 1:
                                raise _CasualtyGestureError('unordered_payment_not_legal')
                            s2 = ai.apply_action(s2, matches[0])
                        stats['detected']['cc_unordered_commuting_payment_group'] += 1
                    continue
                if ec.get('stage') == 'round':
                    if not getattr(ai, '_COMBAT_ROUND_CONTEXT_ON', False):
                        broke = 'cf_round_effect_requires_v52'
                        break
                    from combat_choice_evidence import match_round_choice
                    if not (0 <= r_i < len(_recorded_rounds)):
                        broke = 'cf_round_effect_missing_dice'
                        break
                    w = wins[r_i]
                    lo = _observed_round_choice_lower(batch, wins, cards, r_i)
                    hidx, evidence_line, evidence_reason = match_round_choice(
                        s2, batch, la, _recorded_rounds[r_i], lo,
                        w['next_first'], _round_choice_lines.get(r_i, 0), chats=_COMBAT_CHAT_INDEX)
                    if hidx is None:
                        broke = 'cf_round_effect_' + evidence_reason
                        break
                    effect = str(ec.get('effect', ''))
                    stats['detected']['cc_round_effect_' + effect] += 1
                    if len(la) >= 2:
                        evidence = {
                            'schema': 'observed_round_v1', 'effect': effect,
                            'round': r_i, 'line': evidence_line, 'reason': evidence_reason,
                            'observed_rolls': sorted(pc['_round_resume']['cursor'].observed_rolls)}
                        sample = _emit_observed_combat_sample(s2, dside, la, hidx,
                                                              batch['turn'], evidence)
                        emitted.append(('combat_effect', r_i, sample))
                    _drive_fx9.add(effect)
                    # RA's count and subsequent unit payment describe the
                    # same physical sacrifice; do not consume its gesture
                    # before the separate payment matcher can inspect it.
                    is_payment = effect in ('relentless_assault_payment', 'pre_combat_casualty')
                    if not effect.startswith('relentless_assault') and not is_payment:
                        _round_choice_lines[r_i] = evidence_line
                    s2 = ai.apply_action(s2, la[hidx])
                    next_pc = getattr(s2, 'pending_combat', None) or {}
                    if is_payment and (next_pc.get('effect_choice') or {}).get('effect') != effect:
                        _round_choice_lines[r_i] = evidence_line
                    continue
                if len(la) == 1:
                    s2 = ai.apply_action(s2, la[0])
                    continue
                # spec 1.216 item 3/5: back-calculate the magnitude K.  Dread
                # and Despair ('ff N' chat) + Heroic Death (owner figure death)
                # are derivable; ONSLAUGHT self-hits back-calc from how many
                # units died (owner window hits MINUS the engine roll casualties
                # captured at the owner's casualty_allocation this round).
                # Relentless Assault (stage 'pre', pre-roll) has no captured
                # roll-hit denominator at the decision point -> censused.
                ec = pc.get('effect_choice') or {}
                owner = ec.get('owner')
                owner_str = atk_str if owner == 'atk' else dfn_str
                w = wins[min(max(r_i, 0), len(wins) - 1)]
                lo = (wins[r_i - 1]['last'] if 0 < r_i < len(wins) else 0)
                if ec.get('targets') or ec.get('pairs'):
                    # 1.254 TARGET-BEARING menus (DET census P1-a):
                    # WoP/BB/BoW carry ec['targets'] (which figure),
                    # Heroic Death ec['pairs'] ((count, figure)).  The
                    # victim is NAMED in the window's kill lines - match
                    # on it; count-only multi-matched and dropped.
                    hidx = _effect_target_pick(batch, ec, la, owner_str,
                                               lo, w['next_first'],
                                               stats)
                    if hidx is not None:
                        _drive_fx9.add(str(ec.get('effect', '')).lower())
                        stats['detected']['cc_effect_target'] += 1
                        emitted.append(('combat_effect', r_i,
                                        _emit_sample(
                                            s2, dside, la, hidx,
                                            'combat_effect_choice',
                                            batch['turn'])))
                        broke = 'cf_effect_emitted'
                    else:
                        broke = 'cf_effect_choice'
                    break
                _counts_legal = [int(a.params.get('count', -1)) for a in la
                                 if a.action_type == 'combat_effect_choice']
                k = None
                _ekind = str(ec.get('effect', '')).lower()
                _ons_dup = False
                _ons_done = False
                if _CLOSURE and _ekind == 'onslaught':
                    # W2-008: the sacrifice is rendered as a PARK of the
                    # units onto the committed card (GAME01 T3 MT: [4]Sauron
                    # Regular -> Current SA Card, then 4 dice); parked units
                    # never reach a casualty zone, so the kills-minus-roll
                    # back-calc below read K=0 for them.  The park is the
                    # decision when it exists (Regulars = 1 hit each; an
                    # Elite on the card is 1 or 2 hits = ambiguous = drop).
                    # rounds_fought is already incremented: the round just
                    # fought is wins[r_i - 1].
                    _wo = wins[min(max(r_i - 1, 0), len(wins) - 1)]
                    _kp, _kp_why = _onslaught_park_k(
                        batch, owner_str, regs, _wo['first'],
                        _wo['next_first'])
                    if _kp is not None and _kp in _counts_legal:
                        stats['detected']['cc_onslaught_park'] += 1
                        k = _kp
                        _ons_done = True
                    elif _kp_why != 'no_park':
                        stats['drop_reasons']['cf_onslaught_%s'
                                              % _kp_why] += 1
                        _ons_done = True
                if _CLOSURE and _ekind == 'onslaught' and not _ons_done:
                    # rounds_fought is incremented before the onslaught choice,
                    # so the owner's roll-hits may be keyed at r_i, r_i-1, or the
                    # latest captured this combat (onslaught resolves in the
                    # round just fought).
                    _rh = round_roll_hits.get((r_i, owner))
                    if _rh is None:
                        _rh = round_roll_hits.get((r_i - 1, owner))
                    if _rh is None:
                        _cand = [v for (ri2, ow2), v in round_roll_hits.items()
                                 if ow2 == owner]
                        _rh = _cand[-1] if _cand else None
                    if _rh is not None:
                        _tot = _window_owner_hits(batch, owner_str, regs,
                                                  w['first'], w['next_first'])
                        _kk = _tot - _rh
                        if _kk in _counts_legal:
                            stats['detected']['cc_onslaught_backcalc'] += 1
                            k = _kk
                        else:
                            stats['drop_reasons']['cf_onslaught_unclean'] += 1
                    else:
                        stats['drop_reasons']['cf_onslaught_no_hits'] += 1
                if _ekind == 'onslaught' and k is not None:
                    # W2-008: the resolver stash (_onslaught_decision) may
                    # already have emitted this batch's magnitude - never
                    # a second row for one decision; a disagreeing K is a
                    # conflict (drop beats wrong).
                    _ons_prev = stats.get('_ons_k', {}).get(
                        batch.get('start_line') or 0)
                    if _ons_prev is not None:
                        if int(_ons_prev) == int(k):
                            _ons_dup = True
                        else:
                            stats['drop_reasons']['cf_onslaught_conflict'] += 1
                            k = None
                if k is None and _ekind != 'onslaught':
                    # (relentless_assault now back-calcs from its park
                    # gesture inside _combat_effect_count - census P1-a)
                    k = _combat_effect_count(batch, ec, owner_str,
                                             _counts_legal, lo,
                                             w['next_first'])
                hidx = None if k is None else _match_index(
                    la, lambda a: a.action_type == 'combat_effect_choice' and
                    int(a.params.get('count', -2)) == k)
                if hidx is not None:
                    # emit the magnitude sample, then BREAK (keeping all
                    # pre-break samples).  We do NOT continue the drive past a
                    # back-calculated effect: a wrong K would fail the final
                    # gate and drop the earlier faithfully-forced samples the
                    # break preserves (net-negative), so emit-and-stop is the
                    # strictly-additive choice.
                    _drive_fx9.add(_ekind)
                    if _ons_dup:
                        # W2-008: same K the resolver stash already
                        # emitted for this line - counted, not doubled
                        stats['detected']['cc_onslaught_dup_skip'] += 1
                    else:
                        stats['detected']['cc_effect_backcalc'] += 1
                        emitted.append(('combat_effect', r_i, _emit_sample(
                            s2, dside, la, hidx, 'combat_effect_choice',
                            batch['turn'])))
                    broke = 'cf_effect_emitted'
                else:
                    broke = 'cf_effect_choice'
                break

            broke = 'cf_phase_%s' % str(ph)[:18]
            break
        if not broke and not _strike9f:
            _recorded_rng.finish()
    except _RecordedCombatDiceError as e:
        stats['drop_reasons']['cf_rng_mismatch:' + e.reason] += 1
        broke = 'cf_rng_mismatch:counted'
    except Exception as e:
        stats['drop_reasons']['cc_drive_exc:%s' % str(e)[:12]] += 1
        broke = broke or 'cc_drive_exc:counted'
    finally:
        random.randint, random.choice = ori, orc
        _combat_rng.roll_d6_batch = _original_combat_d6
        _combat_rng._apply_pre_combat_for_side = _original_pre_attack
        _onslaught_rng._advance = _original_onslaught_advance
        _combat_rng.FINAL_HITS_HOOK = _original_final_hits_hook

    if broke and (broke.startswith('cf_shine_choice_unproven:')
                  or broke.startswith('cf_words_of_power_target_unproven')):
        # No later phase has an authenticated state after this unresolved
        # choice. Do not feed the gesture-recovery paths a guessed effect.
        stats['drop_reasons'][broke] += 1
        return [sample for _, _, sample in emitted]

    def _recovery_context(state9, **fields):
        """Install a reconstructed phase with the original battle formation."""
        if not _strike9f and _declared_rear is None:
            raise ValueError('missing declared battle formation')
        state9.pending_combat = {
            'attacker_region': atk, 'defender_region': dfn,
            'attacker_side': side,
            'defender_side': Side.FREE_PEOPLES if side == Side.SHADOW else Side.SHADOW,
            'is_siege': _sieg, **fields}
        if _declared_rear is not None:
            state9.pending_combat['attacker_rearguard'] = _declared_rear
            if _declared_rear[2][1] or _declared_rear[2][2]:
                # A missing source Army cannot turn a nonempty rear into
                # an empty one (partition(None) intentionally returns []).
                from army_participants import partition_battle_army
                import combat as combat9
                origin9 = _declared_rear[0]
                key9, rear_army9 = combat9._army_entry(state9, origin9)
                if rear_army9 is None:
                    raise ValueError('missing protected rearguard origin')
                partition_battle_army(state9, rear_army9, key9, side)
        # A gesture that removed a protected figure cannot establish a
        # recoverable state. Partition validates the frozen protected counts.
        _recovery_units(state9, side)

    def _recovery_units(state9, owner9):
        from army_participants import partition_battle_army
        import combat as combat9
        pc9 = state9.pending_combat
        region9 = pc9['attacker_region' if owner9 == side else 'defender_region']
        key9, army9 = combat9._army_entry(state9, region9)
        return partition_battle_army(state9, army9, key9 or region9, owner9)[0]

    def _gd_rounds(_start):
        # USER truth-table fix #9 FINAL (game01_verification.md): the
        # unverified rounds' sub-decisions are still GESTURES - emit
        # them independently of the drive: per round window, each
        # side's allocation (k = downgrade-returns, hits = gesture
        # count) and the attacker's press (elite-payment gesture
        # between windows), with menus from the engine's own pending
        # on a mirror-derived mid-battle state (recon + casualties
        # applied up to the round's first roll line).
        try:
            if not _strike9f and _declared_rear is None:
                stats['drop_reasons']['gd_missing_formation'] += 1
                return
            import game_state as gs9
            from collections import Counter as _Ctr9
            _atk_e = (gs9.Side.SHADOW if batch['die_side'] == 'SHADOW'
                      else gs9.Side.FREE_PEOPLES)
            _dfn_e = (gs9.Side.FREE_PEOPLES
                      if _atk_e == gs9.Side.SHADOW else gs9.Side.SHADOW)
            # what the drive already kept (kind, round) - the gesture
            # re-emissions below must never double a surviving sample
            _have = _Ctr9((k9, ri9) for (k9, ri9, _s9) in keep)
            _have_k = _Ctr9(k9 for (k9, ri9, _s9) in keep)
            try:
                import rendered_census as rc9z
                rc9d0 = rc9z._combat_atk_player(batch)
            except Exception:
                rc9d0 = None
            _atk_slot = 'sh' if batch['die_side'] == 'SHADOW' else 'fp'
            _dfn_slot = 'fp' if _atk_slot == 'sh' else 'sh'
            # the loop runs over ALL windows: the CC block self-guards
            # via _have (the drive's 1-based alloc indices made _start
            # skip round-2 commits - GAME19 T8 WWSAT/Athelas); allocs/
            # press/field stay gated on _start below
            for r in range(0, len(wins)):
                w = wins[r]
                try:
                    st_r = _recon_state(batch['truth_before'], batch.get('turn'))
                except Exception:
                    break
                _install_ledger_hands(st_r, batch)
                _apply_window_casualties(st_r, batch, regs,
                                         w['first'])
                def _side_key(region, s_en):
                    # recon keeps a besieged garrison under the
                    # 'X stronghold' BOX key while the engine pending
                    # expects the key its units actually sit under -
                    # pick whichever army key holds that side's units
                    import rendered_replay as rr9
                    import ai as ai9
                    best = region
                    for k2, a2 in st_r.armies.items():
                        ks = rr9.sanitize(k2)
                        if not (ks == region or
                                ks == region + ' stronghold' or
                                ks.startswith(region)):
                            continue
                        if any(ai9._side_of_nation(u.nation) == s_en
                               for u in a2.units):
                            best = k2
                            break
                    return best
                _recovery_context(st_r,
                    attacker_region=_side_key(atk, _atk_e),
                    defender_region=_side_key(dfn, _dfn_e))
                # No later round exists after either participating Army
                # has been eliminated. A surviving rear is not a fighter.
                if not _strike9f and (not _recovery_units(st_r, _atk_e)
                                      or not _recovery_units(st_r, _dfn_e)):
                    stats['detected']['gd_round_after_battle_end'] += 1
                    break
                # per-round COMBAT-CARD commit/decline (truth rows
                # T5-15/T5-16 class): the commit is a physical pre-roll
                # card move (line-segmented by _round_cards, gesture
                # truth); the decline is the proven round rolling with
                # no commit.  Round-0 commits keep the drive's standing
                # guarantee - gd owns rounds the keep-filter dropped.
                # a REAL fought round has chunks from BOTH players in
                # its window; a lone attacker chunk is a card's bonus
                # roll (T3 Onslaught magnitude) - never a cc window
                _real_rd = False
                try:
                    import rendered_census as rc9c
                    _apc = rc9c._combat_atk_player(batch)
                    _real_rd = any(
                        c.get('player') != _apc
                        for c in batch['rng']
                        if c.get('kind') == 'roll_num'
                        and w['first'] <= c.get('line', 0) <= w['last'])
                except Exception:
                    pass
                if _real_rd and _have[('combat_card', r)] == 0 \
                        and r < len(cards):
                    for ph_nm, s_en9, slot9 in (
                            ('attacker_combat_card', _atk_e, _atk_slot),
                            ('defender_combat_card', _dfn_e, _dfn_slot)):
                        cid9 = cards[r].get(slot9)
                        if cid9 is None and slot9 in _anom_slots:
                            # this side has an UNASSIGNED play in the
                            # batch - a decline here could be a false
                            # label (the play may belong to this round)
                            stats['drop_reasons']['gd_cc_anom'] += 1
                            continue
                        # hand AS OF the commit itself: mid-battle
                        # draws (WK ability) put the played card in
                        # hand only moments before its exit (T9 Grond)
                        _install_ledger_hands(
                            st_r, batch,
                            at_line=cards[r].get(slot9 + '_ln')
                            or w['first'])
                        _recovery_context(st_r, **{
                            'phase': ph_nm, 'round': r + 1,
                            'attacker_region': _side_key(atk, _atk_e),
                            'defender_region': _side_key(dfn, _dfn_e),
                            'attacker_side': _atk_e,
                            'defender_side': _dfn_e,
                            'deciding_side': s_en9, 'is_siege': _sieg})
                        if cid9 is not None and not _combat_commit_board_matches(
                                st_r, batch, cards[r].get(slot9 + '_ln')):
                            stats['drop_reasons']['gd_cc_commit_board_mismatch'] += 1
                            continue
                        try:
                            lc9 = [a for a in ai.get_legal_actions(
                                st_r, s_en9) if a.action_type
                                .startswith('combat_card')]
                        except Exception:
                            lc9 = []
                        if len(lc9) >= 2:
                            _pend_note(stats, 'combat_card', 'open')
                        if cid9 is not None:
                            hc9 = _match_index(
                                lc9, lambda a: a.action_type ==
                                'combat_card_play' and
                                a.params.get('card_id') == cid9)
                            if hc9 is None:
                                # DET census P2-a: named card, ledger
                                # gap - inject (log = truth)
                                lc9 = [a for a in _inject_named_cc(
                                    st_r, s_en9, cid9, lc9, stats)
                                    if a.action_type
                                    .startswith('combat_card')]
                                hc9 = _match_index(
                                    lc9, lambda a: a.action_type ==
                                    'combat_card_play' and
                                    a.params.get('card_id') == cid9)
                        else:
                            hc9 = _match_index(
                                lc9, lambda a: a.action_type ==
                                'combat_card_none')
                        if hc9 is not None and len(lc9) >= 2:
                            sc9 = _emit_sample(st_r, s_en9, lc9, hc9,
                                               'combat_card',
                                               batch['turn'])
                            if sc9 is not None:
                                keep.append(('combat_card', r, sc9))
                        elif cid9 is not None:
                            stats['drop_reasons']['gd_cc_unmatch'] += 1
                    st_r.pending_combat = None
                    _install_ledger_hands(st_r, batch)   # restore
                if r < _start:
                    continue
                round_start_state = st_r
                allocation_lines = _recovery_allocation_lines(
                    batch, regs, w['first'], w['next_first'], atk_str)
                if allocation_lines is None:
                    stats['drop_reasons']['gd_alloc_order_ambiguous'] += 1
                for stage, s_str, s_enum in (
                        ('attacker', atk_str, _atk_e),
                        ('defender', dfn_str, _dfn_e)):
                    if allocation_lines is None or s_str not in allocation_lines:
                        continue
                    # Reconstruct the actual decision prefix independently.
                    # A forced attacker allocation changes the defender's
                    # observation too, even though no attacker row is emitted.
                    st_r = _recon_state(batch['truth_before'], batch.get('turn'))
                    _apply_window_casualties(st_r, batch, regs,
                                             allocation_lines[s_str])
                    _install_ledger_hands(st_r, batch,
                                          at_line=allocation_lines[s_str])
                    gw = _window_owner_hits(batch, s_str, regs,
                                            w['first'],
                                            w['next_first'])
                    from cards import get_card as _cas_get_card
                    _separate = any(
                        getattr(_cas_get_card(cards[r].get(slot)), 'combat_name', '').lower()
                        in ('onslaught', 'relentless assault', 'great host', 'we come to kill')
                        for slot in ('fp', 'sh') if cards[r].get(slot))
                    if (_separate or _onslaught_owner_commit(batch, s_str, w['first'], w['next_first'])
                            or (_sieg and stage == 'attacker' and r + 1 < len(wins)
                                and not (_mand9 and r + 1 < _mand9))):
                        stats['drop_reasons']['gd_alloc_separate_loss_window'] += 1
                        continue
                    try:
                        events = _casualty_gesture_operations(
                            batch, s_str, regs, w['first'], w['next_first'])
                        if not events:
                            continue
                        role = 'atk' if stage == 'attacker' else 'def'
                        recorded_hits = round_roll_hits.get((r + 1, role))
                        gw = (int(recorded_hits) if recorded_hits is not None else
                              sum(2 if event[2] == 'loss_elite' else 1 for event in events))
                        _recovery_context(st_r, **{
                            'phase': 'casualty_allocation', 'alloc_stage': stage,
                            'attacker_region': _side_key(atk, _atk_e),
                            'defender_region': _side_key(dfn, _dfn_e),
                            'attacker_side': _atk_e, 'defender_side': _dfn_e,
                            'atk_hits': gw if stage == 'attacker' else 0,
                            'def_hits': gw if stage == 'defender' else 0,
                            'deciding_side': s_enum, 'is_siege': _sieg})
                        _, allocations = _drive_casualty_gestures(
                            st_r, s_enum, events, known_hits=recorded_hits is not None)
                    except _CasualtyGestureError as error:
                        reason = ('gd_alloc_unmatch' if str(error) == 'operation_unmatched'
                                  else 'gd_alloc_operation:' + str(error))
                        stats['drop_reasons'][reason] += 1
                        continue
                    for before, menu, chosen in allocations:
                        _pend_note(stats, 'casualty_alloc', 'open')
                        sample = _emit_sample(before, s_enum, menu, chosen,
                                              'casualty_alloc', batch['turn'])
                        if sample is not None:
                            keep.append(('casualty_alloc', r, sample))
                    if not allocations:
                        stats['detected']['gd_alloc_forced'] += 1
                # Later phase recovery has its own reconstruction rules.
                # Do not leak the last allocation prefix into those paths.
                st_r = round_start_state
                # press payment between this round and the next window
                nlo = w['last']
                nhi = wins[r + 1]['first'] if r + 1 < len(wins) \
                    else batch['end_line'] + 1
                # W2-008 (slice of W2-015): an Elite swap after the LAST
                # window is that round's casualty allocation (elite
                # downgrade), never a press payment - a paid press is
                # followed by the round it bought (GAME01 T3 MT L349:
                # Sudden Strike hit -> Sauron Elite downgraded, no press,
                # die killed).  The 'press' chat evidence below stands.
                paid = (r + 1 < len(wins)
                        and _window_replacements(batch, atk_str, regs,
                                                 nlo, nhi) > 0
                        and _window_owner_hits(batch, atk_str, regs, nlo,
                                               nhi) >= 1)
                if not paid and _CHAT_INDEX:
                    # explicit 'press' chat between this round's last
                    # chunk and the next round's last chunk - the
                    # payment-pair span can be skewed by a card bonus
                    # chunk widening a window (T9 WCtK: press #3)
                    _nhi2 = (wins[r + 1]['last'] if r + 1 < len(wins)
                             else batch['end_line'] + 1)
                    paid = any(
                        w['last'] < ln <= _nhi2
                        and re.match(r'^\s*press\b', t)
                        for (ln, t) in _CHAT_INDEX)
                is_last = (r + 1 >= len(wins))
                # W2-037: a card strike (Dreadful Spells class) fights no
                # round, so there is no siege press/no-press decision
                _press_do = (paid or is_last) and _sieg and not _strike9f
                if _press_do and _mand9 and (r + 1) < _mand9:
                    # 1.267: rounds 1..2 of a Grond / Fighting Uruk-hai
                    # siege are mandatory - no press decision, no Elite
                    # cost (a between-rounds Elite swap here is a
                    # casualty, never a payment)
                    stats['detected']['gd_press_mandatory'] += 1
                    _press_do = False
                if _press_do and is_last and not paid:
                    # the synthetic last-round no-press is a decision
                    # ONLY in a siege whose defender still stands - a
                    # battle that ended by elimination left nothing to
                    # press into (T4 Lossarnach phantom).  NOTE: no
                    # `continue` here - the field continue/retreat
                    # blocks below must still run for this round.
                    try:
                        st_end = _recon_state(batch['truth_before'], batch.get('turn'))
                        _apply_window_casualties(st_end, batch, regs,
                                                 10 ** 9)
                        import ai as ai9
                        import rendered_replay as rr9
                        _alive = 0
                        for _k9, _a9 in st_end.armies.items():
                            _ks9 = rr9.sanitize(_k9)
                            if not any(_ks9 == _rg9 or
                                       _ks9.startswith(_rg9)
                                       for _rg9 in regs):
                                continue
                            _alive += sum(
                                1 for u in _a9.units
                                if ai9._side_of_nation(u.nation)
                                == _dfn_e)
                        if _alive == 0:
                            stats['detected']['gd_press_moot'] += 1
                            _press_do = False
                    except Exception:
                        pass
                if _press_do:
                    # The menu below belongs to the completed round. The
                    # remaining physical rear must not create a press
                    # choice after all participating attackers were lost.
                    st_alive = _recon_state(batch['truth_before'], batch.get('turn'))
                    _apply_window_casualties(st_alive, batch, regs, nhi)
                    _recovery_context(st_alive,
                        attacker_region=_side_key(atk, _atk_e),
                        defender_region=_side_key(dfn, _dfn_e))
                    if not _recovery_units(st_alive, _atk_e) or \
                            not _recovery_units(st_alive, _dfn_e):
                        stats['detected']['gd_press_battle_ended'] += 1
                        _press_do = False
                if _press_do:
                    _recovery_context(st_r, **{
                        'phase': 'combat_round_result',
                        'attacker_region': atk,
                        'defender_region': dfn,
                        'attacker_side': _atk_e,
                        'defender_side': _dfn_e,
                        'deciding_side': _atk_e, 'is_siege': True})
                    try:
                        lp = [a for a in ai.get_legal_actions(
                            st_r, _atk_e) if a.action_type in
                            ('siege_press', 'siege_no_press')]
                    except Exception:
                        lp = []
                    if len(lp) >= 2:
                        _pend_note(stats, 'siege_press', 'open')
                    wantp = 'siege_press' if paid else 'siege_no_press'
                    hp = _match_index(lp, lambda a: a.action_type
                                      == wantp)
                    if hp is not None and len(lp) >= 2:
                        sp = _emit_sample(st_r, _atk_e, lp, hp,
                                          'siege_press', batch['turn'])
                        if sp is not None:
                            keep.append(('siege_press', r, sp))
                    elif paid:
                        stats['drop_reasons']['gd_press_unmatch'] += 1
                # FIELD battle: attacker continue/cease + defender
                # retreat (truth rows T5-29/31 class) - both anchored:
                # a following round window proves continue, the retreat
                # gesture proves continue AND names the destination,
                # a last round with a living defender and no retreat is
                # the structural cease.  Menus on the post-window state
                # (casualties through the window applied).
                if not _sieg and any(
                        c.get('player') != rc9d0
                        for c in batch['rng']
                        if c.get('kind') == 'roll_num'):
                    rk9, rd9 = _window_retreat(batch, dfn, dfn_str,
                                               w['first'],
                                               w['next_first'])
                    if rk9 == 'ambiguous':
                        stats['drop_reasons']['gd_retreat_chain_ambiguous'] += 1
                        continue
                    _more9 = (r + 1) < len(wins)
                    # a window with NO attacker chunk is a defender
                    # pre-roll artifact (Sudden Strike rolls first) -
                    # no round ended here, so no continue decision
                    _fld_ok = True
                    try:
                        import rendered_census as rc9d
                        _apd = rc9d._combat_atk_player(batch)
                        if not any(
                                c.get('player') == _apd
                                for c in batch['rng']
                                if c.get('kind') == 'roll_num'
                                and w['first'] <= c.get('line', 0)
                                <= w['last']):
                            _fld_ok = False
                    except Exception:
                        pass
                    st_c = None
                    if _fld_ok:
                        try:
                            st_c = _recon_state(batch['truth_before'], batch.get('turn'))
                            _install_ledger_hands(st_c, batch)
                            _apply_window_casualties(
                                st_c, batch, regs, w['next_first'])
                            _recovery_context(st_c,
                                attacker_region=_side_key(atk, _atk_e),
                                defender_region=_side_key(dfn, _dfn_e))
                            if not _recovery_units(st_c, _atk_e) or \
                                    not _recovery_units(st_c, _dfn_e):
                                # Rear survivors cannot continue or retreat
                                # on behalf of an eliminated fighting Army.
                                stats['detected']['gd_field_battle_ended'] += 1
                                st_c = None
                        except Exception:
                            st_c = None
                    if st_c is not None and \
                            _have[('attacker_continue', r + 1)] == 0 \
                            and _have[('attacker_continue', r)] == 0:
                        _wantc = None
                        if _more9 or rk9:
                            _wantc = 'combat_continue'
                        else:
                            _alv9 = 0
                            for _k8, _a8 in st_c.armies.items():
                                _ks8 = rr.sanitize(_k8)
                                if any(_ks8 == _rg8 or
                                       _ks8.startswith(_rg8)
                                       for _rg8 in regs):
                                    import ai as ai8
                                    _alv9 += sum(
                                        1 for u in _a8.units
                                        if ai8._side_of_nation(u.nation)
                                        == _dfn_e)
                            if _alv9 > 0:
                                _wantc = 'combat_cease'
                        if _wantc is not None:
                            _recovery_context(st_c, **{
                                'phase': 'attacker_continue',
                                'attacker_region': atk,
                                'defender_region': dfn,
                                'attacker_side': _atk_e,
                                'defender_side': _dfn_e,
                                'deciding_side': _atk_e,
                                'is_siege': False})
                            try:
                                lac = ai.get_legal_actions(st_c, _atk_e)
                            except Exception:
                                lac = []
                            if len(lac) >= 2:
                                _pend_note(stats, 'attacker_continue',
                                           'open')
                            hac = _match_index(
                                lac, lambda a: a.action_type == _wantc)
                            if hac is not None and len(lac) >= 2:
                                sac = _emit_sample(
                                    st_c, _atk_e, lac, hac,
                                    'attacker_continue', batch['turn'])
                                if sac is not None:
                                    keep.append(
                                        ('attacker_continue', r + 1,
                                         sac))
                            else:
                                stats['drop_reasons'][
                                    'gd_continue_unmatch'] += 1
                    if st_c is not None and not rk9 and _more9 and \
                            _have[('defender_retreat', r + 1)] == 0 \
                            and _have[('defender_retreat', r)] == 0:
                        # the battle CONTINUED with no retreat = the
                        # defender STOOD - a decline-class decision
                        # (menu: retreat dests + no_retreat; GAME19 T8
                        # 'stay'; GAME01 T5 FoI T5-29b amended to match)
                        _recovery_context(st_c, **{
                            'phase': 'defender_retreat',
                            'attacker_region': atk,
                            'defender_region': _side_key(dfn, _dfn_e),
                            'attacker_side': _atk_e,
                            'defender_side': _dfn_e,
                            'deciding_side': _dfn_e,
                            'is_siege': False})
                        try:
                            las = ai.get_legal_actions(st_c, _dfn_e)
                        except Exception:
                            las = []
                        if len(las) >= 2:
                            _pend_note(stats, 'defender_retreat', 'open')
                        hs9 = _match_index(
                            las, lambda a: a.action_type ==
                            'combat_no_retreat')
                        if hs9 is not None and len(las) >= 2:
                            ss9 = _emit_sample(
                                st_c, _dfn_e, las, hs9,
                                'defender_retreat', batch['turn'])
                            if ss9 is not None:
                                keep.append(
                                    ('defender_retreat', r + 1, ss9))
                    if st_c is not None and rk9 and \
                            _have[('defender_retreat', r + 1)] == 0 \
                            and _have[('defender_retreat', r)] == 0:
                        _recovery_context(st_c, **{
                            'phase': 'defender_retreat',
                            'attacker_region': atk,
                            'defender_region': _side_key(dfn, _dfn_e),
                            'attacker_side': _atk_e,
                            'defender_side': _dfn_e,
                            'deciding_side': _dfn_e,
                            'is_siege': False})
                        try:
                            lar = ai.get_legal_actions(st_c, _dfn_e)
                        except Exception:
                            lar = []
                        if len(lar) >= 2:
                            _pend_note(stats, 'defender_retreat', 'open')
                        if rk9 == 'siege':
                            har = _match_index(
                                lar, lambda a: a.action_type ==
                                'retreat_into_siege')
                        else:
                            har = _match_index(
                                lar, lambda a: a.action_type ==
                                'combat_retreat' and
                                rr.sanitize(str(a.params.get(
                                    'to_region', ''))) == rd9)
                            # A recorded endpoint must be a legal retreat in
                            # the reconstructed decision state. Do not append
                            # an unsupported action to make a drag match.
                        if har is not None and len(lar) >= 2:
                            sar = _emit_sample(
                                st_c, _dfn_e, lar, har,
                                'defender_retreat', batch['turn'])
                            if sar is not None:
                                keep.append(
                                    ('defender_retreat', r + 1, sar))
                        else:
                            stats['drop_reasons'][
                                'gd_retreat_unmatch'] += 1
            # victor advance ONCE after the last window (field only,
            # positive advance gesture; sieges capture/auto-advance)
            if not _sieg and _have_k['victor_advance'] == 0 \
                    and _start < len(wins):
                comp9v = _moved_group_composition(
                    batch, atk, dfn, lo=wins[-1]['first'])
                mv9v = comp9v['reg'] + comp9v['elite'] + \
                    comp9v['leaders'] + comp9v['nazgul']
                if mv9v > 0:
                    st_v = _recon_state(batch['truth_before'], batch.get('turn'))
                    _install_ledger_hands(st_v, batch)
                    _apply_window_casualties(st_v, batch, regs, 10 ** 9)
                    try:
                        import combat as _cmbv
                        _kv, _av = _cmbv._army_entry(st_v, dfn)
                        if _av is not None:
                            import ai as aiv
                            _av.units = [
                                u for u in _av.units
                                if aiv._side_of_nation(u.nation)
                                != _dfn_e]
                    except Exception:
                        pass
                    _recovery_context(st_v, **{
                        'phase': 'victor_advance',
                        'attacker_region': atk, 'defender_region': dfn,
                        'attacker_side': _atk_e,
                        'defender_side': _dfn_e,
                        'deciding_side': _atk_e, 'is_siege': False})
                    if not _recovery_units(st_v, _atk_e):
                        stats['drop_reasons']['gd_advance_no_participants'] += 1
                        return
                    try:
                        lav = ai.get_legal_actions(st_v, _atk_e)
                    except Exception:
                        lav = []
                    if len(lav) >= 2:
                        _pend_note(stats, 'victor_advance', 'open')
                    hav = _match_army_move(lav, st_v, side, atk, dfn,
                                           comp9v,
                                           atype='victor_advance')
                    if hav is not None and len(lav) >= 2:
                        sav = _emit_sample(st_v, _atk_e, lav, hav,
                                           'victor_advance',
                                           batch['turn'])
                        if sav is not None:
                            keep.append(
                                ('victor_advance', len(wins), sav))
                    else:
                        stats['drop_reasons']['gd_advance_unmatch'] += 1
        except Exception as _egd:
            stats['drop_reasons']['gd_exc:%s' % str(_egd)[:14]] += 1

    keep = emitted
    if os.environ.get('RX_CF_DEBUG'):
        print('[CF] T%s %s->%s siege=%s rounds=%s broke=%r emitted=%s'
              % (batch['turn'], atk, dfn, _sieg, len(wins), broke,
                 [(e[0], e[1]) for e in emitted]))
    if broke:
        if not broke.startswith(('cc_drive_exc', 'cf_rng_mismatch:counted')):
            stats['drop_reasons'][broke] += 1
        # pre-break samples sat on faithfully-forced states -> keep
        if not no_dice and broke not in ('cf_retreat_correction_unproven',
                                          'cf_retreat_chain_ambiguous'):
            # BROKEN drives (carded multi-rounders, e.g. GAME01 T9
            # Corsairs) never reach the final gate - their post-break
            # rounds' gestures still emit via the gesture-driven path
            # allocs/presses prove their round completed; a round-0
            # combat EFFECT does not (GAME02 T4 Lorien: the drive broke
            # right after Heroic Death and _sgd=1 skipped round 0's
            # allocs + press entirely)
            # drive alloc/press rounds are 1-BASED (rounds_fought is
            # incremented before the allocation pending): the last
            # fully-emitted 1-based round == the 0-based index of the
            # NEXT round, so gesture emission resumes exactly there
            _sgd = max([e[1] for e in emitted
                        if e[0] in ('casualty_alloc',
                                    'siege_press')] or [0])
            _gd_rounds(_sgd)
    elif not no_dice and not _combat_final_ok(s2, batch['truth'],
                                              regs | retreat_dests):
        # USER truth-table fix #9 (game01_verification.md): the settled
        # END-total mismatch must not vaporize every sub-decision - the
        # per-round decisions ARE gestures.  Keep samples through the
        # last CONSECUTIVELY-VERIFIED round: a round verifies when, for
        # both sides, the engine hits captured at its casualty
        # allocation equal the gesture-derived hits of its window
        # (kills + downgrades read from the rendered lines).  Rounds at
        # or past the first unverified one still drop (their menus sat
        # on unproven states).  Round-1 pre-roll commits keep their
        # standing guarantee regardless.
        stats['drop_reasons']['cf_final_mismatch'] += 1
        verified = 0
        try:
            for r in range(len(wins)):
                w = wins[r]
                ok = True
                for own, own_str in (('atk', atk_str), ('def', dfn_str)):
                    gw = _window_owner_hits(batch, own_str, regs,
                                            w['first'], w['next_first'])
                    eh = round_roll_hits.get((r, own))
                    if eh is None:
                        eh = round_roll_hits.get((r + 1, own))
                    if eh is None:
                        if gw > 0:
                            ok = False
                        continue
                    if int(eh) != int(gw):
                        ok = False
                if not ok:
                    break
                verified = r + 1
        except Exception:
            verified = 0
        if verified:
            stats['detected']['cf_rounds_verified'] += verified
        # drive alloc/press/continue/retreat rounds are 1-BASED while
        # verified counts 0-based rounds: keep <= verified for the
        # 1-based kinds so the LAST verified round's drive samples
        # survive (gd resumes at 0-based verified = 1-based verified+1,
        # so no double emission)
        _one_based9 = ('casualty_alloc', 'siege_press',
                       'attacker_continue', 'defender_retreat')
        keep = [e for e in emitted
                if (e[0] in ('combat_card', 'defender_siege_choice')
                    and e[1] == 0)
                or (e[0] in _one_based9 and e[1] <= verified)
                or (e[0] not in _one_based9 + ('combat_card',
                                               'defender_siege_choice')
                    and e[1] < verified)]
        if os.environ.get('RX_CF_DEBUG'):
            print('[CF]   final-mismatch verified=%d kept=%s'
                  % (verified, [(e[0], e[1]) for e in keep]))
        _gd_rounds(verified)
        if os.environ.get('RX_CF_DEBUG'):
            print('[CF]   post-gd keep=%s'
                  % [(e[0], e[1]) for e in keep])
    # White Rider labels are emitted only at the authenticated live pause.
    # ONSLAUGHT magnitude by GESTURE (GAME19 T4 FUh K=3): when the
    # drive broke at a DIFFERENT effect (Heroic Death) the Onslaught
    # pending never surfaced - the parked-to-card units ARE the K
    # decision.  Guarded on an Onslaught-side commit + not already
    # emitted by the drive.
    if 'onslaught' not in _drive_fx9 and \
            (batch.get('start_line') or 0) not in \
            stats.get('_ons_lines', set()):
        # the resolver's _od stash may already have emitted this
        # batch's Onslaught (GAME01 T3 OMA K=4) - never double it
        if os.environ.get('RX_CF_DEBUG'):
            print('[ONS] enter line=%s fx=%s' % (batch.get('start_line'),
                                                 sorted(_drive_fx9)))
        try:
            _ons_cid9 = next(
                (c[sl] for c in cards for sl in ('fp', 'sh')
                 if c.get(sl) and ('Fighting_Uruk' in str(c[sl])
                                   or 'Orcs_Multiplying' in str(c[sl]))),
                None)
            if _ons_cid9 is not None:
                _kpark9 = 0
                for r9 in batch['records']:
                    if r9.get('op') != 'move':
                        continue
                    fk9 = rr.zone_kind(r9.get('from', ''))
                    if fk9[0] not in ('region', 'stronghold') \
                            or fk9[1] not in regs:
                        continue
                    if 'current sa card' not in \
                            str(r9.get('to', '')).lower() and \
                            'current fp card' not in \
                            str(r9.get('to', '')).lower():
                        continue
                    for t9 in _group_tokens(r9.get('obj', '')):
                        p9 = rr.parse_piece(t9)
                        if p9 and p9[0] == 'unit':
                            _kpark9 += p9[3]
                if _kpark9 > 0:
                    import game_state as gs9o
                    _own9 = (gs9o.Side.SHADOW
                             if batch['die_side'] == 'SHADOW'
                             else gs9o.Side.FREE_PEOPLES)
                    st_o = _recon_state(batch['truth_before'], batch.get('turn'))
                    _install_ledger_hands(st_o, batch)
                    _recovery_context(st_o, **{
                        'phase': 'combat_effect_choice',
                        'attacker_region': atk, 'defender_region': dfn,
                        'attacker_side': _own9,
                        'defender_side': (
                            gs9o.Side.FREE_PEOPLES
                            if _own9 == gs9o.Side.SHADOW
                            else gs9o.Side.SHADOW),
                        'deciding_side': _own9, 'is_siege': _sieg,
                        'effect_choice': {
                            'stage': 'post', 'owner': 'atk',
                            'effect': 'onslaught',
                            'card_id': _ons_cid9,
                            'max_count': _kpark9 + 2}})
                    la_o = [a for a in ai.get_legal_actions(st_o, _own9)
                            if a.action_type == 'combat_effect_choice']
                    if len(la_o) >= 2:
                        _pend_note(stats, 'combat_effect', 'open')
                    h_o = _match_index(
                        la_o, lambda a: int(a.params.get('count', -2))
                        == _kpark9)
                    if h_o is not None and len(la_o) >= 2:
                        stats['detected']['cc_onslaught_gesture'] += 1
                        s_o = _emit_sample(st_o, _own9, la_o, h_o,
                                           'combat_effect_choice',
                                           batch['turn'])
                        if s_o is not None:
                            keep.append(('combat_effect', 0, s_o))
                    else:
                        stats['drop_reasons']['ons_gesture_unmatch'] += 1
        except Exception as _eo9:
            stats['drop_reasons']['ons_gesture_exc:%s'
                                  % str(_eo9)[:12]] += 1
    for kind, _ri, _smp in keep:
        stats['kinds_emit'][kind] += 1
        _pend_note(stats, kind, 'emit')
    return [smp for (_k, _ri, smp) in keep]


# ==========================================================================
# NO-ROLL SIEGE WITHDRAWAL  (spec 1.216: retreat-into-siege, pre-round-1)
# --------------------------------------------------------------------------
# An attack declared and answered by an IMMEDIATE withdrawal into the
# stronghold never rolls a die: the batch classifies as bookkeeping (no kill,
# no chunk), so the combat family above never sees it.  The withdrawal
# gesture (defender units region R -> R's stronghold box, R not already
# besieged) evidences BOTH decisions: the attacker's army_attack TARGET and
# the defender's retreat_into_siege (plus a possible siege_trim).
# ==========================================================================
def _emit_siege_withdrawal(batch, stats):
    import ai
    import rendered_replay as rr
    from game_state import Side
    # USER truth-table #6 (game01_verification.md): the withdrawal
    # GESTURE (defender units region -> own stronghold box) is the
    # anchor - the attacker's move-in may coexist as an army-move
    # record and the die can be C (T4 DA, T7 HD); only fought-round
    # batches (roll_num) are excluded here (their withdrawals resolve
    # through the combat drive).
    if batch['kind'] == 'hunt':
        return []
    if any(r['kind'] == 'roll_num' for r in batch['rng']):
        return []
    if batch['die_face'] not in ('A', 'H', 'C', 'M', 'W'):
        return []
    besieged_before = set(rr.sanitize(x) for x in
                          (batch['truth_before'].get('besieged') or []))
    target = wside = None
    for r in batch['records']:
        if r.get('op') != 'move':
            continue
        fk = rr.zone_kind(r.get('from', ''))
        tk = rr.zone_kind(r.get('to', ''))
        if fk[0] != 'region' or tk[0] != 'stronghold' or fk[1] != tk[1]:
            continue
        for t in _group_tokens(r.get('obj', '')):
            p = rr.parse_piece(t)
            if p and p[0] == 'unit':
                target, wside = fk[1], _unit_side_str(p[1])
                break
        if target:
            break
    if target is None:
        return []
    if target in besieged_before:
        # the mirror's besieged set is ACCRETIVE (stale after a fall -
        # GAME19 T8 Minas Morgul, retaken by FP then besieged anew by
        # Shadow): a fresh enemy MOVE-IN this batch proves a NEW siege
        import game_state as _gsw9
        import ai as _aiw9
        _movein9 = False
        for r in batch['records']:
            if r.get('op') != 'move':
                continue
            fk9 = rr.zone_kind(r.get('from', ''))
            tk9 = rr.zone_kind(r.get('to', ''))
            if tk9[0] != 'region' or tk9[1] != target \
                    or fk9[0] not in ('region', 'stronghold') \
                    or fk9[1] == target:
                continue
            for t9 in _group_tokens(r.get('obj', '')):
                p9 = rr.parse_piece(t9)
                if p9 and p9[0] == 'unit' and \
                        _unit_side_str(p9[1]) == batch['die_side']:
                    _movein9 = True
        if not _movein9:
            return []
    if wside == batch['die_side']:
        return []                    # own-side box shuffle, not a withdrawal
    stats['detected']['siege_withdrawal'] += 1
    # W2-024 (2026-09-03): the attacker's SOURCE is written in the batch's
    # own move-in record (die-side units, region X -> target: the p.31
    # move-in after the withdrawal).  A target-only predicate is
    # ambiguous whenever two friendly armies border the target (SlickRick
    # L478 osgiliath + lossarnach -> minas tirith; Mol L460 three
    # candidates) and _match_index dropped the whole batch.  Match the
    # human's exact (from, to); fall back to the target-only predicate
    # only when NO move-in record exists (the victor-stay shape).  Never
    # widens: two distinct sources = still ambiguous = drop.
    _srcs24 = set()
    for r in batch['records']:
        if r.get('op') != 'move':
            continue
        fk4 = rr.zone_kind(r.get('from', ''))
        tk4 = rr.zone_kind(r.get('to', ''))
        if tk4[0] != 'region' or tk4[1] != target \
                or fk4[0] not in ('region', 'stronghold') \
                or fk4[1] == target:
            continue
        for t4 in _group_tokens(r.get('obj', '')):
            p4 = rr.parse_piece(t4)
            if p4 and p4[0] == 'unit' and \
                    _unit_side_str(p4[1]) == batch['die_side']:
                _srcs24.add(fk4[1])
                break
    atk_src = next(iter(_srcs24)) if len(_srcs24) == 1 else None
    st, side = _mk_state_with_die(batch, install_hands=True)
    if st is None:
        stats['drop_reasons']['sw_recon'] += 1
        return []
    global _MM_FACE
    _effc = _effective_lead_letter(batch)
    if _effc != batch['die_face']:
        _fc = _letter2face().get(_effc)
        if _fc is not None and st.dice.get(side):
            _set_effective_lead_face(st, side, _fc)
            _MM_FACE = _fc
            stats['detected']['sw_ring_ride:%s>%s' % (
                str(batch['die_face']).upper(), _effc)] += 1
    try:
        legal = ai.get_legal_actions(st, side)
    except Exception:
        stats['drop_reasons']['sw_legal_exc'] += 1
        return []

    def _is_attack24(a):
        if a.action_type != 'army_attack':
            return False
        if rr.sanitize(str(a.params.get('to', a.params.get(
                'to_region', '')))) != target:
            return False
        if atk_src is not None and \
                rr.sanitize(str(a.params.get('from', ''))) != atk_src:
            return False
        return True

    hidx = _match_index(legal, _is_attack24)
    if hidx is None:
        stats['drop_reasons']['sw_attack_unmatched' if atk_src is None
                              else 'sw_attack_src_unmatched'] += 1
        return []
    out = [_emit_sample(st, side, legal, hidx, 'army_attack', batch['turn'])]
    stats['kinds_emit']['army_attack'] += 1
    dfn_str = 'FREE_PEOPLES' if batch['die_side'] == 'SHADOW' else 'SHADOW'
    try:
        s2 = ai.apply_action(st, legal[hidx])
        for _ in range(4):
            pc = getattr(s2, 'pending_combat', None)
            if pc is None:
                break
            ph = pc.get('phase')
            ds = pc.get('deciding_side')
            dside = ds if isinstance(ds, Side) else side
            la = ai.get_legal_actions(s2, dside)
            if ph == 'defender_siege_choice':
                h2 = _match_index(la, lambda a: a.action_type ==
                                  'retreat_into_siege')
                if h2 is None:
                    break
                if len(la) >= 2:
                    stats['kinds_emit']['defender_siege_choice'] += 1
                    out.append(_emit_sample(s2, dside, la, h2,
                                            'defender_siege_choice',
                                            batch['turn']))
                s2 = ai.apply_action(s2, la[h2])
                continue
            if ph == 'siege_trim':
                eb, _rb, to_cas = _window_trim(batch, target, dfn_str)
                if to_cas:
                    stats['detected']['siege_trim_to_casualties'] += 1
                h2 = _match_index(la, lambda a: a.action_type ==
                                  'siege_trim' and
                                  int(a.params.get('elites_back', -1)) == eb)
                if h2 is None:
                    stats['drop_reasons']['cf_trim_unmatched'] += 1
                    break
                if len(la) >= 2:
                    stats['kinds_emit']['siege_trim'] += 1
                    out.append(_emit_sample(s2, dside, la, h2, 'siege_trim',
                                            batch['turn']))
                s2 = ai.apply_action(s2, la[h2])
                continue
            if ph == 'victor_advance':
                # truth rows T4-21/T3-16: after the garrison withdraws
                # into the stronghold the attacker chooses the entering
                # portion (or stays out) - the move-in gesture is the
                # decision
                _atk_r9 = rr.sanitize(str(
                    legal[hidx].params.get('from', '')))
                comp9 = _moved_group_composition(batch, _atk_r9, target)
                mv9 = comp9['reg'] + comp9['elite'] + \
                    comp9['leaders'] + comp9['nazgul']
                if mv9 > 0:
                    h2 = _match_army_move(la, s2, side, _atk_r9, target,
                                          comp9, atype='victor_advance')
                else:
                    h2 = _match_index(la, lambda a: a.action_type ==
                                      'victor_stay')
                if h2 is None:
                    stats['drop_reasons']['sw_advance_unmatched'] += 1
                    break
                if len(la) >= 2:
                    stats['kinds_emit']['victor_advance'] += 1
                    out.append(_emit_sample(s2, dside, la, h2,
                                            'victor_advance',
                                            batch['turn']))
                s2 = ai.apply_action(s2, la[h2])
                continue
            break
    except Exception as e:
        stats['drop_reasons']['sw_drive_exc:%s' % str(e)[:12]] += 1
    return out


# ==========================================================================
# HAND-LIMIT DISCARD IDENTITY  (spec 1.216 META: "NEVER auto or random pick
# â€” the discarded card identity is a critical decision")
# --------------------------------------------------------------------------
# The ledger records every VOLUNTARY in-hand kill (named title, or the
# face-down excess discard resolved through the raw Controls.296 commitment
# FIFO).  Each becomes a discard_card BC sample on the neighbouring board
# snapshot with the ledger-true hand; an unresolvable identity DROPS the
# sample (the ledger has already ambiguity-flagged the deck).  Random
# (forced) discards are RNG, never decisions, and are never recorded.
# ==========================================================================
def _emit_discards(batches, stats):
    import ai
    from game_state import Side
    if _LEDGER is None or not getattr(_LEDGER, 'discard_events', None):
        return []
    out = []
    for (side_tag, line, cid) in _LEDGER.discard_events:
        stats['detected']['discard_card'] += 1
        if cid is None:
            stats['drop_reasons']['discard_unresolved'] += 1
            continue
        host = None
        for b in batches:
            if b['start_line'] <= line < (b.get('end_line') or 10 ** 9):
                host = b
                break
            if b['start_line'] > line:
                host = b
                break
        if host is None:
            host = batches[-1]
        hand = _LEDGER.hand_at(side_tag, line)
        if cid not in hand:
            stats['drop_reasons']['discard_hand_desync'] += 1
            continue
        if len(hand) <= 6:
            # not a hand-limit context (a card-forced kill: the card's own
            # interior models it) â€” census, don't guess
            stats['drop_reasons']['discard_ctx_other'] += 1
            continue
        record = next((r for r in host['records'] if r.get('line') == line), None)
        snapshot = record.get('decision_truth') if record is not None else None
        if snapshot is None and host.get('record_truth_version'):
            stats['drop_reasons']['discard_no_record_snapshot'] += 1
            continue
        try:
            st = _recon_state(snapshot if snapshot is not None else host['truth_before'],
                              host.get('turn'))
        except Exception:
            stats['drop_reasons']['discard_recon'] += 1
            continue
        d_side = Side.FREE_PEOPLES if side_tag == 'fp' else Side.SHADOW
        _install_ledger_hands(st, host, at_line=line)
        _install_persistent_cards(st, line, killable=True)
        if record is not None and '_rx_pools' in record:
            st._rx_full_pools = record['_rx_pools']
            st.dice = _install_full_pools(st._rx_full_pools)
            st._rx_before_hunt_allocation = bool(record.get('_rx_before_hunt_allocation'))
        grant = host.get('_rx_stewards_gandalf')
        if d_side is Side.FREE_PEOPLES and grant is not None:
            prior_draws = [str(r.get('what', '')).lower()
                           for r in host['records'] if r.get('op') == 'draw'
                           and host['start_line'] < r.get('line', 0) < line]
            if prior_draws == ['free strategy card', 'free strategy card']:
                st._gandalf_draw_pending = dict(grant)
        st.pending_discard = {'side': d_side}
        try:
            legal = ai.get_legal_actions(st, d_side)
        except Exception:
            stats['drop_reasons']['discard_legal_exc'] += 1
            continue
        hidx = _match_index(legal, lambda a: a.action_type == 'discard_card'
                            and str(a.params.get('card_id')) == str(cid))
        if hidx is None:
            stats['drop_reasons']['discard_unmatched'] += 1
            continue
        stats['kinds_emit']['discard_card'] += 1
        out.append(_emit_sample(st, d_side, legal, hidx, 'discard_card',
                                host['turn']))
    return out


# ==========================================================================
# CHOOSE GUIDE: optional Fellowship phase and compulsory replacement
# --------------------------------------------------------------------------
# A closed optional phase can prove keeping the current guide. Compulsory
# replacement requires the previous guide to have left. Singleton menus,
# unfinished/contradictory phase histories, and convenience parking are omitted.
# ==========================================================================
def _emit_guides(batches, stats):
    """Guide labels require a literal nomination or a closed optional phase.

    Record snapshots precede gestures. Batch endpoints are too late for a
    turn-opening nomination and may include later companion separations.
    Silence is evidence of keeping the guide only across an observed turn
    opening and Hunt-allocation/action-roll boundary with unchanged company.
    """
    import ai
    import rendered_replay as rr
    from game_state import Side
    out = []
    first_stage = {}
    for batch in batches:
        first_stage.setdefault(batch['turn'], batch)

    def emit(batch, record, snapshot, chosen, optional, evidence):
        try:
            # A Hunt replacement can precede a delayed reveal. The physical
            # snapshot does not describe that rules continuation; use the
            # state reached by the observed tile/casualty sequence.
            driven = batch.get('_rx_hunt_guide_predecision', {}).get(record.get('line'))
            state = (ai.clone_state(driven) if driven is not None and not optional
                     else _recon_state(snapshot, batch.get('turn')))
        except Exception:
            stats['drop_reasons']['guide_recon'] += 1
            return
        comps = state.fellowship.companions
        best_level = max((c.level for c in comps), default=-1)
        best = [c.name for c in comps if c.level == best_level]
        if len(best) < 2:
            stats['detected']['guide_forced_no_tie'] += 1
            return
        chosen = rr.COMP_ALIAS.get(str(chosen).lower(), chosen)
        if chosen not in best:
            stats['drop_reasons']['guide_choice_not_best'] += 1
            return
        state.pending_guide_change = {'candidates': best}
        state._declare_phase = False
        state._eye_allocation_phase = optional
        state._fellowship_tail_phase = False
        state._rx_before_hunt_allocation = optional
        line = record.get('line', 0)
        _install_ledger_hands(state, batch, at_line=line)
        _install_persistent_cards(state, line, killable=True)
        if optional:
            # The future roll/Hunt allocation is not yet observed.
            state.dice = {Side.FREE_PEOPLES: [], Side.SHADOW: []}
            state.hunt_box = []
            state.eyes_allocated = 0
            state._rx_eyes_stamped = True
        elif '_rx_pools' in record:
            state._rx_full_pools = record['_rx_pools']
            state.dice = _install_full_pools(state._rx_full_pools)
        try:
            legal = ai.get_legal_actions(state, Side.FREE_PEOPLES)
        except Exception:
            stats['drop_reasons']['guide_legal_exc'] += 1
            return
        index = _match_index(legal, lambda action:
            action.action_type == 'choose_guide'
            and action.params.get('companion') == chosen)
        if index is None:
            stats['drop_reasons']['guide_unmatched'] += 1
            return
        stats['detected']['choose_guide_real'] += 1
        stats['kinds_emit']['choose_guide'] += 1
        row = _emit_sample(state, Side.FREE_PEOPLES, legal, index,
                           'choose_guide', batch['turn'])
        row['guide_evidence'] = {'kind': evidence, 'line': line}
        out.append(row)

    def incoming(records):
        return [r for r in records if r.get('_guide_gesture') and
                str(r.get('to', '')).strip().lower() == 'guide of the fellowship']

    def same_company(left, right):
        return set(left['fel'].get('comps', [])) == set(right['fel'].get('comps', []))

    for batch in batches:
        if batch.get('_rx_hunt_exclusion'):
            continue
        records = batch['records']
        prefix = [r for r in records if r.get('line', 0) < batch['start_line']]
        if batch is first_stage.get(batch['turn']):
            opening = batch.get('_guide_phase_start')
            closure = next((r for r in prefix if r.get('_guide_phase_close')), None)
            gestures = [r for r in prefix if r.get('_guide_gesture')]
            nominated = incoming(prefix)
            if opening is not None and closure is not None:
                closed = closure.get('decision_truth')
                # A nomination after Hunt allocation contradicts phase order;
                # it cannot justify either an earlier keep or a late choice.
                late = any(r['line'] > closure['line'] for r in gestures)
                unchanged = (closed is not None and same_company(opening, closed)
                             and not any(r.get('_fellowship_composition_changed')
                                         for r in prefix if r['line'] < closure['line']))
                if late:
                    stats['drop_reasons']['guide_late_nomination'] += 1
                elif not unchanged:
                    stats['drop_reasons']['guide_phase_composition_changed'] += 1
                elif gestures:
                    first = gestures[0]
                    before = first.get('decision_truth')
                    after = gestures[-1].get('guide_truth_after')
                    if (len(nominated) == 1 and before is not None and after is not None
                            and before['fel'].get('guide_provenance') in ('declared', 'forced')
                            and after['fel'].get('guide_provenance') == 'declared'
                            and same_company(before, after)
                            and all(before['fel'].get(key) == closed['fel'].get(key)
                                    for key in ('region', 'progress', 'hidden', 'corruption',
                                                'mordor', 'step'))):
                        piece = rr.parse_piece(str(nominated[0].get('obj', '')))
                        chosen = piece[1] if piece and piece[0] == 'named' else None
                        if chosen and rr._named_eq(after['fel'].get('guide', ''), chosen):
                            emit(batch, first, before, chosen, True, 'explicit_optional_nomination')
                        else:
                            stats['drop_reasons']['guide_unsettled_nomination'] += 1
                    else:
                        stats['drop_reasons']['guide_unsettled_nomination'] += 1
                elif (closed['fel'].get('guide_provenance') in ('declared', 'forced')
                      and opening['fel'].get('guide') == closed['fel'].get('guide')):
                    emit(batch, closure, closed, closed['fel']['guide'], True,
                         'closed_phase_keep')
                else:
                    stats['drop_reasons']['guide_unproven_current'] += 1
            elif gestures:
                stats['drop_reasons']['guide_phase_not_closed'] += 1

        # Inside-action replacement, including the optional opportunity after
        # a literal separation while the previous guide remains.
        # Pre-action losses can belong to the resumable Fellowship tail and
        # are deliberately excluded until that context is independently proven.
        body = [r for r in records if r.get('line', 0) >= batch['start_line']]
        nominations = incoming(body)
        if not nominations:
            continue
        if len(nominations) != 1:
            stats['drop_reasons']['guide_multiple_nominations'] += 1
            continue
        record = nominations[0]
        before = record.get('decision_truth')
        after = record.get('guide_truth_after')
        previous = batch['truth_before']['fel'].get('guide')
        piece = rr.parse_piece(str(record.get('obj', '')))
        chosen = piece[1] if piece and piece[0] == 'named' else None
        post_separation = (before is not None and previous in before['fel'].get('comps', [])
                           and _post_separation_names(batch, before['fel'].get('comps', []),
                                                      int(record['line'])))
        if (before is None or after is None or not chosen
                or (previous in before['fel'].get('comps', []) and not post_separation)
                or after['fel'].get('guide_provenance') != 'declared'
                or not rr._named_eq(after['fel'].get('guide', ''), chosen)):
            stats['drop_reasons']['guide_unproven_replacement'] += 1
            continue
        later = [r for r in body if r.get('line', 0) > record['line']]
        parked = any(r.get('op') == 'move'
                     and rr.zone_kind(r.get('from', ''))[0] == 'fellowship'
                     and rr.zone_kind(r.get('to', ''))[0] != 'fellowship'
                     and (p := rr.parse_piece(str(r.get('obj', ''))))
                     and p[0] == 'named' and rr._named_eq(p[1], chosen)
                     for r in later)
        if parked:
            stats['detected']['guide_parking'] += 1
            continue
        emit(batch, record, before, chosen, False,
             'explicit_post_separation_nomination' if post_separation else 'explicit_compulsory_nomination')
    return out


# ==========================================================================
# DWARVEN RING  (spec 1.216 META: "ring killed + which die rerolled")
# --------------------------------------------------------------------------
# Raw piece identity proves the target, original face, and prefix inventory.
# The enclosing action batch can belong to the other player.
def _emit_dwarven_rings(batch, pools, stats):
    """Emit only a raw-ID-proven ring target at its own decision prefix."""
    import ai
    import game_logic
    import game_state as gs
    global _RX_LINE_CUR
    out = []
    for record in batch['records']:
        if record.get('_rx_legacy_ring_hesitation'):
            continue
        if not (record.get('op') == 'move'
                and str(record.get('obj', '')).strip().lower() == 'dwarven ring'
                and str(record.get('from', '')).strip().lower() in ('fp dice', 'sa dice')
                and str(record.get('to', '')).lower().startswith('current')) and not record.get('_rx_unstaged_ring_payment'):
            continue
        stats['detected']['dwarven_ring_use'] += 1
        proof = record.get('_rx_dwarven_physical')
        def refuse(reason):
            raise _Quarantine('coverage_dwarven_ring_unproven',
                              {'detail': reason, 'line': record.get('line')})
        if proof is None:
            refuse('raw_identity_missing')
        if proof.get('timing_variant'):
            stats['drop_reasons'][proof.get('exclusion_reason') or 'dwarven_ring_timing_variant'] += 1
            continue
        side = gs.Side[proof['side']]
        old = _letter2face()[proof['old_face']]
        classes = game_logic._DWARVEN_FACE_CLASSES
        results = [_letter2face()[r['face']] for r in proof['results']]
        if (any(not (classes[f] & classes[old]) for f in results[:-1])
                or classes[results[-1]] & classes[old]):
            refuse('reroll_class_or_completion')
        decision_line = record.get('_rx_dwarven_decision_line', record['line'])
        snapshot = record['decision_truth']
        state = _recon_state(snapshot, batch['turn'])
        state.dice = {s: [] for s in gs.Side}
        target_index = None
        for sd in (gs.Side.FREE_PEOPLES, gs.Side.SHADOW):
            for pid, letter in proof['before'][sd.name].items():
                die = gs.ActionDie(gs.FP_FACES if sd == gs.Side.FREE_PEOPLES else gs.SH_FACES, sd)
                die.current = _letter2face()[letter]
                die.used = letter == 'E'
                if sd == side and pid == proof['target_id']:
                    target_index = len(state.dice[sd])
                state.dice[sd].append(die)
        for pid in proof['allocated']:
            die = gs.ActionDie(gs.SH_FACES, gs.Side.SHADOW)
            die.current = gs.ActionDieFace.EYE
            die.used = True
            state.dice[gs.Side.SHADOW].append(die)
        state.hunt_box = [d for ds in state.dice.values() for d in ds
                          if d.current == gs.ActionDieFace.EYE]
        state.eyes_allocated = len(proof['allocated'])
        state._rx_eyes_stamped = True
        state._rx_full_pools = None
        state.fp_dwarven_rings = proof['inventory']['FREE_PEOPLES']
        state.sh_dwarven_rings = proof['inventory']['SHADOW']
        _install_ledger_hands(state, batch, at_line=decision_line)
        _install_ledger_discards(state, decision_line)
        state._rx_dfu_stamped = True
        _ensure_decks(state)
        _install_persistent_cards(state, decision_line, killable=True)
        legal = ai.get_legal_actions(state, side)
        matches = [i for i, action in enumerate(legal)
                   if action.action_type == 'dwarven_ring'
                   and action.params.get('old_face') == old.value
                   and action.params.get('die_idx') == target_index]
        if len(matches) != 1:
            refuse('public_ring_action_unmatched')
        # Validate the logged final RNG result through the public rules too.
        check = ai.clone_state(state)
        game_logic.use_dwarven_ring(check, side, die_idx=target_index,
                                   forced_face=results[-1])
        stats['kinds_emit']['dwarven_ring'] += 1
        previous_line = _RX_LINE_CUR
        _RX_LINE_CUR = decision_line
        try:
            out.append(_emit_sample(state, side, legal, matches[0],
                                    'dwarven_ring', batch['turn']))
        finally:
            _RX_LINE_CUR = previous_line
    return out

# ==========================================================================
# HUNT-RESOLUTION FAMILY  (emission spec 1.216 item 1)
# --------------------------------------------------------------------------
# The FP hunt-damage decision: after the Shadow's hunt tile lands DMG points,
# the FP chooses HOW to absorb it â€” take it all as corruption, sacrifice a
# companion (the guide deliberately, or a random victim), or use a Merry/Pippin
# guide's separation ability (-1, CHAINING to the next guide) â€” plus Gollum's
# voluntary reveal (-1 when not required) and the placement region of each
# separated hobbit (in Mordor they DIE instead of separating).
#
# The certified hunt driver (rendered_census._drive_hunt, skip_auto_sacrifice)
# reconstructs the exact pre-resolution engine state (accumulated
# _pending_hunt_damage, guide, companions) from the client mirror under forced
# RNG.  We then open the engine's OWN resolution menu (_settle_hunt_damage) and
# walk each pending_hunt phase, legality-matching the human's choice against the
# batch delta: corruption walk / companion elimination / hobbit separation /
# reveal flip.  Emit-or-DROP per phase (purity); the guide identity comes
# from the mirror snapshot ('guide', W3-018) so Gollum-reveal /
# guide-separate menus open correctly.
# ==========================================================================
def _hunt_tile_items(batch):
    """W3-011: the batch's drawn hunt tiles (rng items) in line order."""
    return sorted([it for it in batch['rng'] if it.get('kind') == 'draw_tile'],
                  key=lambda it: int(it.get('line') or 0))


def _hunt_live_tiles(batch):
    """W3-011: the tiles that RESOLVED.  A tile set aside by Mithril Coat
    and Sting (the card leaves its Board slot between two draws:
    Genoah77 L575 Eye set aside, '2' redrawn at L579) never reaches a
    hunt decision - the redrawn tile is the live one (W1-017's
    mithril_redraw evidence rule: card leaves BoardN + a second draw)."""
    tiles = _hunt_tile_items(batch)
    if len(tiles) < 2:
        return tiles
    mith = []
    for r in batch['records']:
        if r.get('op') != 'move':
            continue
        if 'mithril coat' not in str(r.get('obj', '')).lower():
            continue
        if not re.match(r'^board\d*$',
                        str(r.get('from', '')).strip().lower()):
            continue
        mith.append(int(r.get('line') or 0))
    if not mith:
        return tiles
    live = []
    for k, t in enumerate(tiles):
        lo = int(t.get('line') or 0)
        hi = (int(tiles[k + 1].get('line') or 0)
              if k + 1 < len(tiles) else None)
        if hi is not None and any(lo < m < hi for m in mith):
            continue                       # set aside, redrawn
        live.append(t)
    return live


def _hunt_windows(live_tiles):
    """W3-011: [(lo, hi)] line windows, one per live tile - the human's
    gestures for tile k are the records from its draw line up to (not
    including) the next live tile's draw line."""
    lines = [int(t.get('line') or 0) for t in live_tiles]
    out = []
    for k, ln in enumerate(lines):
        hi = lines[k + 1] if k + 1 < len(lines) else 10 ** 9
        out.append((ln, hi))
    return out


def _comp_names_in(obj):
    """Companion names (COMP_ALIAS-normalised) named by a piece token or a
    'group (...)' token."""
    import rendered_replay as rr
    s = str(obj or '').strip()
    if s.lower().startswith('group ('):
        toks = rr.split_group(s[s.find('(') + 1:s.rfind(')')])
    else:
        toks = [s]
    out = []
    for t in toks:
        p = rr.parse_piece(t)
        if p and p[0] == 'named':
            out.append(rr.COMP_ALIAS.get(p[1].lower(), p[1]))
    return out


def _comp_exit_lines(batch):
    """W3-011: {companion: first line} where it leaves a Fellowship zone
    (Fellowship Of The Ring / Guide Of The Fellowship) for a non-
    fellowship zone (casualties, a card slot, a region), is killed
    ('kills Gimli in Guide Of The Fellowship', SlickRick L1370; kill
    records carry 'pieces'), or is DRAWN by the Shadow as the random
    casualty ('draws Strider', CircleDakkon L807 - the draw is the only
    exit the log writes for him)."""
    import rendered_replay as rr
    out = {}
    for r in batch['records']:
        op = r.get('op')
        names = []
        if op == 'move':
            if rr.zone_kind(str(r.get('from', '')))[0] != 'fellowship':
                continue
            if rr.zone_kind(str(r.get('to', '')))[0] == 'fellowship':
                continue
            names = _comp_names_in(r.get('obj', ''))
        elif op == 'kill':
            for tok in (r.get('pieces') or []):
                names.extend(_comp_names_in(tok))
        elif op == 'draw':
            v = str(r.get('what', '') or '').strip()
            if v.lower() in rr.COMP_ALIAS:
                names = [rr.COMP_ALIAS.get(v.lower(), v)]
        else:
            continue
        ln = int(r.get('line') or 0)
        for nm in names:
            if nm not in out:
                out[nm] = ln
    return out


def _hunt_window_ev(batch, lo, hi, left_all, seps_all, exits):
    """W3-011: the human's evidence for ONE drawn tile = the records in
    [lo, hi).  left = victims whose exit line falls in the window; seps =
    their region placements; guide_ev = 'from Guide Of The Fellowship'
    exits; random_ev = 'draws <companion>' lines (rng draw_companion
    items AND the raw draw record's 'what' - the record carries no
    'value', W3-021); corr = Corruption-marker walk; flips = Fellowship
    flips."""
    import rendered_replay as rr
    ev = {}
    ev['left'] = [c for c in left_all if lo <= exits.get(c, -1) < hi]
    ev['seps'] = dict((c, seps_all[c]) for c in ev['left'] if c in seps_all)
    guide_ev, random_ev = set(), set()
    flips = 0
    corr = 0
    _up9 = False
    for r in batch['records']:
        ln = int(r.get('line') or 0)
        if not (lo <= ln < hi):
            continue
        op = r.get('op')
        if op == 'move' and 'Guide Of The Fellowship' in \
                str(r.get('from', '')):
            for nm in _comp_names_in(r.get('obj', '')):
                guide_ev.add(nm)
        if op == 'draw':
            v = str(r.get('what', '') or r.get('value', '') or '').strip()
            if v:
                random_ev.add(rr.COMP_ALIAS.get(v.lower(), v))
        if op == 'flip' and str(r.get('obj', '')).strip().lower() in (
                'the fellowship', 'fsp'):
            flips += 1
        if op == 'move' and 'corruption' in str(r.get('obj', '')).lower():
            _mf9 = re.search(r'fellowship track\s+(\d+)',
                             str(r.get('from', '')).lower())
            _mt9 = re.search(r'fellowship track\s+(\d+)',
                             str(r.get('to', '')).lower())
            if _mf9 and _mt9:
                _d9 = int(_mt9.group(1)) - int(_mf9.group(1))
                if _d9 < 0 and not _up9:
                    # W3-011: a hunt never LOWERS corruption - a downward
                    # move BEFORE any upward one in the window corrects an
                    # earlier over-count, not this tile (LuckyRuds L135-139
                    # "i took 1corr and gandalf, only 1tile drawn, so now i
                    # take 1corr": 1->0 then 0->1 = the Eye's +1, not a
                    # client-null); a downward move AFTER an upward one
                    # still corrects that upward one (net walk).
                    continue
                _up9 = _up9 or _d9 > 0
                corr += _d9
    for it in batch['rng']:
        if it.get('kind') == 'draw_companion' \
                and lo <= int(it.get('line') or 0) < hi:
            v = str(it.get('value', '') or '').strip()
            if v:
                random_ev.add(rr.COMP_ALIAS.get(v.lower(), v))
    ev['guide_ev'] = guide_ev
    ev['random_ev'] = random_ev
    ev['flips'] = flips
    ev['corr'] = corr
    return ev


def _fel_place_dest(batch, after_line):
    """W3-011: the NET destination of the Fellowship pawn's reveal
    placement drag after *after_line* (the LAST 'The Fellowship' region
    move, as _emit_reveal_placements reads it), or None."""
    import rendered_replay as rr
    dest = None
    for r in batch['records']:
        if r.get('op') != 'move' or int(r.get('line') or 0) <= after_line:
            continue
        if 'The Fellowship' not in str(r.get('obj', '')):
            continue
        if 'Mordor' in str(r.get('to', '')):
            continue
        if rr.zone_kind(str(r.get('to', '')))[0] != 'region':
            continue
        dest = rr.sanitize(str(r.get('to', '')))
    return dest


def _hunt_tile_damaging9(t):
    """W3-011: does this drawn tile deal damage (an Eye / a numeric tile
    above 0)?  '0r' and the special tiles (Shelob's Lair, Elven Rope...)
    do not."""
    v = str(t.get('value', '') or '')
    return ('eye' in v.lower()
            or (v.strip('r!').isdigit() and int(v.strip('r!')) > 0))


def _hunt_tile_dmg9(tile, ev, rc):
    """W3-011: the damage the drawn tile dealt - its printed value for a
    numeric tile; for an Eye / special tile the damage the human
    absorbed (Corruption walk + the Level of every victim)."""
    v = str(tile.get('value', '') or '').strip().lower()
    c = v.strip('r!')
    if c.isdigit():
        return max(1, int(c))
    lev = getattr(rc, '_COMP_LEVEL', {})
    return max(1, int(ev.get('corr', 0) or 0) +
               sum(lev.get(n, 1) for n in ev.get('left', [])))


def _logged_null_hunt_reduction(state, batch, tile, stats):
    """Drive a literal single-point table-card cancellation before placement.

    This narrow path requires the card to move from the table into Current FP
    Card after the draw and then be discarded. It never infers a use from an
    unchanged corruption marker alone. Other mitigation histories remain with
    their existing emitter and must pass the separate reveal-state check.
    """
    import ai
    import game_logic as gl
    from game_state import Side
    if _TABLE is None or int(getattr(state, '_pending_hunt_damage', 0) or 0) != 1:
        return None
    lo = int(tile['line'])
    hi = min([int(t['line']) for t in _hunt_tile_items(batch) if int(t['line']) > lo]
             + [int(batch.get('end_line') or 10 ** 9)])
    used = set()
    for record in batch['records']:
        if not lo < int(record.get('line') or 0) < hi:
            continue
        titles = []
        if (record.get('op') == 'move' and _zone_is_table(record.get('from',''))
                and str(record.get('to','')).lower() == 'current fp card'):
            titles = [record.get('obj','')]
        elif record.get('op') == 'kill' and _zone_is_table(record.get('zone','')):
            titles = record.get('pieces',[])
        for title in titles:
            kind = _title_to_ptype(title)
            if (kind in ('AXE_AND_BOW', 'HORN_OF_GONDOR')
                    and _TABLE.on_at(kind, int(batch['start_line']))
                    and _TABLE.killed_in(kind, lo, hi)):
                used.add(kind)
    if len(used) != 1:
        return None
    before = ai.clone_state(state)
    _install_persistent_cards(before, int(batch['start_line']))
    if not gl._offer_hunt_reduction(before):
        return None
    cid = _TABLE.cid[next(iter(used))]
    legal = ai.get_legal_actions(before, Side.FREE_PEOPLES)
    matches = [i for i, a in enumerate(legal) if a.action_type == 'card_option'
               and (a.params.get('choices') or {}).get('use_hunt_reduction') == 1
               and (a.params.get('choices') or {}).get('card_id') == cid]
    if len(matches) != 1:
        return None
    chosen = matches[0]
    after = ai.apply_action(before, legal[chosen])
    if (after.fellowship.corruption != state.fellowship.corruption
            or getattr(after, '_pending_hunt_damage', 0) or after.pending_hunt):
        return None
    from rendered_hunt_observation import mark
    mark(before, lo + 1, 'after_draw')
    mark(after, lo + 1, 'after_draw')
    sample = _emit_sample(before, Side.FREE_PEOPLES, legal, chosen,
                          'card_option', batch['turn'])
    stats['kinds_emit']['hunt_reduction'] += 1
    stats['kinds_emit']['hunt_reduction_use'] += 1
    batch['_rx_hunt_reduction_owned'] = cid
    return after, sample


def _emit_hunt_resolution(batch, stats):
    import copy as _cp9
    import random as _rnd9
    _rng_before_hunt = _rnd9.getstate()
    import ai
    import game_logic as gl
    import game_state as gs
    import rendered_census as rc
    import rendered_replay as rr
    from game_state import Side
    if batch['kind'] != 'hunt' or batch.get('die_side') != 'FREE_PEOPLES':
        return []
    batch.pop('_rx_reveal_predecision', None)
    batch.pop('_rx_hunt_guide_predecision', None)
    batch.pop('_rx_hunt_reduction_owned', None)
    st, _side = _mk_state_with_die(batch)
    if st is None:
        return []
    # Route cards are public table state, also in the independently driven
    # hunt. Do not fabricate a missing Balrog decision by driving without it.
    _install_persistent_cards(st, batch.get('start_line') or 0, route_only=True)
    # W3-018: the recon carries the log guide (mirror snapshot 'guide');
    # the former _guide_at timeline override ignored the initial guide,
    # guide-box exits and kills (MadRobin L66/67: Legolas dragged in and
    # straight back out would have displaced Boromir).
    _prepare_fellowship_dice(st, batch, pending=True)
    # W3-011: ONE owner per drawn tile.  The walk drives the FIRST live
    # tile through the engine (rc._drive_hunt) and then follows the
    # human's chain tile by tile: every further tile the log shows
    # (p.42 draw for a reveal in / a placement into a Shadow stronghold,
    # or a Mithril redraw) is fed to the engine as the ONLY tile in the
    # pool at the moment it draws, so the engine's chain carries the
    # human's tiles and never a re-drawn phantom (the 1.136 recycle put
    # the used tile back: hrafhraf L337 '2r' then Moria '1' walked as
    # '2r' twice).  A tile Mithril set aside is not live (Genoah77 L570
    # Eye -> '2').  The walk marks batch['_rx_huntres'] with the tiles
    # it settled; _emit_cardhunt_arms owns only the rest.
    live_tiles = _hunt_live_tiles(batch)
    _all_tiles9 = _hunt_tile_items(batch)
    n_live = len(live_tiles)
    drive_batch = batch
    if live_tiles and _all_tiles9 and live_tiles[0] is not _all_tiles9[0]:
        stats['detected']['huntres_mithril_redraw'] += 1
        drive_batch = dict(batch)
        drive_batch['rng'] = [it for it in batch['rng']
                              if it.get('kind') != 'draw_tile'
                              or it is live_tiles[0]]
    try:
        status, diffs, reason = rc._drive_hunt(st, drive_batch)
    except Exception:
        stats['drop_reasons']['huntres_drive_exc'] += 1
        return []
    if status != 'drove':
        stats['drop_reasons']['huntres_skip:%s' % str(reason or status)[:14]] \
            += 1
        return []
    from fellowship_action_timing import finish_if_settled
    finish_if_settled(st)
    if any(d.get('resync') for d in diffs):
        stats['drop_reasons']['huntres_divergent'] += 1
        return []
    dmg = int(getattr(st, '_pending_hunt_damage', 0) or 0)
    _wins9 = _hunt_windows(live_tiles) or [(0, 10 ** 9)]
    if dmg <= 0 and not (n_live > 1
                         and getattr(st, '_reveal_place_due', False)):
        # 0-success hunt: no resolution (a damage-less reveal tile with a
        # further stronghold draw continues below through the placement).
        # W3-011: the walk OWNS no tile it did not resolve - a drawn tile
        # the engine priced at 0 (a Mordor Eye with the recon's hunt box
        # empty: FateevVV L469 'Eye' "1", +1 Corruption; a re-rolled hunt
        # the drive misread, W3-002: bowzo L84 '1r') is left to
        # _emit_cardhunt_arms, which reads the damage from the log.
        _dis9 = any(_hunt_tile_damaging9(t) for t in live_tiles[:1])
        if _dis9:
            stats['detected']['huntres_zero_dmg_disowned'] += 1
        batch['_rx_huntres'] = {'tiles_done': 0 if _dis9 else 1,
                                'n_live': n_live,
                                'lines': [w[0] for w in _wins9],
                                'zero_dmg': bool(_dis9)}
        if not _dis9 and live_tiles:
            _remember_reveal_predecision(batch, st, live_tiles[0])
        return []
    if dmg > 0:
        stats['detected']['hunt_resolution'] += 1
    # ---- evidence (PRE vs POST), per drawn tile (W3-011) --------------
    tb = batch['truth_before']['fel']
    ta = batch['truth']['fel']
    before = [rr.COMP_ALIAS.get(c.lower(), c) for c in tb.get('comps', [])]
    after = [rr.COMP_ALIAS.get(c.lower(), c) for c in ta.get('comps', [])]
    left_all = [c for c in before if c not in after]
    seps_all = {}
    for (nm, reg) in _sep_moves(batch):
        seps_all[rr.COMP_ALIAS.get(nm.lower(), nm)] = reg
    # follow the placement DRAG to its net destination (GAME19 T7:
    # Merry Parth Celebrant -> FANGORN)
    for nm9 in list(seps_all):
        cur9 = rr.sanitize(str(seps_all[nm9]))
        _seen9 = set()
        moved9 = True
        while moved9 and cur9 not in _seen9:
            _seen9.add(cur9)
            moved9 = False
            for r9 in batch['records']:
                if r9.get('op') != 'move':
                    continue
                if rr.COMP_ALIAS.get(
                        str(r9.get('obj', '')).strip().lower()) != nm9:
                    continue
                if rr.sanitize(str(r9.get('from', ''))) == cur9 and \
                        rr.zone_kind(str(r9.get('to', '')))[0] == \
                        'region':
                    cur9 = rr.sanitize(str(r9.get('to', '')))
                    moved9 = True
                    break
        seps_all[nm9] = cur9
    # W3-011 (W3-007 placement): a hunt-DRAWN victim leaves through the
    # drawn-card slot ('draws Pippin' -> 'moves Pippin from Current FP
    # Card to South Downs'), not from a fellowship zone, so _sep_moves
    # never saw its placement and the hobbit_place row was lost once the
    # forced victim was the human's (before, the engine's own random pick
    # hid this).  The placement = the victim's LAST region move in the
    # batch, which also settles a drag that revisits a region (CircleDakkon
    # L235-L243 Pippin South Downs -> ... -> BUCKLAND; LuckyRuds L387-L389
    # Merry Fangorn -> Westemnet -> EDORAS; Vayze L535 Merry -> Minas
    # Tirith).
    for nm9 in left_all:
        last9 = None
        for r9 in batch['records']:
            if r9.get('op') != 'move':
                continue
            if nm9 not in _comp_names_in(r9.get('obj', '')):
                continue
            tk9 = rr.zone_kind(str(r9.get('to', '')))
            if tk9[0] in ('region', 'stronghold') and tk9[1]:
                last9 = rr.sanitize(str(tk9[1]))
        if last9 is not None:
            seps_all[nm9] = last9
    exits = _comp_exit_lines(batch)
    _unw9 = [c for c in left_all if c not in exits]
    if _unw9:
        # a companion gone from the snapshot with no exit record in the
        # batch (a pre-hunt separation the mirror folded in) is not a
        # hunt victim: it belongs to no tile window
        stats['detected']['huntres_left_unwindowed'] += len(_unw9)
    revealed = bool(tb.get('hidden')) and not bool(ta.get('hidden'))
    _tb_c9 = int(tb.get('corruption', 0) or 0)
    _corr_total9 = int(ta.get('corruption', 0) or 0) - _tb_c9
    # USER truth-table fix #7 (GESTURE-primary arm evidence): a victim
    # leaving via 'from Guide Of The Fellowship' = GUIDE sacrifice; a
    # victim named on a 'draws <companion>' line = RANDOM.  The old
    # recon-guide comparison mislabeled T1 (Gandalf guide-sac tagged
    # random because the reconstructed guide defaulted wrong).
    # W3-018: Gollum's reveal-to-reduce is evidenced by the GESTURES, not
    # by the mirror's hidden flags alone: on the Mordor track a 'flips The
    # Fellowship' line is directionless and the mirror pins hidden=True in
    # both snapshots, so `revealed` was always False there and every
    # Mordor Gollum decision would label no_reveal (GenericPie L1086 '2r',
    # chat "ok i'll reveal", +1 Corruption, flip L1092).  Evidence = an
    # odd number of Fellowship flips AFTER the tile draw, corroborated by
    # the Corruption walk after the draw (damage - 1 on a reveal, the full
    # damage otherwise); disagreeing evidence drops the phase - never a
    # wrong label (Genoah77 L852 walks the full +4 and flips at L865 after
    # the chat; Wind362 reveals via 'flips FSP' = W3-001's dialect,
    # counted here since W3-001: gus833 L784 / L984).
    # W3-011: the evidence is read per tile WINDOW (draw line to the next
    # live draw line): LuconiusRex L201 Eye (+1 Corruption L211) then the
    # Moria '2' (Gandalf guide sacrifice L218) are two decisions.
    def _win_state9(k):
        lo, hi = _wins9[min(k, len(_wins9) - 1)]
        ev = _hunt_window_ev(batch, lo, hi, left_all, seps_all, exits)
        sq = [c for c in ev['left'] if c in ev['seps']]
        eq = [c for c in ev['left'] if c not in ev['seps']]
        # CLIENT-NULL hunt: no companion left and no corruption taken -
        # the human's interposes (Axe and Bow reduction, Mithril) consumed
        # the damage.  The recon lacks those table cards, so the
        # CORRUPTION arm would be a phantom (GAME01 T6 after the hunt-box
        # seeding made this hunt drive) - apply it silently, emit nothing
        # for it.  Single-tile batches keep the snapshot delta (mirror
        # truth), chain windows read their own Corruption walk.
        cn = (not ev['left'] and
              (_corr_total9 <= 0 if n_live <= 1 else ev['corr'] <= 0))
        return (sq, eq, ev['guide_ev'], ev['random_ev'], ev['corr'],
                ev['flips'], cn)
    k = 0
    (sep_queue, elim_queue, guide_ev, random_ev, _corr_walk9, _fel_flips9,
     _client_null9) = _win_state9(k)
    _gollum_revealed9 = revealed or (_fel_flips9 % 2 == 1)
    _in_mordor9 = bool(tb.get('mordor')) or bool(
        getattr(st.fellowship, 'mordor_entered', False))

    worn_window = None
    def _apply9(action, victim=None):
        nonlocal worn_window
        # W3-011: seed the engine's pool with the human's NEXT tile so a
        # p.42 stronghold draw inside this apply yields it (an Eye = no
        # effect there); with no logged tile left the pool stays empty
        # and the 1.136 recycle is blocked - the engine can never draw a
        # tile the human did not.  A random casualty is forced to the
        # companion the log names (1.24 forced_companion) so the chain
        # state after it is the human's.
        nxt = live_tiles[k + 1] if k + 1 < n_live else None
        fo9 = (rc._tile_obj(gs, str(nxt.get('value', '')).strip().lower())
               if nxt is not None else None)
        st.hunt_pool = [fo9] if fo9 is not None else []
        act = action
        if victim is not None:
            act = _cp9.copy(action)
            act.params = dict(action.params)
            act.params['forced_companion'] = victim
        worn_line = None
        if (_REACTIVE_TABLE is not None and _LEDGER is not None
                and getattr(_LEDGER, 'enabled', False)
                and action.action_type in ('hunt_sacrifice_guide', 'hunt_sacrifice_random')):
            worn_victim = victim or getattr(st.fellowship.guide, 'name', None)
            worn_line = exits.get(worn_victim)
            if worn_line is not None:
                from rendered_reactive_cards import prepare_worn_trigger, worn_casualty_window
                worn_window = worn_casualty_window(batch, worn_line,
                    {name: exits[name] for name in left_all if name in exits}, live_tiles, _REACTIVE_TABLE)
                prepare_worn_trigger(st, batch, *worn_window[:2], _REACTIVE_TABLE)
        _och9, _orc9 = _rnd9.choice, gl._recycle_hunt_pool
        _ocr9 = gl.pick_random_card
        def _worn_random9(candidates, ctx=''):
            if worn_window is not None and ctx == 'WWSAT hand discard':
                from rendered_reactive_cards import recorded_hand_discard, Unproven
                _line, card_id = recorded_hand_discard(batch, *worn_window[:2])
                if card_id not in candidates:
                    raise Unproven('worn_recorded_card_not_available')
                return card_id
            return _ocr9(candidates, ctx)
        gl.pick_random_card = _worn_random9
        _rnd9.choice = lambda s9: s9[-1]
        gl._recycle_hunt_pool = lambda _s9: 0
        try:
            st2 = ai.apply_action(st, act)
        finally:
            _rnd9.choice, gl._recycle_hunt_pool = _och9, _orc9
            gl.pick_random_card = _ocr9
        if worn_window is not None:
            from rendered_reactive_cards import drain_worn, Unproven
            try:
                st2, worn_samples = drain_worn(st2, batch, stats, *worn_window[:2],
                                              complete_window=worn_window[2])
            except Unproven as error:
                stats['drop_reasons'][str(error)] += 1
                raise
            out.extend(worn_samples)
            if not getattr(st2, '_wwsat_owed', None):
                worn_window = None
        consumed = fo9 is not None and not getattr(st2, 'hunt_pool', None)
        return st2, consumed
    # ---- walk the engine's own resolution menu ------------------------
    out = []
    if _client_null9:
        reduced = _logged_null_hunt_reduction(st, batch, live_tiles[0], stats)
        if reduced is not None:
            st, sample = reduced
            out.append(sample)
    gl._settle_hunt_damage(st)
    _sacked9 = False
    for _depth in range(16):
        option = getattr(st, 'pending_card_option', None) or {}
        if any('balrog_declare_draw' in choice
               for choice in option.get('options', [])):
            from rendered_balrog import choose_pending, apply_observed, Unproven
            from rendered_hunt_observation import mark
            try:
                menu, index, boundary, evidence = choose_pending(
                    st, batch, int(live_tiles[k]['line']) + 1)
                mark(st, boundary, 'before_draw')
                sample = _emit_sample(st, Side.SHADOW, menu, index,
                                      'card_option', batch['turn'])
                sample['reactive_evidence'] = evidence
                st, consumed = apply_observed(st, menu[index], batch, k + 1, live_tiles)
            except (Unproven, ValueError) as error:
                if str(error) == 'balrog_reveal_draw_source_unproven':
                    # The client may resolve Moria first and use Balrog later,
                    # or leave the order unrecorded. Do not manufacture a
                    # pre-draw choice from either history. This replay state
                    # is local: later batches reconstruct their own physical
                    # snapshots and still face every hand/pool/board gate.
                    batch['_rx_hunt_exclusion'] = {
                        'reason': str(error), 'start_line': batch['start_line'],
                        'end_line': batch['end_line'],
                        'scope': 'hunt_chain_and_related_targets',
                    }
                    stats['drop_reasons']['local_hunt:' + str(error)] += 1
                    for key in ('_rx_huntres', '_rx_reveal_predecision', '_rx_hunt_guide_predecision', '_rx_hunt_reduction_owned'):
                        batch.pop(key, None)
                    _rnd9.setstate(_rng_before_hunt)
                    return []
                raise _Quarantine('coverage_balrog_window_unresolved', {
                    'batch_start_line': batch.get('start_line'),
                    'tile_line': live_tiles[k].get('line'), 'reason': str(error),
                }) from error
            out.append(sample)
            stats['kinds_emit']['balrog_use' if evidence['used'] else 'balrog_decline'] += 1
            if consumed:
                k += consumed
                (sep_queue, elim_queue, guide_ev, random_ev, _corr_walk9,
                 _fel_flips9, _client_null9) = _win_state9(k)
                _gollum_revealed9 = revealed or (_fel_flips9 % 2 == 1)
                _sacked9 = False
            continue
        if k < n_live:
            _remember_reveal_predecision(batch, st, live_tiles[k])
        ph = getattr(st, 'pending_hunt', None)
        if ph is None and getattr(st, 'pending_guide_change', None):
            # W3-011: the guide died as a casualty and the engine asks
            # for the new guide among tied companions (RoySubs 09-11
            # L539 Strider random, L543 'Boromir ... to Guide Of The
            # Fellowship'); resolve it from the human's gesture so the
            # chain reaches the placement and the next tile.  The
            # gesture = the FIRST 'to Guide Of The Fellowship' move after
            # the latest exit of a companion the engine has already lost
            # (the same chain may re-guide again after a later death:
            # 0911 L547 Boromir sacrificed, L549 Legolas to Guide - the
            # last gesture would choose the wrong companion).  The
            # choose_guide ROW is _emit_guides' (see W3-023: its parking
            # filter drops this one).
            _cur9 = set(rr.COMP_ALIAS.get(str(c.name).lower(), str(c.name))
                        for c in st.fellowship.companions)
            _lo9 = max([_wins9[k][0]] + [int(exits[c]) for c in before
                                          if c not in _cur9 and c in exits])
            _gg9 = [r for r in batch['records']
                    if r.get('op') == 'move'
                    and int(r.get('line') or 0) > _lo9
                    and str(r.get('to', '')).strip().lower()
                    == 'guide of the fellowship']
            _gn9 = None
            if _gg9:
                _p9 = rr.parse_piece(str(_gg9[0].get('obj', '')))
                if _p9 and _p9[0] == 'named':
                    _gn9 = rr.COMP_ALIAS.get(_p9[1].lower(), _p9[1])
            try:
                _gl9 = [a for a in ai.get_legal_actions(st, Side.FREE_PEOPLES)
                        if a.action_type == 'choose_guide']
            except Exception:
                stats['drop_reasons']['huntres_legal_exc'] += 1
                break
            _gi9 = (_match_index(_gl9, lambda a: rr._named_eq(
                str(a.params.get('companion', '')), _gn9))
                if _gn9 is not None else None)
            if _gi9 is None:
                stats['drop_reasons']['huntres_guide_unresolved'] += 1
                break
            # Also retain the last tile's Guide boundary: choosing the Guide
            # must precede its deferred reveal and placement. The per-game
            # Guide emitter owns the training row and its literal nomination.
            from rendered_hunt_observation import mark as mark_hunt_observation
            mark_hunt_observation(st, int(_gg9[0]['line']), 'after_draw')
            batch.setdefault('_rx_hunt_guide_predecision', {})[int(_gg9[0]['line'])] = ai.clone_state(st)
            try:
                st, _ = _apply9(_gl9[_gi9])
            except Exception:
                stats['drop_reasons']['huntres_apply_exc'] += 1
                break
            stats['detected']['huntres_guide_applied'] += 1
            continue
        if ph is None:
            # W3-011: the human's reveal PLACEMENT continues the chain
            # when the log shows a further tile (p.42 entering/leaving a
            # Shadow stronghold on the placement): apply it silently -
            # the placement row itself is _emit_reveal_placements'.
            if not (getattr(st, '_reveal_place_due', False)
                    and not getattr(st, '_pending_hunt_damage', 0)
                    and k + 1 < n_live):
                break
            dest9 = _fel_place_dest(batch, _wins9[k][0])
            if dest9 is None:
                stats['drop_reasons']['huntres_place_nodest'] += 1
                break
            try:
                _pl9 = [a for a in ai.get_legal_actions(st, Side.FREE_PEOPLES)
                        if a.action_type == 'fellowship_reveal_place']
            except Exception:
                stats['drop_reasons']['huntres_legal_exc'] += 1
                break
            _pi9 = _match_index(
                _pl9, lambda a: rr.sanitize(str(a.params.get('region', '')))
                == dest9)
            if _pi9 is None:
                stats['drop_reasons']['huntres_place_unmatched'] += 1
                break
            try:
                placement_action = _pl9[_pi9]
                from fellowship_reveal_route import balrog_card
                if balrog_card(st) is not None:
                    from rendered_balrog import observed_path
                    target, path = observed_path(st, batch, int(live_tiles[k]['line'])+1,
                                                  int(live_tiles[k+1]['line']))
                    if target != dest9:
                        raise ValueError('balrog_reveal_target_mismatch')
                    placement_action = _cp9.deepcopy(placement_action)
                    placement_action.params['path'] = list(path)
                st, _cons9 = _apply9(placement_action)
            except Exception:
                stats['drop_reasons']['huntres_apply_exc'] += 1
                break
            stats['detected']['huntres_place_applied'] += 1
            if _cons9:
                k += 1
                stats['detected']['huntres_chain_tile'] += 1
                (sep_queue, elim_queue, guide_ev, random_ev, _corr_walk9,
                 _fel_flips9, _client_null9) = _win_state9(k)
                _gollum_revealed9 = revealed or (_fel_flips9 % 2 == 1)
                _sacked9 = False
            continue
        phase = ph['phase']
        try:
            legal = ai.get_legal_actions(st, Side.FREE_PEOPLES)
        except Exception:
            stats['drop_reasons']['huntres_legal_exc'] += 1
            break
        _hfam9 = (str(phase) if str(phase).startswith('hunt')
                  else 'hunt_' + str(phase))
        if len(legal) >= 2:
            _pend_note(stats, _hfam9, 'open')
        pick = None
        label = phase
        victim9 = None
        g = st.fellowship.guide
        if phase == 'hunt_gollum':
            # W3-018: the label must agree with BOTH gestures (flip parity
            # after the tile draw and the Corruption walk after it); a
            # walk truncated at 12 Corruption (terminal, rb p.44) is
            # accepted when it stops short of the expected damage
            _dmg_g9 = int(ph.get('damage', 0) or 0)
            _exp_g9 = _dmg_g9 - 1 if _gollum_revealed9 else _dmg_g9
            _ok_g9 = (_corr_walk9 == _exp_g9) or (
                0 <= _corr_walk9 < _exp_g9 and _tb_c9 + _corr_walk9 >= 12)
            if not _ok_g9:
                stats['drop_reasons']['huntres_gollum_ambiguous'] += 1
                break
            want = ('hunt_gollum_reveal' if _gollum_revealed9
                    else 'hunt_gollum_no_reveal')
            pick = _match_index(legal, lambda a: a.action_type == want)
            label = want
        elif phase == 'hunt_sacrifice':
            _hob9 = (g is not None and getattr(g, 'guide_ability', None)
                     == 'sacrifice_reduce_hunt')
            gsep = _hob9 and any(rr._named_eq(g.name, c) for c in sep_queue)
            # W3-011 (W3-003 mechanism, GL-5): on the Mordor track the
            # hobbit guide's separation ELIMINATES him - the human's
            # 'from Guide Of The Fellowship to Current FP Card' + kill is
            # that action (GenericPie L897 Pippin after Gimli)
            gsep_elim = (_hob9 and _in_mordor9 and not gsep
                         and any(rr._named_eq(g.name, c)
                                 for c in elim_queue))
            if gsep or gsep_elim:
                pick = _match_index(legal, lambda a: a.action_type ==
                                    'hunt_guide_separate')
                if pick is not None:
                    sep_queue = [c for c in sep_queue
                                 if not rr._named_eq(g.name, c)]
                    elim_queue = [c for c in elim_queue
                                  if not rr._named_eq(g.name, c)]
                label = 'hunt_guide_separate'
            elif (elim_queue or sep_queue) \
                    and not ph.get('casualty_taken'):
                # a SEPARATED victim (moved to a region, GAME19 T7
                # Merry -> Fangorn) is still the sacrifice ARM choice -
                # emitting corruption instead mislabeled (client corr
                # unchanged); the placement rides hobbit_place
                _from_sep = not elim_queue
                victim = (elim_queue or sep_queue)[0]
                if victim in guide_ev:
                    _is_guide = True        # gesture evidence (primary)
                elif victim in random_ev:
                    _is_guide = False
                else:
                    _is_guide = (g is not None
                                 and rr._named_eq(g.name, victim))
                if _is_guide:
                    pick = _match_index(legal, lambda a: a.action_type ==
                                        'hunt_sacrifice_guide')
                    label = 'hunt_sacrifice_guide'
                else:
                    pick = _match_index(legal, lambda a: a.action_type ==
                                        'hunt_sacrifice_random')
                    label = 'hunt_sacrifice_random'
                    victim9 = victim
                if pick is not None:
                    if _from_sep:
                        sep_queue = sep_queue[1:]
                    else:
                        elim_queue = elim_queue[1:]
                    _sacked9 = True
            elif elim_queue:
                # a residual elimination the sacrifice rules can't place (only
                # ONE non-separate casualty per hunt) - evidence disagrees with
                # the engine chain; drop rather than mislabel corruption
                stats['drop_reasons']['huntres_elim_residual'] += 1
                break
            else:
                pick = _match_index(legal, lambda a: a.action_type ==
                                    'hunt_take_corruption')
                label = 'hunt_take_corruption'
        elif phase == 'hobbit_place':
            comp = rr.COMP_ALIAS.get(str(ph.get('companion', '')).lower(),
                                     str(ph.get('companion', '')))
            dest = None
            for (nm, reg) in seps_all.items():
                if rr._named_eq(nm, comp):
                    dest = reg
                    break
            if dest is not None:
                pick = _match_index(legal, lambda a: a.action_type ==
                                    'companion_separate' and
                                    rr.sanitize(str(a.params.get(
                                        'to_region', ''))) == dest)
            label = 'hobbit_place'
        if pick is None:
            stats['drop_reasons']['huntres_%s_unmatched' % phase] += 1
            break
        if label == 'hunt_take_corruption' and _client_null9:
            stats['detected']['huntres_client_null'] += 1
            if len(legal) >= 2:
                _pend_note(stats, _hfam9, 'auto')
        elif label == 'hunt_take_corruption' and _sacked9:
            # USER RULING (GAME01 chunt amendment): after a sacrifice,
            # the damage REMAINDER becomes corruption AUTOMATICALLY -
            # apply, never emit (GAME02 T4 Merry +2 remainder)
            stats['detected']['huntres_remainder_auto'] += 1
            if len(legal) >= 2:
                _pend_note(stats, _hfam9, 'auto')
        elif len(legal) >= 2:              # a real choice -> emit
            stats['kinds_emit'][label] += 1
            stats['kinds_emit']['hunt_resolution'] += 1
            _pend_note(stats, _hfam9, 'emit')
            from rendered_hunt_observation import mark as mark_hunt_observation
            mark_hunt_observation(st, int(live_tiles[k]['line']) + 1, 'after_draw')
            out.append(_emit_sample(st, Side.FREE_PEOPLES, legal, pick,
                                    phase, batch['turn']))
        try:
            st, _cons9 = _apply9(legal[pick], victim9)
        except Exception:
            stats['drop_reasons']['huntres_apply_exc'] += 1
            break
        if _cons9:
            # the engine drew the human's next tile inside the apply
            # (revealed-in prep draw, 1.132): its decision is next
            k += 1
            stats['detected']['huntres_chain_tile'] += 1
            (sep_queue, elim_queue, guide_ev, random_ev, _corr_walk9,
             _fel_flips9, _client_null9) = _win_state9(k)
            _gollum_revealed9 = revealed or (_fel_flips9 % 2 == 1)
            _sacked9 = False
    batch['_rx_huntres'] = {'tiles_done': k + 1, 'n_live': n_live,
                            'lines': [w[0] for w in _wins9]}
    return out


# ==========================================================================
# HUNT-INTERPOSE TABLE CARDS  (CLOSURE wave, spec 1.216 items 2/3/4 + 1.217)
# --------------------------------------------------------------------------
# With the table state installed (persistent_cards), the engine's OWN
# hunt-reduction interposes reconstruct and the use/decline menu legality-
# matches.  All three are FP binary decisions evidenced by the persistent card
# being KILLED (used) in the hunt-context batch vs staying on the table:
#   * Axe and Bow / Horn of Gondor: discard to reduce Hunt damage by 1
#     (game_logic._offer_hunt_reduction â€” opened only when pending damage > 0).
#   * Wizard's Staff: discard to PREVENT the Shadow tile draw (pre-draw
#     nullify; game_logic._resolve_hunt_tile wizard branch).  Amendment 1.217:
#     the engine now interposes on normal hunts + Isildur's Bane + Foul Thing â€”
#     the same pre-draw interpose fires in every context, so the mirror's
#     kill-in-hunt-window signal covers all three uniformly.
#   * Mithril Coat and Sting: discard to REDRAW the drawn tile
#     (game_logic._resolve_hunt_tile mithril branch, reached with
#     _skip_wizard=True so a co-present staff never masks it).  Crack-of-Doom
#     edge: the card is MOVED to an active-card region instead of killed
#     (position noise, NOT a redraw) â€” the ledger records that as off_move,
#     never off_kill, so it labels KEEP.
# Each probe reconstructs its own state, installs the table cards, opens the
# REAL engine interpose and DROPS on any mismatch (purity, never a guess).
# The card the engine offers is byte-identical to self-play's; the human's
# choice is matched by the KILL evidence.
# ==========================================================================
def _emit_axebow_horn(batch, stats, lo, hi):
    """Emit the actual post-draw Axe/Horn menu after a funded movement."""
    from rendered_reactive_cards import emit_movement_hunt_reduction
    stats['detected']['huntcard_axebow_horn'] += 1
    return emit_movement_hunt_reduction(batch, stats)


def _emit_wizardstaff(batch, stats, lo, hi, is_mordor, ws_killed, tile_drawn):
    """Wizard's Staff prevent (killed + no tile drawn) vs decline (tile drawn +
    staff stays).  A KILL that still drew a tile is contradictory -> drop."""
    import ai
    import game_logic as gl
    from game_state import Side
    stats['detected']['huntcard_wizardstaff'] += 1
    if ws_killed and not tile_drawn:
        used = True                        # prevented the draw
    elif tile_drawn and not ws_killed:
        used = False                       # declined; the tile was drawn
    else:
        stats['drop_reasons']['wizard_ambiguous'] += 1
        return None
    try:
        st = _recon_state(batch['truth_before'], batch.get('turn'))
    except Exception:
        stats['drop_reasons']['wizard_recon'] += 1
        return None
    _prepare_fellowship_dice(st, batch, pending=True, advance=True)
    _install_persistent_cards(st, lo)
    _install_ledger_hands(st, batch)
    st.pending_card_option = None
    try:
        gl._resolve_hunt_tile(st, 1, mordor=is_mordor)
    except Exception:
        stats['drop_reasons']['wizard_interpose_exc'] += 1
        return None
    pco = getattr(st, 'pending_card_option', None)
    if not pco or not any('wizardstaff_prevent' in (o or {})
                          for o in pco.get('options', [])):
        stats['drop_reasons']['wizard_no_interpose'] += 1
        return None
    try:
        legal = ai.get_legal_actions(st, Side.FREE_PEOPLES)
    except Exception:
        stats['drop_reasons']['wizard_legal_exc'] += 1
        return None
    hidx = _match_index(
        legal, lambda a: a.action_type == 'card_option' and
        bool((a.params.get('choices') or {}).get('wizardstaff_prevent'))
        == bool(used))
    if hidx is None:
        stats['drop_reasons']['wizard_unmatched'] += 1
        return None
    stats['kinds_emit']['wizardstaff'] += 1
    stats['kinds_emit']['wizardstaff_' + ('prevent' if used else 'decline')] \
        += 1
    from rendered_hunt_observation import mark as mark_hunt_observation
    drawn = _hunt_tile_items(batch)
    if len(drawn) == 1:
        mark_hunt_observation(st, int(drawn[0]['line']), 'before_draw')
    return _emit_sample(st, Side.FREE_PEOPLES, legal, hidx, 'card_option',
                        batch['turn'])


def _emit_mithril(batch, stats, lo, hi, is_mordor):
    """Mithril Coat and Sting redraw (kill) vs keep."""
    import ai
    import game_state as gs
    import game_logic as gl
    import rendered_census as rc
    from game_state import Side
    from rendered_card_timing import is_fp_challenge_draw
    if is_fp_challenge_draw(batch):
        stats['detected']['mithril_not_applicable_challenge'] += 1
        return None
    tile_drawn = any(it.get('kind') == 'draw_tile' for it in batch['rng'])
    used = _TABLE.killed_in('MITHRIL_COAT', lo, hi)
    if not (tile_drawn or used):
        return None
    stats['detected']['huntcard_mithril'] += 1
    try:
        st = _recon_state(batch['truth_before'], batch.get('turn'))
    except Exception:
        stats['drop_reasons']['mithril_recon'] += 1
        return None
    _prepare_fellowship_dice(st, batch, pending=True, advance=True)
    _install_persistent_cards(st, lo)
    _install_ledger_hands(st, batch)
    st.pending_card_option = None
    tile = next((r for r in batch['rng'] if r['kind'] == 'draw_tile'), None)
    tname = str(tile['value']).strip().lower() if tile else None
    fo = rc._tile_obj(gs, tname) if tname else None
    if fo is None:
        # A discarded card is not evidence of the tile's damage/reveal
        # properties. Never encode a manufactured one-damage replacement.
        stats['drop_reasons']['mithril_drawn_tile_unproven'] += 1
        return None
    st.hunt_pool = [fo]
    try:
        gl._resolve_hunt_tile(st, 1, mordor=is_mordor, _skip_wizard=True)
    except Exception:
        stats['drop_reasons']['mithril_interpose_exc'] += 1
        return None
    pco = getattr(st, 'pending_card_option', None)
    if not pco or not any('mithril_redraw' in (o or {})
                          for o in pco.get('options', [])):
        stats['drop_reasons']['mithril_no_interpose'] += 1
        return None
    try:
        legal = ai.get_legal_actions(st, Side.FREE_PEOPLES)
    except Exception:
        stats['drop_reasons']['mithril_legal_exc'] += 1
        return None
    hidx = _match_index(
        legal, lambda a: a.action_type == 'card_option' and
        bool((a.params.get('choices') or {}).get('mithril_redraw'))
        == bool(used))
    if hidx is None:
        stats['drop_reasons']['mithril_unmatched'] += 1
        return None
    stats['kinds_emit']['mithril'] += 1
    stats['kinds_emit']['mithril_' + ('redraw' if used else 'keep')] += 1
    if tile is not None:
        from rendered_hunt_observation import mark as mark_hunt_observation
        mark_hunt_observation(st, int(tile['line']) + 1, 'after_draw')
    return _emit_sample(st, Side.FREE_PEOPLES, legal, hidx, 'card_option',
                        batch['turn'])


def _emit_hunt_cards(batch, stats):
    """Dispatch the three hunt-interpose table-card decisions for an FP batch.
    Additive + independent (own reconstruction, drop on any failure).

    RNG-NEUTRAL: _emit_axebow_horn re-drives the hunt (rendered_census.
    _drive_hunt), which on a stream-exhaustion fallback advances the global
    generator.  We snapshot/restore random state around the whole dispatch so
    this emitter consumes zero net randomness â€” the downstream _emit_hunt_
    resolution / _drive_* of LATER batches then see the exact baseline RNG
    state, keeping every prior emit-kind count byte-identical (strict A/B)."""
    import random
    if batch.get('_rx_hunt_exclusion') or _TABLE is None:
        return []
    if batch.get('die_side') == 'SHADOW':
        from rendered_reactive_cards import emit_shadow_hunt_reaction
        return emit_shadow_hunt_reaction(batch, stats)
    if batch.get('die_side') != 'FREE_PEOPLES':
        return []
    lo = batch.get('start_line') or 0
    hi = int(batch['end_line']) + 1 if batch.get('end_line') else 10 ** 9
    is_mordor = bool(batch['truth_before']['fel'].get('mordor'))
    tile_drawn = any(it.get('kind') == 'draw_tile' for it in batch['rng'])
    hunt_ctx = batch['kind'] in ('hunt', 'combat?') or tile_drawn
    out = []
    _rng_state = random.getstate()
    try:
        # Axe & Bow / Horn: any hunt-context batch with reducible damage
        if (hunt_ctx and not batch.get('_rx_hunt_reduction_owned')
                and (_TABLE.on_at('AXE_AND_BOW', lo) or
                          _TABLE.on_at('HORN_OF_GONDOR', lo))):
            s = _emit_axebow_horn(batch, stats, lo, hi)
            if s is not None:
                out.append(s)
        # Wizard's Staff: prevent (killed, no tile) or decline (tile drawn)
        ws_killed = _TABLE.killed_in('WIZARD_STAFF', lo, hi)
        if _TABLE.on_at('WIZARD_STAFF', lo) and (tile_drawn or ws_killed):
            s = _emit_wizardstaff(batch, stats, lo, hi, is_mordor,
                                  ws_killed, tile_drawn)
            if s is not None:
                out.append(s)
        # Mithril Coat and Sting: redraw (killed) or keep
        if _TABLE.on_at('MITHRIL_COAT', lo):
            s = _emit_mithril(batch, stats, lo, hi, is_mordor)
            if s is not None:
                out.append(s)
    finally:
        random.setstate(_rng_state)
    return out


# ==========================================================================
# FELLOWSHIP DECLARATIONS  (CLOSURE wave, spec 1.216 item 6)
# --------------------------------------------------------------------------
# COUNTING RULE (v32 gate-(e), USER 2026-08-03): the progress-1+
# declaration WITH PLACEMENT is a decision, AND a progress-0 in-place
# declare that HEALS in a free stronghold/city IS emitted as a DECLARED
# sample (_emit_one_heal_declare below); progress-0 effectful windows
# emit pass-labeled declines (_emit_one_decline).  spec: "every reveal
# must have a source else it is the progress-reset trick (not a
# reveal, and the reset = declaration evidence)."
#
# The engine surfaces declaration via state._declare_phase -> _get_declare_
# actions(allow_in_place=True): a hidden, non-Mordor fellowship declares to any
# region <= progress steps away.  We reconstruct the pre-declaration mirror,
# force hidden=True (declaration's precondition; accumulating progress requires
# hidden movement), open the declare phase and legality-match the placement.
# ==========================================================================
def _emit_one_declare(b, placement, kind, stats):
    import ai
    import rendered_replay as rr
    from game_state import Side
    if not placement:
        stats['drop_reasons']['declare_no_placement'] += 1
        return None
    try:
        st = _recon_state(b['truth_before'], b.get('turn'))
    except Exception:
        stats['drop_reasons']['declare_recon'] += 1
        return None
    st.fellowship.hidden = True            # declaration precondition
    st.fellowship.mordor_entered = False
    st._declare_phase = True
    _install_ledger_hands(st, b)
    _ensure_decks(st)   # T3: live declare states carry populated decks;
    # empty-deck staging made X[3222-3225]==0 a corpus-only declare
    # marker the net keyed P(pass) on (absent live -> declare collapse)
    _install_persistent_cards(st, b.get('start_line') or 0)
    try:
        legal = ai.get_legal_actions(st, Side.FREE_PEOPLES)
    except Exception:
        stats['drop_reasons']['declare_legal_exc'] += 1
        return None
    hidx = _match_index(
        legal, lambda a: a.action_type == 'fellowship_declare' and
        rr.sanitize(str(a.params.get('region', ''))) == placement)
    if hidx is None:
        stats['drop_reasons']['declare_%s_unmatched' % kind] += 1
        return None
    stats['kinds_emit']['fellowship_declare'] += 1
    stats['kinds_emit']['declare_' + kind] += 1
    from rendered_balrog import emit_declaration
    followup = emit_declaration(st, legal[hidx], b, stats)
    if followup is not None:
        stats.setdefault('__declaration_followups', []).append(followup)
    return _emit_sample(st, Side.FREE_PEOPLES, legal, hidx,
                        'fellowship_declare', b['turn'])


def _emit_one_decline(b, stats):
    """DECLINE twin of _emit_one_declare (v29 fix wave, 2026-07-26;
    v32 gate-(e) rework 2026-08-03).

    A turn start where the Fellowship COULD declare (hidden, not in
    Mordor) and the human did NOT is itself the WHETHER decision.
    Without these samples the corpus holds only positive declares, so BC
    learns 'always declare' (v28 self-play: 0.60 declares/TURN vs the
    human rate of ~1.9/GAME, incl. 23 Moria declares in 42 games).  Same
    absence-is-the-choice class as the session-34 zero-eye fix.

    v32 gate-(e) (USER per-turn declare coverage): the old progress>=1
    gate is GONE â€” progress-0 windows with an EFFECTFUL in-place
    declare in the menu (gateway arming / heal / activation, the same
    menu live play sees under allow_in_place=True) now emit the
    pass-labeled twin too; a progress-0 turn start whose declare menu
    is empty returns ('no_declares') and is recorded as the
    declined-no-window coverage category by the caller.  The
    mordor/revealed eligibility gates moved to the CALLER
    (_emit_declarations), which owns per-turn categorization.

    Returns (sample_or_None, code)."""
    import ai
    from game_state import Side
    try:
        st = _recon_state(b['truth_before'], b.get('turn'))
    except Exception:
        stats['drop_reasons']['decline_recon'] += 1
        return None, 'recon'
    st.fellowship.hidden = True
    st.fellowship.mordor_entered = False
    st._declare_phase = True
    _install_ledger_hands(st, b)
    _ensure_decks(st)                      # T3 staging parity with live
    _install_persistent_cards(st, b.get('start_line') or 0)
    try:
        legal = ai.get_legal_actions(st, Side.FREE_PEOPLES)
    except Exception:
        stats['drop_reasons']['decline_legal_exc'] += 1
        return None, 'legal_exc'
    if not any(a.action_type == 'fellowship_declare' for a in legal):
        stats['drop_reasons']['decline_no_declares'] += 1
        return None, 'no_declares'
    hidx = _match_index(legal, lambda a: a.action_type == 'pass')
    if hidx is None:
        stats['drop_reasons']['decline_no_pass'] += 1
        return None, 'no_pass'
    stats['kinds_emit']['declare_decline'] += 1
    if int(getattr(st.fellowship, 'progress', 0) or 0) == 0:
        stats['kinds_emit']['declare_decline_p0'] += 1
    return _emit_sample(st, Side.FREE_PEOPLES, legal, hidx,
                        'fellowship_declare', b['turn']), 'ok'


def _emit_one_heal_declare(b, stats):
    """Progress-0 in-place HEAL declare (v32 gate-(e), USER counting
    rule 2026-08-03): a pre-stage lone corruption down-walk with
    progress_before == 0 IS the human re-declaring in place for the
    heal (rulebook p.34; same dialect log_replay records as
    in_place_heal).  Counts as category DECLARED and emits the
    in-place fellowship_declare sample.  A heal gesture with
    progress > 0 does NOT reach here (caller enforces)."""
    import ai
    import rendered_replay as rr
    from game_state import Side
    try:
        st = _recon_state(b['truth_before'], b.get('turn'))
    except Exception:
        stats['drop_reasons']['healdecl_recon'] += 1
        return None
    st.fellowship.hidden = True
    st.fellowship.mordor_entered = False
    st.fellowship.progress = 0
    st._declare_phase = True
    _install_ledger_hands(st, b)
    _ensure_decks(st)                      # T3 staging parity with live
    _install_persistent_cards(st, b.get('start_line') or 0)
    try:
        legal = ai.get_legal_actions(st, Side.FREE_PEOPLES)
    except Exception:
        stats['drop_reasons']['healdecl_legal_exc'] += 1
        return None
    fel = (b.get('truth_before') or {}).get('fel') or {}
    cur = rr.sanitize(str(fel.get('region') or ''))
    hidx = _match_index(
        legal, lambda a: a.action_type == 'fellowship_declare' and
        rr.sanitize(str(a.params.get('region', ''))) == cur)
    if hidx is None:
        stats['drop_reasons']['healdecl_unmatched'] += 1
        return None
    stats['kinds_emit']['fellowship_declare'] += 1
    stats['kinds_emit']['declare_heal0'] += 1
    return _emit_sample(st, Side.FREE_PEOPLES, legal, hidx,
                        'fellowship_declare', b['turn'])


_CARD_MOVEMENT_NAMES = frozenset({
    'nazgul search', 'the nazgul strike!', 'the ringwraiths are abroad',
    'the shadow is moving', 'the shadow lengthens', 'shadows gather',
    'corsairs of umbar', 'the black captain commands', 'cruel weather',
})


def _emit_cardhunt_worn(state, batch, action, victim, exits, tiles, stats):
    """Replay one evidenced Hunt casualty to reach its real Worn response.

    This is separate from the historical arm-only replay below. In particular
    manually removing a companion cannot manufacture a Worn pending, and a
    card which merely kills a figure is not reinterpreted as a Hunt casualty.
    """
    import ai
    import copy
    import game_logic as gl
    import rendered_replay as rr
    from rendered_reactive_cards import (Unproven, prepare_worn_trigger,
        worn_casualty_window, drain_worn)
    from unittest.mock import patch
    if (_REACTIVE_TABLE is None or _LEDGER is None or not _LEDGER.enabled
            or victim not in exits
            or action.action_type not in ('hunt_sacrifice_guide', 'hunt_sacrifice_random')):
        return []
    lower = int(exits[victim])
    if 'foul thing from the deep' in str(_card_play_title(batch) or '').lower().replace('-', ' '):
        stats['drop_reasons']['worn_foul_requires_event_replay'] += 1
        return []
    if not any(card.card_type.name == 'WORN_WITH_SORROW'
               for card in _REACTIVE_TABLE.cards_at(lower).values()):
        return []
    try:
        window = worn_casualty_window(batch, lower, exits, tiles, _REACTIVE_TABLE)
        replay = ai.clone_state(state)
        prepare_worn_trigger(replay, batch, *window[:2], _REACTIVE_TABLE)
        if action.action_type == 'hunt_sacrifice_guide':
            if not rr._named_eq(getattr(replay.fellowship.guide, 'name', ''), victim):
                raise Unproven('worn_cardhunt_guide_unproven')
        elif not any(rr._named_eq(c.name, victim) for c in replay.fellowship.companions):
            raise Unproven('worn_cardhunt_random_victim_unproven')
        observed_action = copy.copy(action)
        observed_action.params = dict(action.params)
        if action.action_type == 'hunt_sacrifice_random':
            observed_action.params['forced_companion'] = victim
        def unexpected(*args, **kwargs):
            raise Unproven('worn_cardhunt_unrecorded_randomness')
        # forced_companion consumes the ordinary random-choice slot for stream
        # compatibility. Its recorded victim supplies the result, not this draw.
        with patch('random.choice', lambda choices: choices[0]), \
                patch.object(gl, 'pick_random_companion', unexpected), \
                patch.object(gl, 'pick_hunt_tile', unexpected), \
                patch.object(gl, 'pick_random_card', unexpected):
            replay = ai.apply_action(replay, observed_action)
        if ((getattr(replay, 'pending_card_option', None) or {}).get('card_name') != '__wwsat_discard__'
                and getattr(replay, '_wwsat_owed', None)):
            raise Unproven('worn_cardhunt_intervening_choice_unproven')
        _, rows = drain_worn(replay, batch, stats, *window[:2], complete_window=window[2])
        stats['detected']['worn_cardhunt_engine_casualty'] += bool(rows)
        return rows
    except Unproven as error:
        stats['drop_reasons'][str(error)] += 1
        return []


def _emit_cardhunt_arms(batches, stats):
    """USER truth-table fix #7b (game01_verification.md): hunt-damage ARM
    decisions in contexts the resolver skips (Shadow tile-card hunts like
    Nazgul Strike/Search, no-pawn-move card hunts, and the tiles of an
    FP move the walk could not follow).  Evidence per drawn TILE
    (GESTURE-primary, same rules as the resolver path, W3-011 windows):
    corruption-marker walk = take_corruption; a companion exiting 'Guide
    Of The Fellowship' = sacrifice_guide; a companion named on a 'draws
    <name>' line = sacrifice_random; chains emit one sample per link
    (GAME01 T7: Strider->Shelob guide-sac; Gimli+Pippin Eye-tile chain;
    T2/T6 Strike/Search randoms).

    W3-011 ONE OWNER PER TILE: an FP-die hunt the walk drove carries
    batch['_rx_huntres'] = {'tiles_done': n, ...}; its first n live
    tiles are the walk's (rows or drops) and only the tiles after them
    are ours (Columbo L185: Eye walked, the Moria pass-through '1' on the
    placement is ours).  Before W3-011 the Mordor Eye/Shelob and the
    Eye-then-stronghold batches were emitted by BOTH paths (GenericPie
    L782/L897, FateevVV L613/L632, CircleDakkon L802, Genoah77 L570,
    LuconiusRex L201, Vayze L744 ...)."""
    import ai
    import rendered_census as rc
    import rendered_replay as rr
    from game_state import Side
    out = []
    for b in batches:
        if b.get('_rx_hunt_exclusion'):
            continue
        from rendered_reactive_cards import emit_foul_worn
        out.extend(emit_foul_worn(b, stats, _REACTIVE_TABLE))
        if b.get('_rx_nazgul_hunt_owned'):
            # The recorded Event driver owns these post-flight states and
            # Shadow funding. Do not also synthesize a pre-Event FP Hunt.
            stats['detected']['chunt_nazgul_event_owned'] += 1
            continue
        # only batches with a hunt-damage source: drawn tiles among rng
        # (a tile Mithril set aside is not live - _hunt_live_tiles)
        tiles = _hunt_live_tiles(b)
        if not tiles:
            continue
        _is_fp9 = (b['kind'] == 'hunt'
                   and b.get('die_side') == 'FREE_PEOPLES')
        _mordor9 = bool(b['truth_before']['fel'].get('mordor')
                        or b['truth']['fel'].get('mordor'))
        hr = b.get('_rx_huntres')

        _tdmg9 = _hunt_tile_damaging9
        _eyehits9 = None
        if hr is not None:
            own_from = int(hr.get('tiles_done', 0) or 0)
            if own_from >= len(tiles):
                stats['detected']['chunt_walk_owned'] += 1
                continue
            if own_from == 0 and hr.get('zero_dmg'):
                # W3-011: the walk disowned a damaging first tile the
                # engine's drive priced at 0.  The log's damage is read
                # here only when the log ITSELF shows the hit - a 6 on
                # any hunt roll line before the draw (Mol L141 re-roll
                # '6', bowzo L88 '6', FateevVV L170 '6'; W3-002: the
                # drive feeds only the first roll line) or no roll at
                # all (a Mordor / card-driven automatic draw: FateevVV
                # L469).  The hunt modifier (rulebook p.40, standing
                # ruling 2026-08-16: +1 per FP die ALREADY in the Hunt
                # box this turn, the just-moved die excluded) is read
                # from _FP_HUNTBOX_IDX exactly as _seed_fp_huntbox
                # does: LuckyRuds L343-L345 '2 3 1' then '5' hits
                # because L296 moved an FP die into the box earlier the
                # same turn (5 + 1 >= 6; Shiggy L459-L460 likewise after
                # L429).  W3-Q4's premise ('no visible modifier') was
                # wrong; a roll set with no reachable hit is still
                # DROPPED (chunt_hit_unexplained) rather than forced.
                _fl9 = int(tiles[0].get('line') or 0)
                _rolls9 = [int(v) for it in b['rng']
                           if it.get('kind') == 'roll_num'
                           and int(it.get('line') or 0) < _fl9
                           for v in (it.get('ints') or [])
                           if str(v).lstrip('-').isdigit()]
                _mod9 = sum(1 for (_ln9, _tn9) in _FP_HUNTBOX_IDX
                            if _tn9 == b.get('turn')
                            and _ln9 < (b.get('start_line') or 0))
                if _rolls9 and not any(r > 1 and r + _mod9 >= 6
                                       for r in _rolls9):
                    stats['drop_reasons']['chunt_hit_unexplained'] += 1
                    continue
                # an Eye tile's damage = the number of successes (p.41);
                # the log's rolls price it exactly (Shiggy L459-L460:
                # one success = '1r', Legolas L2 absorbs it whole - the
                # victim's Level is NOT the damage)
                if _rolls9:
                    _eyehits9 = sum(1 for r in _rolls9
                                    if r > 1 and r + _mod9 >= 6)
            stats['detected']['chunt_walk_tail'] += 1
        elif _is_fp9 and _mordor9 and all(
                str(t.get('value', '')).strip('r!').isdigit()
                for t in tiles):
            # FP-die MORDOR hunts with a PLAIN NUMERIC tile drive through
            # the huntres walk (emitting here too DOUBLED the arm, GAME02
            # T4 step 1); an undriven one (walk skip) stays dropped there
            stats['detected']['chunt_own_walk'] += 1
            continue
        elif _is_fp9 and not _mordor9:
            # an FP move the walk did NOT drive (skip / divergent): its
            # first tile is the walk's (dropped there); we own the p.42
            # re-draw tiles only when the first is a special (Eye) or
            # damage-less (0r) tile - a plain damaging-first multi-draw
            # resolves through the walk (GAME02 T3 1r+1r verified)
            _dt9 = [t for t in tiles if _tdmg9(t)]
            _first_dmg9 = bool(tiles) and _tdmg9(tiles[0])
            _has_eye9 = any('eye' in str(t.get('value', '')).lower()
                            for t in _dt9)
            if len(tiles) <= 1 or (_first_dmg9 and not _has_eye9):
                continue
            own_from = 1
        else:
            own_from = 0
        # Foul Thing's first tile forces a random casualty; there is no
        # optional take-corruption/guide/random arm for a human to choose.
        # Further revealed-in-stronghold tiles remain ordinary Hunt choices.
        if 'foul thing from the deep' in str(_card_play_title(b) or '').lower().replace('-', ' '):
            if own_from < 1:
                stats['detected']['chunt_foul_forced_victim_not_decision'] += 1
                own_from = 1
        tb = b['truth_before']['fel']
        ta = b['truth']['fel']
        before = [rr.COMP_ALIAS.get(c.lower(), c)
                  for c in tb.get('comps', [])]
        after = [rr.COMP_ALIAS.get(c.lower(), c)
                 for c in ta.get('comps', [])]
        left_all = [c for c in before if c not in after]
        corr_gain = int(ta.get('corruption', 0) or 0) - \
            int(tb.get('corruption', 0) or 0)
        if not left_all and corr_gain <= 0:
            continue
        exits = _comp_exit_lines(b)
        seps_all = {}
        for (nm9, reg9) in _sep_moves(b):
            seps_all[rr.COMP_ALIAS.get(nm9.lower(), nm9)] = \
                rr.sanitize(str(reg9))
        wins = _hunt_windows(tiles)
        # a single-tile batch nobody drove keeps the BATCH-WIDE evidence
        # (the mirror's snapshot delta) it always had
        _whole9 = (len(tiles) == 1 and hr is None)
        if _whole9:
            wins = [(0, 10 ** 9)]
        try:
            st = _recon_state(b['truth_before'], b.get('turn'))
        except Exception:
            stats['drop_reasons']['chunt_recon'] += 1
            continue
        _prepare_fellowship_dice(st, b, pending=True, advance=True)
        _install_ledger_hands(st, b)

        def _drop_comp9(nm):
            for c in list(st.fellowship.companions):
                if rr._named_eq(c.name, nm):
                    st.fellowship.companions.remove(c)
                    break
            if st.fellowship.guide is not None and rr._named_eq(
                    st.fellowship.guide.name, nm):
                st.fellowship.guide = (
                    max(st.fellowship.companions,
                        key=lambda x: x.level)
                    if st.fellowship.companions else None)
        # W3-011: advance the working state through the walk's tiles so
        # our first tile sees the human's chain (victims gone, corruption
        # taken, pawn placed and revealed)
        for k in range(own_from):
            lo, hi = wins[k]
            ev = _hunt_window_ev(b, lo, hi, left_all, seps_all, exits)
            for nm in ev['left']:
                _drop_comp9(nm)
            st.fellowship.corruption = min(
                12, st.fellowship.corruption + max(0, ev['corr']))
            if ev['flips'] % 2 == 1:
                st.fellowship.hidden = False
            dest9 = _fel_place_dest(b, lo)
            if dest9 is not None:
                try:
                    st.fellowship.region = st.board.region(dest9)
                    st.fellowship.progress = 0
                    st.fellowship.hidden = False
                except Exception:
                    pass
        for k in range(own_from, len(tiles)):
            lo, hi = wins[k]
            ev = _hunt_window_ev(b, lo, hi, left_all, seps_all, exits)
            if _whole9:
                ev['left'] = list(left_all)
                ev['corr'] = corr_gain
            left = ev['left']
            guide_ev, random_ev = ev['guide_ev'], ev['random_ev']
            hobbit_place = {}
            for r in b['records']:
                if r.get('op') != 'move':
                    continue
                if not (lo <= int(r.get('line') or 0) < hi):
                    continue
                tk = rr.zone_kind(str(r.get('to', '')))
                if tk[0] not in ('region', 'stronghold') or not tk[1]:
                    continue
                # W3-011: the LAST region move settles the placement, a
                # group drag included (bowzo L102 Merry -> Hollin, L104
                # 'group (Merry, Isengard Regular)' -> DIMRILL DALE where
                # Merry stays: L216)
                for nm in _comp_names_in(r.get('obj', '')):
                    if nm in left:
                        hobbit_place[nm] = rr.sanitize(str(tk[1]))
            # one sample per victim link.  '__corr__' ONLY when the arm
            # was pure take-corruption (no victim): after a sacrifice the
            # damage REMAINDER becomes corruption automatically (rulebook)
            # - not a separate decision.  Each damage TILE is its own
            # resolution (GAME19 T1: Eye then 1r = two separate
            # corruption ARMS, not one big link).
            links = list(left) if left else (
                ['__corr__'] if ev['corr'] > 0 else [])
            if not links:
                continue
            # the pending damage the human decided on = the drawn tile's
            # value (an Eye / special: the damage absorbed)
            dmg9 = _hunt_tile_dmg9(tiles[k], ev, rc)
            if k == 0 and _eyehits9 and \
                    'eye' in str(tiles[0].get('value', '')).lower():
                dmg9 = _eyehits9
            for nm in links:
                st.pending_hunt = {'phase': 'hunt_sacrifice',
                                   'damage': dmg9}
                from rendered_hunt_observation import mark as mark_hunt_observation
                mark_hunt_observation(st, int(tiles[k]['line']) + 1, 'after_draw')
                try:
                    legal = ai.get_legal_actions(st, Side.FREE_PEOPLES)
                except Exception:
                    stats['drop_reasons']['chunt_legal_exc'] += 1
                    break
                legal = [a for a in legal
                         if a.action_type.startswith('hunt_')]
                g = st.fellowship.guide
                _hob_g9 = (g is not None and rr._named_eq(g.name, nm)
                           and getattr(g, 'guide_ability', None)
                           == 'sacrifice_reduce_hunt')
                if nm == '__corr__':
                    want = 'hunt_take_corruption'
                elif _hob_g9:
                    # GL-5: the hobbit guide SEPARATES (in Mordor: is
                    # eliminated) - never a sacrifice row
                    want = 'hunt_guide_separate'
                elif nm in guide_ev:
                    want = 'hunt_sacrifice_guide'
                elif nm in random_ev:
                    want = 'hunt_sacrifice_random'
                else:
                    want = ('hunt_sacrifice_guide' if g is not None
                            and rr._named_eq(g.name, nm)
                            else 'hunt_sacrifice_random')
                hidx = _match_index(legal,
                                    lambda a: a.action_type == want)
                if hidx is None or len(legal) < 2:
                    stats['drop_reasons']['chunt_%s_unmatched' % want[5:]] \
                        += 1
                else:
                    stats['detected']['cardhunt_arm'] += 1
                    stats['kinds_emit'][want] += 1
                    s = _emit_sample(st, Side.FREE_PEOPLES, legal, hidx,
                                     'hunt_sacrifice', b['turn'])
                    if s is not None:
                        out.append(s)
                    if nm in guide_ev or nm in random_ev:
                        out.extend(_emit_cardhunt_worn(st, b, legal[hidx], nm,
                            {name: exits[name] for name in left_all if name in exits}, tiles, stats))
                    # hobbit PLACED not eliminated (Take Them Alive): the
                    # placement region is its own decision (GAME01 T2:
                    # Merry -> BREE)
                    if nm in hobbit_place:
                        lvl = {'Meriadoc': 1, 'Peregrin': 1}.get(nm, 1)
                        st.pending_hunt = {'phase': 'hobbit_place',
                                           'companion': nm, 'level': lvl}
                        try:
                            la2 = [a for a in ai.get_legal_actions(
                                st, Side.FREE_PEOPLES)
                                if a.action_type == 'companion_separate']
                            h2 = _match_index(
                                la2, lambda a: rr.sanitize(str(
                                    a.params.get('region',
                                                 a.params.get('to_region',
                                                              '')))) ==
                                hobbit_place[nm])
                            if h2 is not None and len(la2) >= 2:
                                stats['kinds_emit']['hobbit_place'] += 1
                                s2 = _emit_sample(st, Side.FREE_PEOPLES,
                                                  la2, h2, 'hobbit_place',
                                                  b['turn'])
                                if s2 is not None:
                                    out.append(s2)
                            else:
                                stats['drop_reasons'][
                                    'chunt_hobbitplace_unmatched'] += 1
                        except Exception:
                            stats['drop_reasons'][
                                'chunt_hobbitplace_exc'] += 1
                # apply the link to the working state so the next link's
                # menu reflects it (chain semantics)
                if nm != '__corr__':
                    _drop_comp9(nm)
                st.pending_hunt = None
            # the tile's corruption and reveal carry into the next window
            if not _whole9:
                st.fellowship.corruption = min(
                    12, st.fellowship.corruption + max(0, ev['corr']))
                if ev['flips'] % 2 == 1:
                    st.fellowship.hidden = False
    return out


def _remember_reveal_predecision(batch, st, tile):
    """Retain a real engine settle point, never a synthesized pending flag.

    This state still carries forced replay RNG. The placement emitter must
    verify its public prefix and replace that artificial bag before encoding.
    """
    import ai
    import game_logic as gl
    from game_state import Side
    if (not getattr(st, '_reveal_place_due', False)
            or st.fellowship.hidden or st.fellowship.mordor_entered
            or st.fellowship.progress <= 0 or st.fellowship.corruption >= 12
            or st.pending_hunt is not None or st.pending_guide_change is not None
            or getattr(st, '_pending_hunt_damage', 0)):
        return
    cur = gl.fellowship_region(st)
    if ai._is_shadow_stronghold(st, cur) and cur not in (
            getattr(st, '_sh_tile_drawn_regions', None) or set()):
        return                       # revealed-in draw still precedes placement
    legal = ai.get_legal_actions(st, Side.FREE_PEOPLES)
    if len(legal) < 2 or any(a.action_type != 'fellowship_reveal_place'
                              for a in legal):
        return
    key = int(tile.get('line') or 0)
    stored = batch.setdefault('_rx_reveal_predecision', [])
    if not any(row['tile_line'] == key for row in stored):
        stored.append({'tile_line': key, 'state': ai.clone_state(st)})


def _reveal_public_pool(batches, before_line):
    """Read the public physical tile prefix; card-face names never move tiles."""
    from rendered_mordor_pool import public_pool
    return public_pool(batches, before_line, require_entry=False)


def _reveal_verified_state(batch, batches, progress, placement, stats):
    """Verify the actual damage/guide prefix before an observed pawn placement."""
    import ai
    import rendered_replay as rr
    from game_state import Side
    candidates = batch.get('_rx_reveal_predecision') or []
    if not candidates:
        return None, None, 'missing_driven_prestate'
    matched = []
    for candidate in candidates:
        st = candidate['state']
        if st.fellowship.progress != progress:
            continue
        tile_line = candidate['tile_line']
        # A pawn move gives a precise decision boundary. If the player stays,
        # use the reveal flip only when all preceding damage is already shown.
        movements = [r for r in batch['records']
                     if int(r.get('line') or 0) > tile_line
                     and r.get('op') == 'move'
                     and str(r.get('obj', '')).strip().lower() == 'the fellowship'
                     and rr.zone_kind(str(r.get('to', '')))[0] == 'region']
        if movements:
            anchor = movements[0]
            if any(anchor['line'] < int(t.get('line') or 0) < movements[-1]['line']
                   for t in _hunt_tile_items(batch)):
                continue             # changed destination across later chance
        else:
            anchor = next((r for r in batch['records']
                           if int(r.get('line') or 0) > tile_line
                           and r.get('op') == 'flip' and r.get('revealed')
                           and str(r.get('obj', '')).strip().lower()
                           in ('the fellowship', 'fsp')), None)
        if anchor is None or not isinstance(anchor.get('decision_truth'), dict):
            continue
        prefix = anchor['decision_truth'].get('fel', {})
        names = lambda values: sorted(rr.COMP_ALIAS.get(str(v).lower(), str(v))
                                      for v in values if str(v).lower() != 'gollum')
        if (int(prefix.get('corruption', -1)) != st.fellowship.corruption
                or names(prefix.get('comps', [])) != names(
                    c.name for c in st.fellowship.companions)
                or not rr._named_eq(str(prefix.get('guide') or ''),
                                    str(getattr(st.fellowship.guide, 'name', '')))):
            continue
        legal = ai.get_legal_actions(st, Side.FREE_PEOPLES)
        if not any(a.action_type == 'fellowship_reveal_place'
                   and rr.sanitize(str(a.params.get('region'))) == placement
                   for a in legal):
            continue
        matched.append((candidate, anchor))
    if len(matched) != 1:
        return None, None, 'postdamage_prefix_unproven'
    candidate, anchor = matched[0]
    line = int(anchor['line'])
    pool, discarded, reason = _reveal_public_pool(batches, line)
    if reason:
        return None, None, reason
    st = ai.clone_state(candidate['state'])
    st.hunt_pool, st.hunt_discard = pool, discarded
    from rendered_hunt_observation import mark as mark_hunt_observation
    mark_hunt_observation(st, line, 'verified_reveal_prefix')
    _install_ledger_hands(st, batch, at_line=line)
    _install_persistent_cards(st, line, killable=True)
    # The logged card has already been spent even if the player clears its
    # Current-card piece only after moving the Fellowship pawn.
    spent = batch.get('_rx_hunt_reduction_owned')
    if spent:
        st.persistent_cards.remove_card(spent)
        discard = st.discard_face_up.setdefault('fp_character', [])
        if spent not in discard:
            discard.append(spent)
    prefix_snapshot = anchor['decision_truth']
    if isinstance(prefix_snapshot.get('_rx_rings'), dict):
        _stamp_rings(st, prefix_snapshot['_rx_rings'])
    if isinstance(prefix_snapshot.get('_rx_tokens'), dict):
        _stamp_tokens(st, prefix_snapshot['_rx_tokens'])
    if (isinstance(anchor.get('_rx_pools'), dict)
            and getattr(st, '_fellowship_move_die_pending', None) is None):
        st._rx_full_pools = anchor['_rx_pools']
        st.dice = _install_full_pools(st._rx_full_pools)
    return st, {'kind': 'engine_settled_hunt', 'tile_line': candidate['tile_line'],
                'placement_line': line, 'public_pool': 'ordinary_prefix_v1'}, None


def _emit_reveal_placements(batches, stats):
    """USER truth-table fix #5 (game01_verification.md): on a sourced
    reveal (hunt tile with reveal / Eye tile / revealing card), the FP
    chooses WHERE the pawn lands (<= progress away, never an FP
    city/SH).  Gesture = fellowship region move(s) + a
    'Previously on Fellowship Track N' flip in the POST-stage window
    (pre-stage resets are declarations, handled separately).  GAME01:
    T1 Hollin, T6 Dagorlad, T6 Morannon (Search)."""
    import ai
    import rendered_replay as rr
    from game_state import Side
    out = []
    for b in batches:
        if b.get('_rx_hunt_exclusion'):
            continue
        recs = b['records']
        stage_seen = False
        post = []
        for r in recs:
            if r.get('op') == 'move' and r.get('die_stage'):
                stage_seen = True
                continue
            if stage_seen:
                post.append(r)
        if not stage_seen:
            post = recs
        # W3-001 (COORD C-BO-1): the FSP-counter dialect writes the reveal
        # as '<FP> flips FSP in Fellowship Track N. Previously on
        # Fellowship Track N' (the counter reset IS the 'Previously'
        # report) - accept the FSP object next to the figure's.
        flips = [r for r in post if r.get('op') == 'flip'
                 and r.get('revealed')
                 and ('Fellowship' in str(r.get('obj', ''))
                      or str(r.get('obj', '')).strip().lower() == 'fsp')]
        if not flips:
            # GAME19 dialect: the reveal flip has NO 'Previously' marker
            # at all - accept a plain Fellowship flip when the window
            # also shows the FSP track RESET (the reveal signature)
            if any(r.get('op') == 'move'
                   and str(r.get('obj', '')).strip() == 'FSP'
                   and 'Fellowship Track 0' in str(r.get('to', ''))
                   for r in post):
                flips = [r for r in post if r.get('op') == 'flip'
                         and 'Fellowship' in str(r.get('obj', ''))]
        if not flips:
            continue
        if (b.get('truth_before') or {}).get('fel', {}).get('mordor'):
            # W3-001: on the Mordor track a reveal never places the pawn
            # (SlickRick L1413 'Previously on Fellowship Track 1' is a
            # HIDE after a mis-dragged counter) - the engine menu is
            # empty there (ai._get_reveal_placement_actions) and a
            # figure flip inside Mordor parses as a Mordor step anyway
            stats['drop_reasons']['revplace_mordor_track'] += 1
            continue
        _fsp_flip9 = (str(flips[0].get('obj', '')).strip().lower()
                      == 'fsp')
        n = int(flips[0].get('prev_track', 0) or 0)
        if n < 1:
            # GAME19 dialect: the flip line carries no 'Previously on
            # Fellowship Track N' - the progress shows as a separate
            # FSP track RESET move (N -> 0) in the same window
            for r in post:
                if r.get('op') == 'move' \
                        and str(r.get('obj', '')).strip() in ('FSP',) \
                        and 'Fellowship Track 0' in str(r.get('to', '')):
                    import re as _re9
                    _mm9 = _re9.search(r'Fellowship Track (\d+)',
                                       str(r.get('from', '')))
                    if _mm9:
                        n = max(n, int(_mm9.group(1)))
        if n < 1:
            continue
        has_tile = any(it.get('kind') == 'draw_tile' for it in b['rng'])
        fel_reg_b = rr.sanitize(str(b['truth_before']['fel'].get('region')
                                    or ''))
        naz_to_fel = any(
            r.get('op') == 'move' and 'Nazgul' in str(r.get('obj', ''))
            and rr.sanitize(str(r.get('to', ''))) == fel_reg_b
            for r in post)
        _reveal_card9 = any(
            r.get('op') == 'move' and '/' in str(r.get('obj', ''))
            and any(k9 in str(r.get('obj', '')).lower()
                    for k9 in ('nazgul search', 'the palant'))
            for r in b['records'])
        if not (has_tile or naz_to_fel or b['kind'] == 'hunt'
                or _reveal_card9):
            stats['drop_reasons']['revplace_unsourced'] += 1
            continue
        placement = rr.sanitize(str(flips[0].get('region', '')))
        if _fsp_flip9:
            # W3-001: the FSP flip line names the TRACK, not a region -
            # the pawn stays where it was unless a figure move follows
            placement = fel_reg_b
        # the human may continue the placement drag AFTER the flip
        # (GAME02 T3: Rivendell->Dagorlad, flip, ->North Ithilien) -
        # the NET pawn destination is the LAST fellowship region move
        # in the window, not the region named on the flip line
        _lastmv9 = None
        for r in post:
            if r.get('op') == 'move' \
                    and 'The Fellowship' in str(r.get('obj', '')) \
                    and rr.zone_kind(str(r.get('to', '')))[0] == 'region' \
                    and 'Mordor' not in str(r.get('to', '')):
                _lastmv9 = rr.sanitize(str(r.get('to', '')))
        if _lastmv9:
            placement = _lastmv9
        stats['detected']['reveal_place'] += 1
        st, evidence, reason = _reveal_verified_state(b, batches, n, placement, stats)
        if st is None:
            stats['drop_reasons']['revplace_' + reason] += 1
            continue
        try:
            legal = ai.get_legal_actions(st, Side.FREE_PEOPLES)
        except Exception:
            stats['drop_reasons']['revplace_legal_exc'] += 1
            continue
        hidx = _match_index(
            legal, lambda a: rr.sanitize(str(a.params.get('region', '')))
            == placement)
        if hidx is None or len(legal) < 2:
            stats['drop_reasons']['revplace_unmatched' if hidx is None
                                  else 'revplace_forced'] += 1
            continue
        stats['kinds_emit']['fellowship_reveal_place'] += 1
        sample = _emit_sample(st, Side.FREE_PEOPLES, legal, hidx,
                              'fellowship_reveal_place', b['turn'])
        sample['reveal_evidence'] = evidence
        out.append(sample)
    return out


_COV_CATS = {'declared': 1, 'declined': 2, 'revealed': 3,
             'declined_nowindow': 4, 'ambiguous': 5}


def _emit_declarations(batches, stats):
    """Turn-start declaration decisions + PER-TURN COVERAGE (v32
    gate-(e), USER 2026-08-03): every eligible turn (fellowship not in
    Mordor at the turn's declare point) contributes exactly one
    coverage record â€” DECLARED (placement / mordor-entry / progress-0
    in-place heal per the USER counting rule), DECLINED (window
    existed, pass-labeled sample emitted; progress-0 windows with an
    effectful in-place menu included â€” the old progress>=1 gate is
    gone), REVEALED (no window existed â€” its own category, never a
    silent skip), DECLINED-NOWINDOW (hidden at progress 0 with an
    empty declare menu: live play opens no window either; coverage
    record only, a [pass]-only menu carries no CE gradient), or
    AMBIGUOUS (pre-stage tile draw).  Returns (samples, cov) where cov
    rows carry {turn, cat, reg, x} â€” x the fp16 state encoding at the
    declare point (zeros + xok=0 when reconstruction fails)."""
    import ai
    import game_state as gs
    import rendered_census as rc
    from game_state import Side
    import rendered_replay as rr
    out = []
    cov = []

    def _cov_x(bb, turn):
        try:
            st = _recon_state(bb['truth_before'], turn)
            st._declare_phase = True
            _ensure_decks(st)              # T3 staging parity with live
            return np.asarray(ai.encode_state(st, Side.FREE_PEOPLES),
                              dtype=np.float16)
        except Exception:
            return None

    def _rec(turn, bb, cat, x=None):
        fel = (bb.get('truth_before') or {}).get('fel') or {}
        reg = rr.sanitize(str(fel.get('region') or ''))
        cov.append({'turn': int(turn), 'cat': _COV_CATS[cat],
                    'reg': ai.REGION_INDEX.get(reg, -1),
                    'x': (_cov_x(bb, turn) if x is None else
                          np.asarray(x, dtype=np.float16))})

    # USER truth-table fix #4 (game01_verification.md): declarations are
    # PRE-DICE gestures (pawn region moves + a 'Previously on Fellowship
    # Track N' flip with NO reveal source, often followed by the unflip
    # of the progress-reset trick).  Batch truth-diffs cannot see them
    # (the pre-stage records precede the first batch's window), so detect
    # from the FIRST batch's record stream directly, in the pre-stage
    # region (records before the die_stage record).
    first_of_turn = {}
    for b in batches:
        first_of_turn.setdefault(b['turn'], b)
    for turn in sorted(first_of_turn):
        b = first_of_turn[turn]
        fel_tb = (b.get('truth_before') or {}).get('fel') or {}
        pre = []
        for r in b['records']:
            if r.get('op') == 'move' and r.get('die_stage'):
                break
            pre.append(r)
        fel_moves = [r for r in pre if r.get('op') == 'move'
                     and 'The Fellowship' in str(r.get('obj', ''))
                     and 'Mordor' not in str(r.get('to', ''))]
        flips = [r for r in pre if r.get('op') == 'flip'
                 and r.get('revealed') and 'Fellowship'
                 in str(r.get('obj', ''))]
        mordor_pre = [r for r in pre if r.get('op') == 'move'
                      and 'The Fellowship' in str(r.get('obj', ''))
                      and 'Mordor 0' in str(r.get('to', ''))]
        # Named special-tile captions can also be Event-card draws. Treat that
        # ambiguity as a possible chance boundary, never silently look past it.
        tile_pre = [r for r in pre if r.get('op') == 'draw'
                    and rc._tile_obj(gs, str(r.get('what', '')).strip().lower()
                                     .replace('on, on they went', 'on on they went')) is not None]
        first_tile = min((int(r['line']) for r in tile_pre), default=float('inf'))
        # Later Hunt/reveal placements cannot change the preceding nomination.
        fel_moves = [r for r in fel_moves if int(r['line']) < first_tile]
        flips = [r for r in flips if int(r['line']) < first_tile]

        def actual_prefix(anchor):
            # Take the complete public snapshot at the actual nomination,
            # before healing, Balrog, guide replacement, or Mordor refill.
            snap = anchor.get('decision_truth')
            if (not isinstance(snap, dict) or not snap.get('fel', {}).get('hidden')
                    or snap.get('fel', {}).get('mordor')
                    or any(r['line'] < anchor['line'] for r in tile_pre)):
                stats['drop_reasons']['declare_predecision_unproven'] += 1
                return None
            return dict(b, truth_before=snap, start_line=int(anchor['line']))
        if mordor_pre:
            # The automatic entry itself does not prove an optional in-place
            # declaration. Require a preceding actual region nomination.
            stats['detected']['declare_mordor_entry'] += 1
            nominations = [r for r in fel_moves if r['line'] < mordor_pre[0]['line']]
            if not nominations:
                stats['drop_reasons']['declare_entry_without_nomination'] += 1
                _rec(turn, b, 'ambiguous')
                continue
            b2 = actual_prefix(nominations[0])
            if b2 is None:
                _rec(turn, b, 'ambiguous')
                continue
            placement = rr.sanitize(str(nominations[-1].get('to', '')))
            s = _emit_one_declare(b2, placement, 'mordor', stats)
            if s is not None:
                out.append(s)
            _rec(turn, b2, 'declared',
                 x=None if s is None else s['x'])
            continue
        # gate-(e) ELIGIBILITY: fellowship already ON the Mordor track
        # at the turn's declare point -> declarations do not exist;
        # the turn leaves the denominator (counted for the report).
        if fel_tb.get('mordor'):
            stats['detected']['declare_cov_inmordor'] += 1
            continue
        if tile_pre and not fel_moves and not flips:
            stats['drop_reasons']['declare_tile_pre_ambiguous'] += 1
            _rec(turn, b, 'ambiguous')
            continue
        n = 0
        if flips:
            n = int(flips[0].get('prev_track', 0) or 0)
        if n < 1:
            # GAME19 dialect: NO flip at all - the declaration is the
            # pre-stage pawn REGION move + the FSP track RESET (N->0),
            # often with the declare-heal corruption walk alongside
            if fel_moves:
                for r in pre:
                    if r.get('op') == 'move' \
                            and str(r.get('obj', '')).strip() == 'FSP' \
                            and 'Fellowship Track 0' in str(
                                r.get('to', '')):
                        _mm9 = re.search(r'Fellowship Track (\d+)',
                                         str(r.get('from', '')))
                        if _mm9:
                            n = max(n, int(_mm9.group(1)))
        if n < 1:
            # no PLACEMENT declaration gesture this turn start
            if not fel_tb.get('hidden', True):
                # REVEALED at the declare point: no window existed â€”
                # its own coverage category (gate-(e)), never a
                # silent skip.
                stats['drop_reasons']['decline_gate_revealed'] += 1
                _rec(turn, b, 'revealed')
                continue
            # USER counting rule (v32 gate-(e)): a pre-stage lone
            # corruption DOWN-walk with progress_before == 0 is the
            # in-place HEAL declare -> category DECLARED.  With
            # progress > 0 a heal walk does NOT count as declared
            # (falls through to the decline path).
            _heal9 = False
            for r in pre:
                if r.get('op') == 'move' and \
                        str(r.get('obj', '')).strip().lower() \
                        == 'corruption':
                    _hf9 = re.search(r'Fellowship Track (\d+)',
                                     str(r.get('from', '')))
                    _ht9 = re.search(r'Fellowship Track (\d+)',
                                     str(r.get('to', '')))
                    if _hf9 and _ht9 and int(_ht9.group(1)) \
                            == int(_hf9.group(1)) - 1:
                        _heal9 = True
                        break
            try:
                _prog9 = int(fel_tb.get('progress', 0) or 0)
            except (TypeError, ValueError):
                _prog9 = 0
            if _heal9 and _prog9 == 0:
                stats['detected']['declare_heal0'] += 1
                s = _emit_one_heal_declare(b, stats)
                if s is not None:
                    out.append(s)
                _rec(turn, b, 'declared',
                     x=None if s is None else s['x'])
                continue
            # v29/v32: the human DECLINED the declare window; emit the
            # pass-labeled twin (progress-0 windows included).
            s, _code9 = _emit_one_decline(b, stats)
            if s is not None:
                out.append(s)
                _rec(turn, b, 'declined', x=s['x'])
            elif _code9 == 'no_declares':
                _rec(turn, b, 'declined_nowindow')
            else:
                _rec(turn, b, 'declined')
            continue
        stats['detected']['declare_progress_reset'] += 1
        placement = (rr.sanitize(str(fel_moves[-1].get('to')))
                     if fel_moves else
                     rr.sanitize(str(flips[0].get('region', ''))))
        b2 = actual_prefix(fel_moves[0] if fel_moves else flips[0])
        if b2 is None:
            _rec(turn, b, 'ambiguous')
            continue
        s = _emit_one_declare(b2, placement, 'placement', stats)
        if s is not None:
            out.append(s)
        _rec(turn, b2, 'declared', x=None if s is None else s['x'])
    out.extend(stats.pop('__declaration_followups', []))
    return out, cov


# ==========================================================================
# PASS DETECTION  (amendment 1.223, USER 2026-07-28)
# --------------------------------------------------------------------------
# "A pass happens when one player acts twice in a row or Shadow goes
# first.  That ALWAYS means it's a pass."  Detection is STRUCTURAL from
# the action-slot sequence: the action phase alternates starting with
# Free Peoples; whenever the expected side is skipped while still
# holding unused action dice, that side passed â€” emit the sample.
# ACTION TOKENS occupy a turn-order slot WITHOUT spending a die (USER
# 2026-07-28: 'tokens let players manipulate turn order... FP can get
# the first and last action') â€” a token use fills its side's slot and
# must not read as a pass.  Chat lines are NOT consulted (structural
# detection only).  An out-of-dice skip is NOT a pass (no decision
# existed).
# NEVER SKIP DICE â€” a pass is a turn-ORDER pass, not a discarded die.
# ==========================================================================
_DIE_FACES_ACT = ('A', 'C', 'M', 'P', 'H', 'W')
# FP action-token USE in the rendered stream ('Hunt for the ring
# (Political)'/'(Card Draw)' leaving FP Dice for Current Action)
_TOKEN_USE_RE = re.compile(
    r'moves Hunt for the ring \((Political|Card|Card Draw)\) '
    r'from FP Dice to Current (Action|FP Card)', re.I)
# W1-011: the module renders the draw token as '(Card)' (never '(Card
# Draw)') and some clients stage either token into 'Current FP Card'
# (2024-08-11 L45/L98); a token use the walk cannot see became a
# phantom FP pass event at the next Shadow die (9 events in 8 games, 2
# of them emitted as rows: 2023-12-25 L34, 2024-08-11 L101; the other
# 7 were pass-illegal under 1.223 and never reached the ledger).


def _emit_one_pass(snap, passer, actor, init, spent, turn, hand_line, stats):
    import ai
    import game_state as gs
    from game_state import Side
    rem_p = init[passer] - spent[passer]        # Counter: drops non-positive
    rem_a = init[actor] - spent[actor]
    np_ = sum(rem_p.values())
    na_ = sum(rem_a.values())
    if np_ <= 0 or na_ <= 0:
        stats['drop_reasons']['pass_no_dice'] += 1
        return None
    # 1.223: structural evidence WINS â€” no count-relation drop.  The
    # engine gate is now dice-only, so a structurally evident pass with
    # fewer dice always finds 'pass' in the menu; the equal/greater
    # edge is counted for visibility (pass_unmatched below catches the
    # menu miss rather than silently fabricating one).
    if np_ >= na_:
        stats['detected']['pass_geq_dice'] += 1
    try:
        st = _recon_state(snap, turn)
    except Exception:
        stats['drop_reasons']['pass_recon'] += 1
        return None
    pside = Side.SHADOW if passer == 'SHADOW' else Side.FREE_PEOPLES
    st.dice = {Side.FREE_PEOPLES: [], Side.SHADOW: []}
    for sstr, rem in ((passer, rem_p), (actor, rem_a)):
        sd = Side.SHADOW if sstr == 'SHADOW' else Side.FREE_PEOPLES
        pool = gs.SH_FACES if sd == Side.SHADOW else gs.FP_FACES
        for letter, cnt in rem.items():
            f = _letter2face().get(letter)
            if f is None or cnt <= 0:
                continue
            for _ in range(cnt):
                d = gs.ActionDie(pool, sd)
                d.current = f
                d.used = False
                st.dice[sd].append(d)
    _install_ledger_hands(st, {'start_line': hand_line}, at_line=hand_line)
    try:
        legal = ai.get_legal_actions(st, pside)
    except Exception:
        stats['drop_reasons']['pass_legal_exc'] += 1
        return None
    hidx = _match_index(legal, lambda a: a.action_type == 'pass')
    if hidx is None:
        stats['drop_reasons']['pass_unmatched'] += 1
        return None
    stats['kinds_emit']['pass'] += 1
    return _emit_sample(st, pside, legal, hidx, 'pass', turn)


def _emit_passes(batches, pools, stats, rendered_path):
    """One pass BC sample per STRUCTURAL turn-order pass (1.223).

    The action phase alternates starting with Free Peoples; whenever
    the expected side is skipped (opponent acts twice in a row, or
    Shadow takes the turn's first action) while still holding unused
    action dice, that side PASSED â€” emit the sample.  Out-of-dice
    skips are not passes.  Dice = turn roll pool minus faces spent so
    far; state = the last die-batch's truth (turn-start truth for an
    opening pass)."""
    import rendered_replay as rr
    out = []
    u2side = {}
    for b in batches:
        for r in b['records']:
            if r.get('op') == 'move' and r.get('die_stage'):
                u2side[r.get('player')] = r['die_stage'][0]
    turn_batches = {}
    for b in batches:
        turn_batches.setdefault(b['turn'], []).append(b)
    turn = 1
    # init dice per (turn, side) scanned DIRECTLY from the letter-roll lines
    # ('%player rolls P M E A P A'): the turn-start action roll PRECEDES the
    # first die-stage, so _roll_pools (batch rng only) loses it.  First roll per
    # side/turn = the initial allocation; natural Eye ('E') faces excluded (they
    # go to the hunt box, not the action pool).
    turn_init = {}
    turn_roll_line = {}
    token_slots = {}
    with open(rendered_path, encoding='utf-8', errors='ignore') as f:
        for i, raw in enumerate(f, 1):
            s = raw.strip()
            if not s.startswith('%'):
                continue
            body = s[1:].strip()
            mt = rr._TURN_RE.match(body)
            if mt:
                turn = int(mt.group(1))
                continue
            ml = rr._ROLL_LET_RE.match(body)
            if ml:
                rside = u2side.get(ml.group(1).strip())
                if rside is not None:
                    d = turn_init.setdefault(turn, {})
                    if rside not in d:          # first roll per side/turn
                        d[rside] = Counter(f for f in ml.group(2).split()
                                           if f in _DIE_FACES_ACT)
                        turn_roll_line.setdefault(turn, {})[rside] = i
                continue
            if _TOKEN_USE_RE.search(body):
                # FP action-token use = an FP turn-order SLOT (no die)
                token_slots.setdefault(turn, []).append(i)

    # W2-038: dice that LEAVE the pool without a die-stage are not
    # available for the rest of the turn - The Day Without Dawn's
    # Will-of-the-West dice (FP Dice -> Current FP Card), The Lidless
    # Eye's dice into the Hunt box, a die killed in its pool (a rolled-
    # too-many correction, an Event die burned), a die removed to
    # Reinforcements.  They are neither spent by a die batch nor a
    # turn-order slot, so the 1.223 fewer-dice count read at every
    # later pass slot was off by them (2023-12-25 L197: FP passed with
    # its W die on the card; the model saw 2 v 2 -> pass_unmatched).
    # Events are taken from the NET (undo-resolved) batch records, only
    # AFTER that side's turn roll (pre-roll pool corrections are already
    # in the roll line), Eye faces excluded (never in the action pool);
    # a die coming back to its pool from an off-pool zone, or staged
    # from a card slot (W4-018 via-card dialect), re-credits one.
    removal_ev = _pass_pool_removals(batches, turn_roll_line)

    # ---- structural alternation walk (1.223) --------------------------
    def _other(sd):
        return 'SHADOW' if sd == 'FREE_PEOPLES' else 'FREE_PEOPLES'

    for t in sorted(turn_batches):
        die_ev = [('die', b['start_line'], b['die_side'], b)
                  for b in turn_batches[t]
                  if b.get('die_face') in _DIE_FACES_ACT
                  and b.get('die_side') in ('FREE_PEOPLES', 'SHADOW')]
        tok_ev = [('token', ln, 'FREE_PEOPLES', None)
                  for ln in token_slots.get(t, [])]
        seq = sorted(die_ev + tok_ev, key=lambda e: e[1])
        if not any(e[0] == 'die' for e in seq):
            continue
        init = turn_init.get(t, {})
        init = {'FREE_PEOPLES': init.get('FREE_PEOPLES', Counter()),
                'SHADOW': init.get('SHADOW', Counter())}
        spent = {'FREE_PEOPLES': Counter(), 'SHADOW': Counter()}
        removed = {'FREE_PEOPLES': Counter(), 'SHADOW': Counter()}
        rm_ev = removal_ev.get(t, [])
        rm_i = 0
        first_b = next(e[3] for e in seq if e[0] == 'die')
        expected = 'FREE_PEOPLES'       # FP holds the turn's first slot
        prev_b = None
        for kind, ln, actor, b in seq:
            # W2-038: pool exits / kills / re-credits before this slot
            while rm_i < len(rm_ev) and rm_ev[rm_i][0] < ln:
                _, rside, rface, rn = rm_ev[rm_i]
                rm_i += 1
                if rn < 0:
                    rn = -min(-rn, removed[rside][rface])
                if rn:
                    removed[rside][rface] += rn
                    stats['detected']['pass_pool_removal'] += 1
            gone = {sd: spent[sd] + removed[sd] for sd in spent}
            if actor != expected:
                # the expected side was skipped: a PASS if it still
                # holds unused action dice, an out-of-dice skip if not
                rem_e = init[expected] - gone[expected]
                if sum(rem_e.values()) > 0:
                    stats['detected']['pass'] += 1
                    snap = (prev_b['truth'] if prev_b is not None
                            else first_b.get('truth_before'))
                    if snap is None:
                        stats['drop_reasons']['pass_no_context'] += 1
                    else:
                        s9 = _emit_one_pass(snap, expected, actor, init,
                                            gone, t, ln, stats)
                        if s9 is not None:
                            out.append(s9)
            if kind == 'die':
                spent[actor][b['die_face']] += 1
                prev_b = b
            else:
                stats['detected']['token_slot'] += 1
            expected = _other(actor)
    return out


_POOL_ZONES = {'FP Dice': 'FREE_PEOPLES', 'SA Dice': 'SHADOW'}
_DIE_PIECE_RE = re.compile(
    r'^(?:\[(\d+)\])?\s*(Free|Shadow)\s+Action\s+Die\s+\(([ACEHMPW])\)\s*$',
    re.I)


def _pass_pool_removals(batches, turn_roll_line):
    """W2-038: {turn: [(line, side, face, +n | -n), ...]} of action dice
    leaving / re-entering their pool OUTSIDE a die stage, in line order,
    from the net batch records.  +n = a pool exit to any zone other than
    Current Action (card slot, Hunt box, Reinforcements, Casualties,
    a 'rolled too many' correction) or a kill in the pool zone; -n = the
    same side/face coming back into its pool from an off-pool zone.  A
    die parked on a card slot and staged from there (W4-018 via-card
    dialect: the die_stage record's from-zone is the park zone) was never
    removed - its park event is cancelled outright.  Only events after
    that side's turn roll count (the roll line already reflects earlier
    corrections); Eye faces are never in the action pool."""
    import rendered_replay as rr
    ev = {}
    parked = {}       # (turn, side, face, zone) -> [event lists] (LIFO)

    def pieces(objs):
        out = []
        for p in objs:
            m = _DIE_PIECE_RE.match(p.strip())
            if not m:
                continue
            n = int(m.group(1)) if m.group(1) else 1
            side = ('FREE_PEOPLES' if m.group(2).lower() == 'free'
                    else 'SHADOW')
            out.append((side, m.group(3).upper(), n))
        return out

    def objs_of(obj):
        s = (obj or '').strip()
        if s.startswith('group (') and s.endswith(')'):
            return rr.split_group(s[len('group ('):-1])
        return [s]

    for b in batches:
        t = b.get('turn')
        for r in b.get('records', ()):
            ln = r.get('line')
            op = r.get('op')
            if not isinstance(ln, int) or op not in ('move', 'kill'):
                continue
            if op == 'kill':
                zone = (r.get('zone') or '').strip()
                if zone not in _POOL_ZONES:
                    continue
                for side, face, n in pieces(r.get('pieces') or []):
                    if _POOL_ZONES[zone] != side or face not in _DIE_FACES_ACT:
                        continue
                    if ln <= turn_roll_line.get(t, {}).get(side, 10 ** 9):
                        continue
                    ev.setdefault(t, []).append((ln, side, face, n))
                continue
            frm = (r.get('from') or '').strip()
            to = (r.get('to') or '').strip()
            if frm == to:
                continue
            if frm in _POOL_ZONES and not r.get('die_stage'):
                if to == 'Current Action':
                    continue
                for side, face, n in pieces(objs_of(r.get('obj'))):
                    if _POOL_ZONES[frm] != side or face not in _DIE_FACES_ACT:
                        continue
                    if ln <= turn_roll_line.get(t, {}).get(side, 10 ** 9):
                        continue
                    e = [ln, side, face, n]
                    ev.setdefault(t, []).append(e)
                    parked.setdefault((t, side, face, to), []).append(e)
            elif frm not in _POOL_ZONES and frm != 'Current Action':
                if r.get('die_stage'):
                    # via-card stage: the park was a step of the staging
                    for side, face, n in pieces(objs_of(r.get('obj'))):
                        lst = parked.get((t, side, face, frm))
                        while n > 0 and lst:
                            e = lst[-1]
                            k = min(n, e[3])
                            e[3] -= k
                            n -= k
                            if e[3] == 0:
                                lst.pop()
                elif to in _POOL_ZONES:
                    for side, face, n in pieces(objs_of(r.get('obj'))):
                        if face not in _DIE_FACES_ACT:
                            continue
                        if _POOL_ZONES[to] != side:
                            continue
                        ev.setdefault(t, []).append([ln, side, face, -n])
    out = {}
    for t in ev:
        out[t] = sorted((tuple(e) for e in ev[t] if e[3]),
                        key=lambda e: e[0])
    return out


def _catalog_unemitted(batch, stats):
    """Count visible gestures separately from their sample emitters.

    This function emits no samples. Several counted families now have their
    own emitters below; detection counts alone do not establish completeness.
    """
    det = stats['detected']
    # RETURN OF THE WITCH-KING non-decision anomaly flag (spec 1.216: "NO
    # choices â€” flag if the log shows WK anywhere but Angmar").  Gated on the
    # RotWK card actually being PLAYED this batch (a normal WK muster may go
    # anywhere â€” only RotWK forces Angmar).
    _pt = _card_play_title(batch)
    _is_rotwk = _pt is not None and 'return' in str(_pt).lower() and \
        'witch' in str(_pt).lower()
    if _is_rotwk:
        det['rotwk_played'] += 1
        for r in batch['records']:
            if r.get('op') != 'move':
                continue
            obj = str(r.get('obj', '')).lower()
            frm = str(r.get('from', '')).lower()
            to = str(r.get('to', '')).lower()
            if 'witch' in obj and 'reinforcement' in frm and to \
                    and 'angmar' not in to:
                det['rotwk_wk_not_angmar'] += 1
    for r in batch['records']:
        op = r.get('op')
        if op == 'move':
            obj = str(r.get('obj', '')).lower()
            frm = str(r.get('from', '')).lower()
            to = str(r.get('to', '')).lower()
            if 'guide of the fellowship' in to:
                det['choose_guide'] += 1
            if 'mordor' in to and 'track' in frm:
                det['fellowship_declare_mordor'] += 1
            if 'dwarven ring' in obj:
                det['dwarven_ring'] += 1
            if 'token' in obj:
                # muster/action token gesture (spec 1.216 META) â€” censused;
                # no engine muster_token action exists to legality-match yet
                det['muster_token_gesture'] += 1
            if '/' in str(r.get('obj', '')) and 'current' in to and \
                    'card' in to:
                det['combat_card_play'] += 1
        elif op == 'kill':
            joined = ' '.join(str(p) for p in r.get('pieces', [])).lower()
            if 'dwarven ring' in joined:
                det['dwarven_ring'] += 1
    # ---- hunt-reduction / redraw table cards (spec 1.216 items 2/3/4) ---
    # Axe and Bow / Horn of Gondor (-1 hunt damage), Mithril Coat and Sting
    # (kill + redraw a hunt tile), Wizard's Staff (prevent the Shadow tile
    # draw).  Each is a binary FP decision evidenced by the persistent/table
    # card being KILLED in a hunt-context batch.  The table ledger
    # (_build_table_ledger) reconstructs which cards sit on the table from
    # the rendered gestures and _install_persistent_cards installs them so
    # the engine's OWN interposes (game_logic._offer_hunt_reduction /
    # _resolve_hunt_tile) open and legality-match - EMITTED via
    # _emit_hunt_cards; this gesture census is an independent cross-check
    # tally alongside the emitted counts.
    _hunt_ctx = batch['kind'] in ('hunt', 'combat?') or any(
        it.get('kind') == 'draw_tile' for it in batch['rng'])
    for r in batch['records']:
        blob = None
        if r.get('op') == 'kill':
            blob = ' '.join(str(p) for p in r.get('pieces', [])).lower()
        elif r.get('op') == 'move' and _BOARD_DEST_RE.match(
                str(r.get('from', '')).strip().lower()):
            blob = str(r.get('obj', '')).lower()
        if not blob:
            continue
        for cardnm, key in (('mithril coat', 'mithril_use'),
                            ('axe and bow', 'axebow_use'),
                            ('horn of gondor', 'horn_use'),
                            ("wizard's staff", 'wizardstaff_use'),
                            ('wizard s staff', 'wizardstaff_use')):
            if cardnm in blob:
                det['huntcard_%s%s' % (key, '' if _hunt_ctx else '_nonhunt')] \
                    += 1
    # ---- heal-declaration census (rough tally) -------------------------
    # The 0-progress declare that HEALS in a free stronghold/city is itself
    # emitted as a DECLARED sample by _emit_one_heal_declare (v32 gate-(e),
    # USER 2026-08-03); this stays a rough census only (a companion/Athelas
    # heal also lowers corruption -> upper bound).
    fb = batch['truth_before']['fel']
    fa = batch['truth']['fel']
    if (fa.get('corruption', 0) < fb.get('corruption', 0)
            and int(fa.get('progress', 1) or 0) == 0
            and not fa.get('mordor')):
        det['declare_heal_candidate'] += 1


def process_batch(batch, stats):
    """Drive + legality-match ONE batch.  Returns a sample dict, or None.
    Mutates stats Counters (kinds_emit, drop_reasons, drove, diverged,
    nomatch, ambiguous).  Raises _Quarantine on a hard legality no-match."""
    import ai
    import rendered_census as rc
    import rendered_replay as rr
    from game_state import Side
    kind = batch['kind']
    if batch.get('_phase_only'):
        return None

    # CENSUS (1.149): every batch that carries a table-card removal,
    # tagged with the KIND that will handle it - makes the yield gap
    # legible ("never detected" vs "detected elsewhere").
    if _table_kill_title(batch) is not None:
        stats['detected']['tablekill_seen:%s' % kind] += 1

    # ---- resolver batches: hunt / combat (engine-driven) ----------------
    if kind in ('hunt', 'combat', 'combat?'):
        st, side = _mk_state_with_die(batch)
        if st is None:
            stats['drop_reasons']['resolver_recon:' + side] += 1
            return None
        # USER truth-table reverify gap (T2/T6/T8): a card PLAY that
        # opens a hunt-kind batch (Nazgul Strike/Search, Isildur's
        # Bane, Orc Patrol) never reached the play emitter - the
        # resolver path swallowed it.  Emit the play decision
        # ADDITIVELY on a hand-installed clone; the resolver keeps its
        # own un-installed state (certified drive unchanged).
        _pt9 = _card_play_title(batch)
        if _pt9 is not None and kind != 'hunt':
            # on combat batches only a TABLE event play counts (title
            # moved to a Board slot: Corsairs, Fighting Uruk-hai) -
            # plain combat-card commits are the cc family's, not plays.
            # EXCEPTION: a CARD STRIKE (GAME19 T6 Dreadful Spells) is
            # a play, never a commit (W1-001: shared predicate - the
            # actor rolled alone with a strike-effect title; a battle
            # where only the defender rolled is a COMMIT).
            _strike9 = _card_strike(batch)
            if not _strike9 and not any(
                    r.get('op') == 'move'
                    and _pt9 in str(r.get('obj', ''))
                    and str(r.get('to', '')).startswith('Board')
                    for r in batch['records']):
                _pt9 = None
        _play_extra = None
        if _pt9 is not None:
            # This literal event classification remains relevant even when
            # its policy row is refused for unproven legality/payment: its
            # battle must not become a second, unrelated root die spend.
            batch['_rx_die_bought_card'] = _pt9
            try:
                import ai as _aiP
                stP, sideP = _mk_state_with_die(batch,
                                                install_hands=True)
                if stP is not None:
                    legalP = _aiP.get_legal_actions(stP, sideP)
                    _play_extra = _emit_play_event(batch, stP, sideP,
                                                   legalP, _pt9, stats)
            except Exception:
                stats['drop_reasons']['huntplay_exc'] += 1
        if _play_extra is not None:
            # routed to the sample list by the extract loop (avoids
            # threading through every resolver return site)
            stats.setdefault('__extra_samples', []).append(_play_extra)
            # W4-018: the die is spent on the card - the battle below
            # is the card's interior (_combat_target_sample)
            batch['_rx_die_bought_card'] = _pt9
        try:
            if kind == 'hunt':
                _seed_fp_huntbox(st, batch)
                status, diffs, reason = rc._drive_hunt(st, batch)
            else:
                status, diffs, reason = rc._drive_combat(st, batch)
        except Exception as e:
            stats['drop_reasons']['drive_exc:%s' % str(e)[:20]] += 1
            return None
        if status != 'drove':
            stats['drop_reasons']['resolver_skip:%s' % (reason or status)] += 1
            # An un-drivable hunt (Shadow tile-card / special / Mordor) still
            # carries the FP fellowship-MOVE policy decision when it was an FP
            # pawn advance â€” that choice is pure and tile-independent.
            if kind == 'hunt':
                return _fellowship_move_sample(batch, stats)
            return None
        stats['drove'] += 1
        stats.setdefault('_driven_lines', set()).add(
            batch.get('start_line') or 0)
        # a reliable (gate-counting) divergence = engine disagrees with the
        # client-resolved truth -> DROP this batch's sample (gate 5 hygiene).
        if any(d.get('resync') for d in diffs):
            stats['diverged'] += 1
            stats['drop_reasons']['resolver_divergent'] += 1
            return None
        # policy-class (non-resync) diffs are legal loser-choice noise, not a
        # bug â€” the batch is faithfully driven.  We do not label the resolver
        # micro-choice (casualty allocation / hunt sacrifice are the loser's
        # private choice), but for a cleanly-driven COMBAT we DO label the
        # attacker's TARGET decision (which region to assault), and for a
        # cleanly-driven FP hunt the fellowship-MOVE decision â€” both pure,
        # snapshot-faithful choices whose candidate sets are legal.
        if kind in ('combat', 'combat?'):
            _tgt = _combat_target_sample(batch, stats)
            # ONSLAUGHT magnitude (spec 1.216, USER 2026-07-24): the
            # census drive surfaced the decision point (state clone
            # BEFORE the sacrifice + back-calced K).  Reconstruct the
            # SAME pending ai's self-play loop opens (ai.py ~L8759) so
            # get_legal_actions yields the true combat_effect_choice
            # menu, then legality-match K and emit.  Enemy-destroyed
            # battles never defer (engine gate) = moot per user ruling.
            _od = getattr(st, '_onslaught_decision', None)
            _oe = getattr(st, '_onslaught_stash_err', None)
            if _od is None and _oe:
                stats['drop_reasons']['cf_onslaught_stash:%s'
                                      % _oe[:18]] += 1
            if _od is not None:
                try:
                    s9 = _od['state']
                    s9.pending_combat = {
                        'phase': 'combat_effect_choice',
                        'attacker_region': _od['own_r']
                        if _od['role'] == 'attacker' else _od['enemy_r'],
                        'defender_region': _od['enemy_r']
                        if _od['role'] == 'attacker' else _od['own_r'],
                        'attacker_side': _od['own_side']
                        if _od['role'] == 'attacker'
                        else _od['enemy_side'],
                        'defender_side': _od['enemy_side']
                        if _od['role'] == 'attacker' else _od['own_side'],
                        'deciding_side': _od['own_side'],
                        'is_siege': False,
                        'effect_choice': {
                            'stage': 'post',
                            'owner': ('atk' if _od['role'] == 'attacker'
                                      else 'def'),
                            'effect': 'onslaught',
                            'card_id': _od['card_id'],
                            'max_count': _od['mx']}}
                    la9 = ai.get_legal_actions(s9, _od['own_side'])
                    la9 = [a for a in la9
                           if a.action_type == 'combat_effect_choice']
                    if len(la9) >= 2:
                        _pend_note(stats, 'combat_effect', 'open')
                    # W2-008: K is the human's GESTURE (parked units /
                    # Onslaught dice rolled), never the engine-vs-truth
                    # unit delta (MadRobin L1119-1120: 2 Regulars parked,
                    # 2 dice, delta said 3; Thingchilla L501-502: two
                    # Elites parked, 4 dice, delta said 2)
                    _own9 = ('SHADOW' if _od['own_side'] == Side.SHADOW
                             else 'FREE_PEOPLES')
                    _kg9, _why9 = _onslaught_park_k(
                        batch, _own9, {_od['own_r']},
                        batch.get('start_line') or 0, 10 ** 9)
                    if _kg9 is None:
                        stats['drop_reasons']['cf_onslaught_%s'
                                              % _why9] += 1
                        la9 = []
                    else:
                        _od['k'] = int(_kg9)
                    h9 = next((i for i, a in enumerate(la9)
                               if int(a.params.get('count', -2))
                               == _od['k']), None)
                    if la9 and len(la9) >= 2 and h9 is not None:
                        stats['detected']['cc_onslaught_backcalc'] += 1
                        stats.setdefault('_ons_lines', set()).add(
                            batch.get('start_line') or 0)
                        # W2-008: the drive's own onslaught branch dedups
                        # against this K (one row per decision)
                        stats.setdefault('_ons_k', {})[
                            batch.get('start_line') or 0] = int(_od['k'])
                        _pend_note(stats, 'combat_effect', 'emit')
                        smp = _emit_sample(s9, _od['own_side'], la9, h9,
                                           'combat_effect_choice',
                                           batch['turn'])
                        if smp is not None:
                            return ([_tgt, smp] if _tgt is not None
                                    else smp)
                    else:
                        stats['drop_reasons']['cf_onslaught_unclean'] += 1
                except Exception as _e9:
                    stats['drop_reasons'][
                        'cf_onslaught_exc:%s' % str(_e9)[:18]] += 1
            return _tgt
        # a DRIVEN non-combat batch can still carry labelable decisions
        # the resolver never surfaces: the T9 nazgul convergence (ring
        # A->C + three leader-only flights) drives cleanly, so route it
        # through the bookkeeping matchers (eff-face + chain legs).
        # Tight gates: Shadow leader-only flights, no card play, no
        # army-unit move (those classes have their own paths).
        if batch.get('die_side') == 'SHADOW' \
                and _leader_only_moves(batch) \
                and _card_play_title(batch) is None \
                and not [1 for (_f9, _t9, _u9) in
                         _army_move_records(batch) if _u9]:
            st9, side9 = _mk_state_with_die(batch, install_hands=True)
            if st9 is not None:
                try:
                    lg9 = ai.get_legal_actions(st9, side9)
                    return _emit_bookkeeping(batch, st9, side9, lg9,
                                             stats)
                except Exception:
                    pass
        return _fellowship_move_sample(batch, stats)

    # ---- zero-kill battle family (chat-declared, no casualties) --------
    _zk = _emit_chat_battle(batch, stats)
    if _zk is not None:
        return _zk

    # ---- bookkeeping batches: legality-match the primary die-usage ------
    st, side = _mk_state_with_die(batch, install_hands=True)
    if st is None:
        # die face not enumerable (EYE / recon failure) â€” covered as
        # bookkeeping (mirror is client-truth), no sample.
        stats['drop_reasons']['book_recon:' + side] += 1
        return None
    try:
        legal = ai.get_legal_actions(st, side)
    except Exception as e:
        # legal-action enumeration crash on a reconstructed state â€” covered
        # as bookkeeping, not a legality failure (Will-of-West crown crash is
        # fixed by the completed region reconstruction in _recon_state).
        stats['drop_reasons']['legal_exc:%s' % str(e)[:18]] += 1
        return None
    return _emit_bookkeeping(batch, st, side, legal, stats)


def _nation_of(action):
    n = action.params.get('nation', '')
    try:
        return n.name if hasattr(n, 'name') else str(n).upper()
    except Exception:
        return str(n).upper()


class _Quarantine(Exception):
    def __init__(self, reason, detail=None):
        self.reason = reason
        self.detail = detail


# ==========================================================================
# per-game extraction
# ==========================================================================
_ATTACK_FAMILY_PREFIXES = ('sw_', 'attack_', 'cf_', 'cc_')


def _attack_family_drops(stats):
    """W4-033: total of the attack-family emitters' own drop ticks
    (_emit_siege_withdrawal `sw_*`, the attack driver `attack_*`, the
    combat family `cf_*`, combat cards `cc_*`) - sampled before / after a
    batch to tell whether THEY already counted the batch's lost decision."""
    return sum(v for k, v in stats['drop_reasons'].items()
               if str(k).startswith(_ATTACK_FAMILY_PREFIXES))


def _rebucket_drop(stats, reason, n, detected_key):
    """W4-033: move n ticks of drop_reasons[reason] into detected[detected_key]
    (the key is deleted at zero so the manifest's most_common(15) never
    shows a 0 entry).  Meters only - never a sample."""
    stats['drop_reasons'][reason] -= n
    if stats['drop_reasons'][reason] <= 0:
        del stats['drop_reasons'][reason]
    stats['detected'][detected_key] += n


def _rebucket_mcu(stats, new_samples, mcu0, afd0):
    """W4-033 (P meter, rows untouched).  `move_comp_unmatched` ticks in
    _emit_bookkeeping BEFORE the List-2 extras see the batch.  A
    siege-withdrawal / attack batch renders the attacker's units moving
    INTO the defended region (no army_move is legal there - it is the
    attack), so the primary matcher ticks the drop and
    _emit_siege_withdrawal then emits the army_attack (+
    defender_siege_choice) rows for the SAME batch: the decision is
    captured, not dropped (24-game census 116 of 132 ticks,
    reports/W4_mcu24).  Called once per batch with the samples that batch
    added (`new_samples`) and the two meters' values before the batch:
      - an `army_attack` row landed        -> detected
                                              move_comp_recovered_as_attack
      - no row, but an attack-family meter (`sw_* attack_* cf_* cc_*`)
        counted the batch's lost decision  -> detected
                                              move_comp_attack_family_drop
                                              (ONE reason per lost decision;
                                              W2-024 batches, 6 of 132)
      - neither                            -> the tick stays a drop
    Returns the detected key used, or None."""
    d = stats['drop_reasons'].get('move_comp_unmatched', 0) - mcu0
    if d <= 0:
        return None
    phases = set(str(x.get('phase')) for x in new_samples
                 if isinstance(x, dict))
    if 'army_attack' in phases:
        key = 'move_comp_recovered_as_attack'
    elif _attack_family_drops(stats) > afd0:
        key = 'move_comp_attack_family_drop'
    else:
        return None
    _rebucket_drop(stats, 'move_comp_unmatched', d, key)
    return key

_DIE_POOL_ZONES = frozenset({'sa dice', 'fp dice'})


def _hunt_box_stage_return_hesitation(b, si):
    """Prove a first-action die staging undone before any game effect.

    The staged FP face must be unique in its actual preceding roll. Any
    intervening mixed-dice drag must be an immediately reversed pair from
    the known Hunt Box to the FP pool and back, with identical die cargo.
    This deliberately does not reconstruct arbitrary later-turn box edits.
    """
    records = b.get('records') or []
    stage = records[si]
    box = 'the hunt for the ring'
    zone = lambda value: str(value or '').strip().lower()
    if (zone(stage.get('from')) != 'fp dice'
            or zone(stage.get('to')) != box
            or stage.get('die_stage', (None,))[0] != 'FREE_PEOPLES'):
        return None
    tokens = _die_tokens(stage.get('obj'))
    if len(tokens) != 1 or tokens[0] not in (('FREE_PEOPLES', 'C'), ('FREE_PEOPLES', 'W')):
        return None
    if any(r.get('die_stage') for r in records[:si]):
        return None
    own = [r for r in records[:si] if r.get('op') == 'roll_dice'
           and r.get('player') == stage.get('player')]
    other = [r for r in records[:si] if r.get('op') == 'roll_dice'
             and r.get('player') != stage.get('player')]
    if len(own) != 1 or len(other) != 1:
        return None
    face = tokens[0][1]
    if list(own[0].get('faces') or []).count(face) != 1:
        return None
    # No die changes, removals, or earlier actions may invalidate uniqueness.
    roll_line = min(int(own[0]['line']), int(other[0]['line']))
    prefix = [r for r in records[:si] if int(r.get('line') or 0) > roll_line]
    if any(r.get('op') not in ('roll_dice', 'add_eye', 'chat') for r in prefix):
        return None
    shadow = other[0].get('player')
    allocations = [r for r in records[:si] if r.get('op') == 'alloc_eye'
                   and r.get('player') == shadow]
    eyes = len(allocations) + sum(f == 'E' for f in other[0].get('faces', []))
    if not allocations:
        return None
    tail = records[si + 1:]
    if not tail:
        return None
    returned = tail[-1]
    if (returned.get('op') != 'move' or returned.get('obj') != stage.get('obj')
            or returned.get('player') != stage.get('player')
            or zone(returned.get('from')) != box
            or zone(returned.get('to')) != 'fp dice'):
        return None
    between = tail[:-1]
    if len(between) % 2:
        return None
    pairs = []
    available = Counter({tokens[0]: 1, ('SHADOW', 'E'): eyes})
    for first, second in zip(between[::2], between[1::2]):
        if (first.get('op') != 'move' or second.get('op') != 'move'
                or first.get('player') != stage.get('player')
                or second.get('player') != stage.get('player')
                or zone(first.get('from')) != box
                or zone(first.get('to')) != 'fp dice'
                or zone(second.get('from')) != 'fp dice'
                or zone(second.get('to')) != box):
            return None
        cargo = Counter(_die_tokens(first.get('obj')))
        if not cargo or cargo != Counter(_die_tokens(second.get('obj'))) or cargo - available:
            return None
        for record in (first, second):
            remainder = _DIE_TOKEN_RE.sub('', str(record.get('obj', '')))
            if re.sub(r'group|[\s,()]', '', remainder, flags=re.I):
                return None
        pairs.append((first['line'], second['line']))
    b['_rx_hunt_box_hesitation'] = {
        'stage_line': stage['line'], 'return_line': returned['line'],
        'roundtrips': pairs, 'unique_fp_face': face, 'shadow_eyes': eyes}
    return si, len(records) - 1, []


def _stage_return_hesitation(b):
    """W2-031: (si, ri, between) when die-batch `b` is a STAGE-AND-RETURN
    hesitation - the staging player dragged the action die back from
    Current Action into its pool inside the same batch, rolled / drew
    nothing, and gestured nothing of their own in between; else None.
    `between` = the records strictly between the stage and the return
    (opponent piece moves only; the caller requires each to be exactly
    reverted inside the following batch)."""
    recs = b.get('records') or []
    if b.get('rng'):
        return None
    start = int(b.get('start_line') or -1)
    si = next((k for k, r in enumerate(recs)
               if r.get('op') == 'move' and r.get('die_stage')
               and int(r.get('line', -2)) == start), None)
    if si is None:
        return None
    S = recs[si]
    if str(S.get('to', '')).strip().lower() == 'the hunt for the ring':
        return _hunt_box_stage_return_hesitation(b, si)
    ri = None
    for k in range(si + 1, len(recs)):
        r = recs[k]
        if r.get('op') == 'move' and r.get('obj') == S.get('obj') \
                and str(r.get('from', '')).strip().lower() == 'current action' \
                and str(r.get('to', '')).strip().lower() in _DIE_POOL_ZONES:
            if r.get('player') != S.get('player'):
                return None         # the OPPONENT returned it (a real action)
            ri = k
            break
    if ri is None or ri != len(recs) - 1:
        return None                 # no return, or gestures after it
    between = recs[si + 1:ri]
    for r in between:
        if r.get('op') != 'move' or r.get('player') == S.get('player'):
            return None             # the owner did something with the die
    return si, ri, between


def _pre_roll_return_hesitation(g, b, rendered_path=None, raw_path=None, allocation=False):
    """Prove a neutral return or an explicitly requested pre-roll Eye allocation.

    The raw single-PID return must immediately follow the matching stage.
    Trailing setup is retained by the caller; no missing or post-roll evidence
    is interpreted as permission to fold an action.
    """
    if rendered_path is None:
        return None
    records = b.get('records') or []
    si = next((i for i, r in enumerate(records)
               if r.get('die_stage') and r.get('line') == b.get('start_line')), None)
    if si is None or si + 1 >= len(records):
        return None
    stage, returned = records[si:si + 2]
    side = stage['die_stage'][0]
    if allocation and (side != 'SHADOW' or _die_tokens(stage.get('obj')) != [('SHADOW', 'E')]):
        return None
    zone = 'FP Dice' if side == 'FREE_PEOPLES' else 'SA Dice'
    if (stage.get('op') != 'move' or stage.get('from') != zone
            or stage.get('to') != 'Current Action'
            or returned.get('op') != 'move' or returned.get('obj') != stage.get('obj')
            or returned.get('player') != stage.get('player')
            or returned.get('from') != 'Current Action' or returned.get('to') != ('The Hunt For The Ring' if allocation else zone)
            or len(_die_tokens(stage.get('obj'))) != 1
            or re.sub(r'group|[\s,()]', '', _DIE_TOKEN_RE.sub('', stage.get('obj', '')), flags=re.I)):
        return None
    turn_batches = [x for x in g.get('batches', []) if x.get('turn') == b.get('turn')]
    if any(int(r.get('line') or 0) <= int(returned['line'])
           for x in turn_batches for r in x.get('rng', [])):
        return None
    if any(r.get('op') == 'roll_dice' and int(r.get('line') or 0) <= int(returned['line'])
           for x in turn_batches for r in x.get('records', [])):
        return None
    tail = list(records[si + 2:])
    following = next((x for x in turn_batches if int(x.get('start_line') or 0) > int(b.get('start_line') or 0)), None)
    if following is not None:
        tail.extend(r for r in following.get('records', []) if int(r.get('line') or 0) < int(following['start_line']))
    # A succeeding full setup roll provides the affirmative phase boundary.
    if not any(r.get('op') == 'roll_dice' and len(r.get('faces', [])) > 1
               for r in tail):
        return None
    # Folding reuses the pre-stage board snapshot. Every trailing gesture
    # must therefore leave that board unchanged, even after the die returned.
    # Drawn hands and rolled pools are reconstructed at their own prefixes.
    for record in tail:
        op = record.get('op')
        safe = op in ('draw', 'roll_dice', 'alloc_eye', 'add_eye', 'view_hand')
        if op == 'other':
            safe = bool(re.fullmatch(r'\S+ views the (?:Free Peoples|Shadow) event hand\.',
                                     str(record.get('raw', ''))))
        if op == 'move':
            obj = str(record.get('obj', ''))
            safe = obj.strip().lower() == 'dwarven ring' or (bool(_die_tokens(obj))
                and not re.sub(r'group|[\s,()]', '', _DIE_TOKEN_RE.sub('', obj), flags=re.I))
        if op == 'kill':
            pieces = record.get('pieces') or []
            safe = bool(pieces) and all(str(obj).strip().lower() == 'dwarven ring'
                or (bool(_die_tokens(obj)) and not re.sub(r'group|[\s,()]', '',
                    _DIE_TOKEN_RE.sub('', str(obj)), flags=re.I)) for obj in pieces)
        if not safe:
            return None
    if any(r.get('kind') not in ('action_roll', 'add_eye')
           for r in b.get('rng', []) if int(r.get('line') or 0) > int(returned['line'])):
        return None
    import log_census as lc
    if raw_path is None:
        raw_path = _raw_log_path(os.path.basename(str(rendered_path)).replace('.rendered.txt', ''))
    if not raw_path or not os.path.isfile(raw_path):
        return None
    try:
        raw_text = read_raw_log_text(raw_path)
    except RawLogEncodingError as error:
        raise _Quarantine(error.reason, error.detail) from error
    raw = _paired_physical_raw_view(g['batches'], raw_text)
    own_zone = 'Game.2042' if side == 'FREE_PEOPLES' else 'Game.2043'
    allowed = ((lc.BASE_FP_DIE_PIDS | lc.KNOWN_BONUS_FP) if side == 'FREE_PEOPLES'
               else (lc.BASE_SH_DIE_PIDS | lc.KNOWN_BONUS_SH))
    turn = 1
    commands = []
    all_ids = lc.BASE_FP_DIE_PIDS | lc.KNOWN_BONUS_FP | lc.BASE_SH_DIE_PIDS | lc.KNOWN_BONUS_SH
    positions = {pid: ('Game.2042' if pid in lc.BASE_FP_DIE_PIDS else 'Game.2043'
                 if pid in lc.BASE_SH_DIE_PIDS else 'Game.2038' if pid in lc.KNOWN_BONUS_FP else 'Game.2040') for pid in all_ids}
    command_positions = []
    for line in raw:
        mt = re.search(r'<~Game\.7~>(\d+)\b', line)
        if mt:
            turn = int(mt.group(1))
        if turn == b.get('turn') and (line.startswith('$') or '<~Controls.1~>' in line):
            commands.append(line)
            command_positions.append(dict(positions))
        moved = lc.MOVE_RE.search(line)
        if moved:
            for pid in moved.group(1).split():
                if pid in all_ids: positions[pid] = moved.group(3)
        killed = lc.KILL_RE.search(line)
        if killed:
            for pid in killed.group(1).split():
                if pid in all_ids: positions[pid] = 'Game.2039' if pid in (lc.BASE_FP_DIE_PIDS | lc.KNOWN_BONUS_FP) else 'Game.2041'
        rolled = lc.ROLL_RE.search(line)
        if rolled:
            for pid, value in zip(rolled.group(1).split(), rolled.group(2).split()):
                if pid in (lc.BASE_SH_DIE_PIDS | lc.KNOWN_BONUS_SH) and int(value) % 10 == 5:
                    positions[pid] = 'Game.1997'
    raw_stages = []
    for i, line in enumerate(commands):
        move = lc.MOVE_RE.search(line)
        if move and move.group(2) == own_zone and move.group(3) == 'Game.2035':
            ids = move.group(1).split()
            if any(pid in allowed for pid in ids):
                raw_stages.append((i, line.split('<~', 1)[0][1:], ids))
    # net_records preserves ordinal identity even when an earlier hesitation
    # was already removed from batches by the existing normalizer.
    raw_turn = 1
    rendered_stages = []
    for r in g.get('net_records', []):
        if r.get('op') == 'turn':
            raw_turn = r.get('turn', raw_turn)
        if (r.get('op') == 'move' and r.get('from') == zone
                and r.get('to') == 'Current Action' and _die_tokens(r.get('obj'))):
            if raw_turn == b.get('turn'):
                rendered_stages.append(r)
    if len(raw_stages) != len(rendered_stages):
        return None
    ordinal = next((i for i, r in enumerate(rendered_stages) if r.get('line') == stage['line']), None)
    if ordinal is None:
        return None
    ci, player, ids = raw_stages[ordinal]
    if player != stage.get('player') or len(ids) != 1 or ids[0] not in allowed or ci + 1 >= len(commands):
        return None
    if any(lc.ROLL_RE.search(line) or lc.TILE_DRAW_RE.search(line)
           or '<~Controls.1~>' in line for line in commands[:ci]):
        return None
    return_line = commands[ci + 1]
    move = lc.MOVE_RE.search(return_line)
    if (move is None or move.group(1).split() != ids or move.group(2) != 'Game.2035'
            or move.group(3) != ('Game.1997' if allocation else own_zone) or return_line.split('<~', 1)[0][1:] != player):
        return None
    all_dice = lc.BASE_FP_DIE_PIDS | lc.KNOWN_BONUS_FP | lc.BASE_SH_DIE_PIDS | lc.KNOWN_BONUS_SH
    # A later setup allocation can itself be a proven, already folded
    # transfer. Only its exact raw/rendered stage identity is exempt from
    # the next paid-action boundary; an unlogged or ordinary stage is not.
    setup_proofs = {int(p['stage_line']): p for p in (following or {}).get('_rx_pre_roll_allocations', [])}
    setup_indices = set()
    for (command_i, command_player, command_ids), rendered_stage in zip(raw_stages, rendered_stages):
        proof = setup_proofs.get(int(rendered_stage.get('line') or 0))
        if (proof and command_ids == [proof['pid']]
                and command_player == rendered_stage.get('player')):
            setup_indices.add(command_i)
    next_paid_index = len(commands)
    for command_i, line in enumerate(commands[ci + 2:], ci + 2):
        next_move = lc.MOVE_RE.search(line)
        neutral_ids = all_dice | lc.DWARVEN_RING_PIDS
        if next_move and not set(next_move.group(1).split()) <= neutral_ids:
            return None
        killed = lc.KILL_RE.search(line)
        if killed and not set(killed.group(1).split()) <= neutral_ids:
            return None
        if lc.FLIP_RE.search(line) or lc.SWAP_REMOVE_RE.search(line) or lc.SWAP_PLACE_RE.search(line):
            return None
        if (next_move and next_move.group(2) in ('Game.2042', 'Game.2043')
                and next_move.group(3) == 'Game.2035'
                and any(pid in all_dice for pid in next_move.group(1).split())
                and command_i not in setup_indices):
            next_paid_index = command_i
            break
        if '<~Controls.1~>' in line or lc.TILE_DRAW_RE.search(line):
            return None
    if allocation:
        if command_positions[ci].get(ids[0]) != own_zone:
            return None
        shadow_roll = next(((j, lc.ROLL_RE.search(line)) for j, line in enumerate(commands)
                            if ci + 1 < j < next_paid_index and line.split('<~', 1)[0][1:] == player and lc.ROLL_RE.search(line)), None)
        if shadow_roll is None:
            return None
        j, roll = shadow_roll
        pool = {pid for pid, location in command_positions[j].items() if location == own_zone}
        rolled_ids = roll.group(1).split()
        if ids[0] in rolled_ids or len(set(rolled_ids)) != len(rolled_ids) or set(rolled_ids) != pool:
            return None
        rendered_roll = next((r for r in tail if r.get('op') == 'roll_dice' and r.get('player') == player), None)
        try:
            if any(int(v) not in (*range(6), *range(10, 16)) for v in roll.group(2).split()):
                return None
            faces = ['MHACPE'[int(v) % 10] for v in roll.group(2).split()]
        except (IndexError, ValueError):
            return None
        if rendered_roll is None or faces != rendered_roll.get('faces') or len(faces) != len(rolled_ids):
            return None
        # Compare the complete roll sequence in this setup interval.
        # Matching faces alone cannot borrow a roll beyond a paid stage or
        # match the right player's roll at the wrong phase boundary.
        sh_ids = lc.BASE_SH_DIE_PIDS | lc.KNOWN_BONUS_SH
        fp_ids = lc.BASE_FP_DIE_PIDS | lc.KNOWN_BONUS_FP
        raw_rolls = []
        for line in commands[ci + 2:next_paid_index]:
            matched = lc.ROLL_RE.search(line)
            if matched is None:
                continue
            owner = line.split('<~', 1)[0][1:]
            # The physical dice in this roll determine their side. A user
            # can later manipulate the opponent's dice; that later gesture
            # cannot change the meaning of this already completed prefix.
            roll_ids = matched.group(1).split()
            if not roll_ids or len(set(roll_ids)) != len(roll_ids):
                return None
            if set(roll_ids) <= sh_ids:
                side_name = 'SHADOW'
            elif set(roll_ids) <= fp_ids:
                side_name = 'FREE_PEOPLES'
            else:
                return None
            try:
                values = [int(v) for v in matched.group(2).split()]
                if len(values) != len(roll_ids):
                    return None
                if side_name == 'SHADOW' and all(v in (*range(6), *range(10, 16)) for v in values):
                    rolled_faces = ['MHACPE'[v % 10] for v in values]
                elif side_name == 'FREE_PEOPLES' and all(v in range(5) for v in values):
                    rolled_faces = ['WCPMH'[v] for v in values]
                else:
                    return None
            except ValueError:
                return None
            raw_rolls.append((owner, rolled_faces))
        rendered_rolls = [(r.get('player'), r.get('faces')) for r in tail if r.get('op') == 'roll_dice']
        if raw_rolls != rendered_rolls:
            return None
        b['_rx_pre_roll_allocation'] = {'stage_line': stage['line'], 'box_line': returned['line'], 'pid': ids[0]}
    else:
        b['_rx_pre_roll_return'] = {'stage_line': stage['line'], 'return_line': returned['line'], 'pid': ids[0]}
    return si, si + 1, []


def _fold_staged_eye_allocations(g, rendered_path=None, raw_path=None):
    """Re-home a proved pre-roll allocation without inventing a paid action."""
    import rendered_replay as rr
    batches = g.get('batches') or []
    folds = 0
    for i in range(len(batches) - 2, -1, -1):
        h, n = batches[i:i + 2]
        if h.get('turn') != n.get('turn'):
            continue
        hit = _pre_roll_return_hesitation(g, h, rendered_path, raw_path, allocation=True)
        if hit is None:
            continue
        proof = dict(h['_rx_pre_roll_allocation'])
        kept = [dict(r) for r in h['records']]
        for record in kept:
            if record.get('line') == proof['stage_line']:
                record.pop('die_stage', None)
        merged = dict(n)
        merged['records'] = kept + list(n['records'])
        merged['rng'] = list(h.get('rng') or []) + list(n.get('rng') or [])
        merged['truth_before'] = h['truth_before']
        merged['_chat_lo'] = min(int(h.get('_chat_lo') or h['start_line']), int(n.get('_chat_lo') or n['start_line']))
        merged['_rx_pre_roll_allocations'] = [proof] + list(n.get('_rx_pre_roll_allocations') or [])
        merged['kind'] = rr.classify_batch(merged)
        batches[i + 1] = merged
        del batches[i]
        folds += 1
    return folds


def _fold_die_restage_hesitations(g, rendered_path=None, raw_path=None):
    """W2-031 (2026-09-03): a die STAGED and RETURNED to its pool with no
    roll, draw or owner gesture is not an action - the human hesitated
    and re-staged (3-6 lines later) or passed.  parse_game opens a
    batch at every die-stage record, so the hesitation was a batch of
    its own: _emit_passes counted it as a turn-order slot AND a spent
    die (a duplicate pass at the re-stage line when the dice gate let
    it through - 2023-03-24 L759/L761, 2024-05-21 L374/L380 - one die
    too few on every later pass row of the turn, and a side-CHANGING
    hesitation followed by an explicit 'pass' chat hid the real pass:
    2023-12-25 L117-L120, L460-L463; 2023-01-20 L592-L595), and an
    opponent correction made inside the window classified as the
    hesitating side's action (2023-03-06 L1269: FP W staged, Shadow
    withdrew a Regular into the Minas Tirith stronghold, FP returned
    the die; the withdrawal was reverted at L1275 -> the extractor
    minted an FP army_attack + a Shadow retreat_into_siege that never
    happened).  38 such batches in the 24 card-cover games, 30 pure.

    Fold each hesitation batch H into the NEXT batch N of the same turn
    in place on g['batches']: N keeps its die, its stage line (the true
    action line - the remaining-dice zone replay snaps there) and its
    after-snapshot; it inherits H's pre-stage turn-setup records and
    H's before-snapshot, and its chat window opens at H's line.  Between
    records (opponent moves during the hesitation) fold only when each
    is exactly reverted inside N - both records are dropped as the
    null pair they are; an unreverted opponent correction (2023-12-25
    L881) and any owner gesture (a card parked and taken back, an army
    moved and moved back: 2023-01-24 L144, 2023-03-24 L481) leave the
    batch alone.  A hesitation with no following batch in its turn is
    left alone too.  Returns the number of folds."""
    import rendered_replay as rr
    bs = g.get('batches') or []
    folds = _fold_staged_eye_allocations(g, rendered_path, raw_path)
    i = 0
    while i + 1 < len(bs):
        h, n = bs[i], bs[i + 1]
        hit = _pre_roll_return_hesitation(g, h, rendered_path, raw_path)
        if hit is None:
            hit = _stage_return_hesitation(h)
        if hit is None or n.get('turn') != h.get('turn'):
            i += 1
            continue
        si, ri, between = hit
        n_recs = list(n.get('records') or [])
        drop_n = []
        ok = True
        for r in between:
            j = next((k for k, x in enumerate(n_recs)
                      if k not in drop_n and x.get('op') == 'move'
                      and x.get('obj') == r.get('obj')
                      and x.get('from') == r.get('to')
                      and x.get('to') == r.get('from')), None)
            if j is None:
                ok = False
                break
            drop_n.append(j)
        if not ok:
            i += 1
            continue
        merged = dict(n)
        if h.get('_rx_hunt_box_hesitation'):
            merged['_rx_hunt_box_hesitations'] = list(h.get('_rx_hunt_box_hesitations') or []) + [h['_rx_hunt_box_hesitation']]
        if h.get('_rx_pre_roll_return') or h.get('_rx_pre_roll_returns'):
            merged['_rx_pre_roll_returns'] = (list(h.get('_rx_pre_roll_returns') or [])
                + ([h['_rx_pre_roll_return']] if h.get('_rx_pre_roll_return') else [])
                + list(n.get('_rx_pre_roll_returns') or []))
        merged['records'] = (list(h['records'][:si]) + list(h['records'][ri + 1:])
                             + [x for k, x in enumerate(n_recs)
                                if k not in drop_n])
        merged['rng'] = list(h.get('rng') or []) + list(n.get('rng') or [])
        merged['truth_before'] = h.get('truth_before')
        merged['_chat_lo'] = min(int(h.get('_chat_lo') or h['start_line']),
                                 int(n.get('_chat_lo') or n['start_line']))
        merged['kind'] = rr.classify_batch(merged)
        bs[i + 1] = merged
        del bs[i]
        folds += 1
        # the merged batch may itself be a hesitation (2024-05-21
        # L1278 -> L1280 -> L1283 -> L1288): re-test it at index i
    for k, b in enumerate(bs):
        b['idx'] = k
    return folds


def _validate_unpaid_phase_tails(batches):
    """Unpaid EOF containers are evidenced setup, never an inferred action."""
    import rendered_replay as rr
    for batch in batches:
        if not batch.get('_phase_only'):
            continue
        if not any(r.get('op') == 'roll_dice' for r in batch['records']):
            # A draw-only unfinished setup cannot prove a silent allocation
            # or declaration decline in the generic per-turn emitters.
            raise _Quarantine('coverage_unpaid_phase_tail_unproven',
                              {'detail': 'phase_closure_missing', 'line': batch['start_line']})
        for record in batch['records']:
            op = record.get('op')
            allowed = op in ('roll_dice', 'draw', 'alloc_eye', 'add_eye',
                             'pass', 'view_hand')
            if op == 'draw':
                allowed = rr._draw_kind(str(record.get('what', '')))[0] == 'deck'
            if op == 'other':
                allowed = bool(re.fullmatch(r'\S+ views the (?:Free Peoples|Shadow) event hand\.',
                                           str(record.get('raw', ''))))
            if op in ('move', 'kill'):
                tokens = [record.get('obj', '')] if op == 'move' else record.get('pieces', [])
                allowed = bool(tokens)
                for token in tokens:
                    cargo = str(token)
                    residual = _DIE_TOKEN_RE.sub('', cargo)
                    residual = re.sub(r'\bDwarven Ring\b', '', residual, flags=re.I)
                    if residual.replace('group', '').strip(' (),'):
                        allowed = False
                if record.get('die_stage') and not record.get('_rx_dwarven_selection'):
                    allowed = False
            if not allowed:
                raise _Quarantine('coverage_unpaid_phase_tail_unproven',
                                  {'detail': 'unsupported_phase_record', 'line': record.get('line'), 'op': op})


def _validate_unpaid_eye_transport(batches, raw):
    """Pair terminal Eye gestures at their roll prefix, without a future return."""
    import log_census as lc
    tails = {int(b['turn']): b for b in batches if b.get('_phase_only')}
    if not tails:
        return
    expected = []
    turn, rolls, known = 1, 0, {}
    rolled_shadow = set()
    shadow = lc.BASE_SH_DIE_PIDS | lc.KNOWN_BONUS_SH
    for line in raw:
        marker = re.search(r'<~Game\.7~>(\d+)\b', line)
        if marker:
            turn = int(marker.group(1))
            rolled_shadow = set()
        roll = lc.ROLL_RE.search(line)
        if roll:
            rolls += 1
            ids, nums = roll.group(1).split(), roll.group(2).split()
            for pid, num in zip(ids, nums):
                if pid in shadow:
                    known[pid] = 'MHACPE'[int(num) % 10]
                    rolled_shadow.add(pid)
        change = lc.CHANGE_RE.search(line)
        if change and change.group(1) in shadow:
            known[change.group(1)] = change.group(2)
        actor = line.split('<~', 1)[0].lstrip('$')
        event = None
        eye = lc.EYE_BTN_RE.search(line)
        if eye:
            known[eye.group(1)] = 'E'
            event = ('allocate', actor, 1)
        move = lc.MOVE_RE.search(line)
        if move:
            ids, origin, target = move.group(1).split(), move.group(2), move.group(3)
            if ids and set(ids) <= shadow:
                if turn in tails and (origin == 'Game.1997' or target == 'Game.1997') and len(ids) != len(set(ids)):
                    raise _Quarantine('coverage_unpaid_phase_tail_unproven',
                                      {'detail': 'duplicate_physical_eye'})
                if (origin, target) == ('Game.2043', 'Game.1997'):
                    known.update({pid: 'E' for pid in ids})
                    event = ('move_in', actor, len(ids))
                elif (origin, target) == ('Game.1997', 'Game.2043'):
                    event = ('return', actor, len(ids))
                elif target == 'Game.1997':
                    event = ('unsupported_staged_eye', actor, len(ids))
        added = re.search(r'<~Controls\.350~>([\d ]+)', line)
        if added:
            ids = added.group(1).split()
            if turn in tails and (not ids or len(ids) != len(set(ids))
                                  or not set(ids) <= rolled_shadow or any(pid not in known for pid in ids)):
                raise _Quarantine('coverage_unpaid_phase_tail_unproven',
                                  {'detail': 'added_eyes_physical_identity'})
            count = sum(known.get(pid) == 'E' for pid in ids)
            event = ('add', actor, count)
        if turn in tails and event:
            expected.append((turn, rolls, *event))
    rendered_rolls = [r for b in batches for r in b['records'] if r.get('op') in ('roll_dice', 'allocated_eye_display')]
    witnessed = []
    for turn, batch in tails.items():
        for rec in batch['records']:
            event = None
            if rec.get('op') == 'alloc_eye':
                event = ('allocate', rec.get('player'), 1)
            elif rec.get('op') == 'add_eye':
                event = ('add', rec.get('player'), int(rec['n']))
            elif rec.get('op') == 'move':
                direction = (rec.get('from'), rec.get('to'))
                if direction in (('SA Dice', 'The Hunt For The Ring'),
                                 ('The Hunt For The Ring', 'SA Dice')):
                    cargo = str(rec.get('obj', ''))
                    tokens = _die_tokens(cargo)
                    residual = _DIE_TOKEN_RE.sub('', cargo).replace('group', '').strip(' (),')
                    if residual or not tokens or any(s != 'SHADOW' or face != 'E' for s, face in tokens):
                        raise _Quarantine('coverage_unpaid_phase_tail_unproven',
                                          {'detail': 'eye_cargo_unproven', 'line': rec['line']})
                    event = ('move_in' if direction[0] == 'SA Dice' else 'return', rec.get('player'), len(tokens))
                elif direction[1] == 'The Hunt For The Ring':
                    event = ('unsupported_rendered_eye', rec.get('player'), 0)
            if event:
                before = sum(int(r['line']) < int(rec['line']) for r in rendered_rolls)
                witnessed.append((turn, before, *event))
    if expected != witnessed:
        raise _Quarantine('coverage_unpaid_phase_tail_unproven',
                          {'detail': 'eye_transport_order_count_or_actor'})


def extract_one(rendered_path, stem, meta):
    """Run all gates + emit samples for one rendered game.  Returns
    (samples, line).  Raises _Quarantine for a gate failure."""
    import rendered_replay as rr
    import random
    global _LEDGER, _CHAT_INDEX, _TABLE, _FP_HUNTBOX_IDX, _FP_HUNTBOX_FACES
    global _RX_HUNT_HISTORY, _REACTIVE_TABLE
    _REACTIVE_TABLE = None
    _RX_HUNT_HISTORY = None
    _LEDGER = None
    _CHAT_INDEX = None
    _TABLE = None
    _FP_HUNTBOX_IDX = []
    _FP_HUNTBOX_FACES = []
    # Deterministic RNG per game: the certified hunt/combat drivers FORCE the
    # rendered dice locally (monkeypatched random.randint/choice) and fall back
    # to the global generator only on stream exhaustion; seeding that fallback
    # makes the whole extraction reproducible (gate verdicts + forced-dice
    # resync are unaffected).
    random.seed(0)
    t0 = time.time()
    try:
        g = rr.parse_game(rendered_path)
    except (rr.PhysicalRecruitIdentityError, rr.ClientUndoHistoryError, RawLogEncodingError) as error:
        raise _Quarantine(error.reason, error.detail) from error
    # W2-031: stage-and-return die hesitations are not actions - fold
    # each into its turn's next batch BEFORE anything reads the batches
    _normalize_physical_combat_corrections(g['batches'], rendered_path)
    _normalize_reversed_hunt_parking(g['batches'], rendered_path)
    _n_folds9 = _fold_die_restage_hesitations(g, rendered_path)
    _n_folds9 += _fold_cancelled_preroll_move(g, rendered_path)
    batches = g['batches']
    if _LIST2:
        from rendered_reactive_cards import Timeline
        _REACTIVE_TABLE = Timeline(g)
    _normalize_repeated_ring_moves(batches, rendered_path)
    _validate_unpaid_phase_tails(batches)
    _annotate_legacy_bid_ring_markers(batches, rendered_path)
    _annotate_dwarven_physical_rolls(batches, rendered_path)
    from rendered_hunt_die_transport import mark_repeated_deposits, mark_shadow_roundtrips, mark_lidless_original_faces, mark_restored_shadow_allocation
    mark_repeated_deposits(batches, rendered_path, _raw_log_path(stem))
    mark_shadow_roundtrips(batches, rendered_path, _raw_log_path(stem))
    mark_lidless_original_faces(batches, rendered_path, _raw_log_path(stem))
    mark_restored_shadow_allocation(batches, rendered_path, _raw_log_path(stem))
    _n_folds9 += _fold_dwarven_selection_batches(batches)
    # DET census P2-b: a Mordor-track Eye tile's Hunt damage is the Hunt
    # Box die count; without batch['_mordor_box'] the drive returns
    # 'hunt:special(eye)' and every in-log FP decision in those batches
    # is discarded (8,327 corpus-wide).  Import-only use of the census's
    # verified box simulator (29/29 Mordor-eye forensic 2026-07-25).
    import rendered_census as _rc9m
    _rc9m._prescan_mordor_box(g)
    # per-turn FP hunt-box die placements (each earlier fellowship move
    # this turn = +1 hunt modifier, p.40); the per-batch recon loses
    # them (GAME02 T2: 3rd move's roll 4+2 hit; the engine scored a
    # miss and silently skipped the whole hunt resolution)
    if _LIST2:
        try:
            _CHAT_INDEX = _build_chat_index(rendered_path)
        except Exception:
            _CHAT_INDEX = None
        try:
            _TABLE = _build_table_ledger(rendered_path)
        except Exception:
            _TABLE = None

    # ---- Gate 1: structure -------------------------------------------
    n_roll = sum(1 for b in batches for it in b['rng']
                 if it.get('kind') in ('action_roll', 'roll_num'))
    if not batches or g['final_turn'] < 1 or n_roll == 0:
        raise _Quarantine('gate1_no_structure',
                          {'batches': len(batches),
                           'turns': g['final_turn'], 'rolls': n_roll})

    # ---- Gate 2: no unparsed decision-bearing lines ------------------
    ok, g2reason, g2cat = _scan_lines_gate2(rendered_path)
    if not ok:
        raise _Quarantine(g2reason, dict(g2cat))

    # ---- Gate 3: draw-pairing vs raw log -----------------------------
    ok3, g3note = _gate3_draws(rendered_path, stem)
    if not ok3:
        raise _Quarantine(g3note.split(':')[0], g3note)

    # ---- Hand-state ledger (v27 event-card-play enablement) ----------
    # Gate 3 guarantees the rendered/raw draw streams are ORDER-exact, so the
    # raw password-decoded per-deck FIFO supplies each rendered draw's true
    # identity.  A hard desync (a NAMED play removing a card not in hand, or a
    # draw-FIFO underflow) quarantines with gate3_hand_desync; passwords absent
    # / raw undecodable -> empty-hand fallback (board samples still emit).
    _LEDGER = _build_hand_ledger(rendered_path, stem)
    if _LEDGER is not None and _LEDGER.desync is not None:
        raise _Quarantine(_LEDGER.desync[0], _LEDGER.desync[1])

    # ---- Gate 6: board invariants every batch ------------------------
    for b in batches:
        for snap_key in ('truth_before', 'truth'):
            snap = b.get(snap_key)
            if snap is None:
                continue
            ok6, detail = _conservation_ok(snap)
            if not ok6:
                raise _Quarantine('gate6_invariant',
                                  {'batch': b['idx'], 'detail': detail})

    # ---- Gates 4/5/8: drive + legality-matched sample emission -------
    stats = {'kinds_emit': Counter(), 'drop_reasons': Counter(), 'drove': 0,
             'diverged': 0, 'nomatch': 0, 'ambiguous': 0,
             'played_cids': set(), 'detected': Counter(),
             # 1.149 provenance: '<victim cid>|<paid cid>+<paid cid>'
             # per extracted table kill (USER 2026-08-20)
             'tablekill_paid': Counter()}
    global _RX_STATS
    _RX_STATS = stats
    from rendered_hunt_observation import PublicHuntHistory
    _RX_HUNT_HISTORY = PublicHuntHistory(batches, _reveal_public_pool)
    if _n_folds9:
        stats['detected']['die_restage_fold'] += _n_folds9
    samples = []
    declare_cov = []      # v32 gate-(e) per-turn coverage records
    # v27 List-2 pre-computation (game-level): per-turn Shadow eye allocation
    # + per-(turn,side) rolled dice pool for elven-ring menu reconstruction.
    try:
        eye_counts = _turn_eye_counts(rendered_path, batches)
    except Exception:
        eye_counts = {}
    try:
        pools = _roll_pools(batches)
    except Exception:
        pools = {}
    # X-audit B1 (2026-08-31): eyes_allocated was NEVER reconstructed -
    # X3192 sat at 0 corpus-wide and staged hunt-box dice all read as
    # "rolled".  Publish per-turn allocation + rolled-Eye counts for the
    # _emit_sample encoding-only backstop (P0-a doctrine: menus are
    # enumerated before the stamp, and the eye_allocate decision's own
    # pre-roll state is exempt - stamping it would leak the label).
    global _RX_EYEALLOC, _RX_EYEROLL
    _RX_EYEALLOC = dict(eye_counts)
    _RX_EYEROLL = {}
    for (t9, sd9), faces9 in pools.items():
        n9 = sum(1 for f9 in faces9 if str(f9).upper().startswith('E'))
        if n9:
            _RX_EYEROLL[t9] = max(_RX_EYEROLL.get(t9, 0), n9)
    # v31 dice-context + which-die marginalization annotation
    try:
        _annotate_remaining_pools(batches, pools, eye_counts)
    except _Quarantine:
        raise
    except Exception:
        pass
    # Hunt restaging proof depends on these actual physical-prefix pools.
    _annotate_delayed_fellowship_ring(batches)
    from rendered_hunt_die_transport import annotate_direct_fellowship_ring, annotate_refaced_fellowship_ring
    annotate_direct_fellowship_ring(batches, rendered_path, _raw_log_path(stem))
    annotate_refaced_fellowship_ring(batches, rendered_path, _raw_log_path(stem))
    # elven-ring inventory ledger (D1/D2 fix): stamps batches + snapshots
    try:
        ring_audit = _annotate_ring_ledger(batches)
    except Exception:
        ring_audit = None
    # W3-005: pre-staging ring uses re-homed to their owner's batch
    try:
        _n_carry9 = _annotate_ring_carry(batches)
        if _n_carry9 and isinstance(ring_audit, dict):
            ring_audit['carried'] = _n_carry9
    except Exception:
        pass
    # W1-025: FP hero flags (gandalf_the_white.in_play /
    # aragorn_crowned.crowned) stamped into both truth snapshots
    try:
        _annotate_hero_flags(batches)
    except Exception:
        pass
    # W1-011: FP bid-token holdings (political / card) + once-per-turn
    # latch stamped into both truth snapshots
    try:
        token_audit = _annotate_token_ledger(batches)
    except Exception:
        token_audit = None
    _annotate_fellowship_dice(batches)
    for b in batches:
        source_batch = b
        from rendered_mixed_hunt_combat import project_hunt_prefix
        mixed_hunt = project_hunt_prefix(b)
        if mixed_hunt is not None:
            stats['drop_reasons'][mixed_hunt['reason']] += 1
            if mixed_hunt['batch'] is None:
                continue
            b = mixed_hunt['batch']
            stats['detected']['mixed_hunt_completed_prefix'] += 1
        global _RX_RINGS_CUR, _RX_LINE_CUR
        _RX_RINGS_CUR = (b.get('_rx_rings') or {}).get('post')
        _RX_LINE_CUR = b.get('start_line') or 0
        _RX_HUNT_HISTORY.current_batch = b
        # W4-033: per-batch baselines for the move_comp_unmatched re-bucket
        # at the tail of this iteration (meter only; no row is touched)
        _n_smp0 = len(samples)
        _mcu0 = stats['drop_reasons'].get('move_comp_unmatched', 0)
        _afd0 = _attack_family_drops(stats)
        try:
            s = process_batch(b, stats)
        except _Quarantine:
            raise
        except Exception as e:                 # never let one batch kill game
            stats['drop_reasons']['batch_exc:%s' % str(e)[:20]] += 1
            s = None
        if s is not None:
            # onslaught wiring may return [target_sample, magnitude_sample]
            if isinstance(s, list):
                samples.extend(x for x in s if x is not None)
            else:
                samples.append(s)
        _xs = stats.pop('__extra_samples', None)
        if _xs:
            samples.extend(x for x in _xs if x is not None)
        # ADDITIVE List-2 extras â€” never touch the primary sample or any gate,
        # each is its own reconstruction and drops on no-match (purity).
        if _LIST2:
            try:
                _catalog_unemitted(b, stats)
                samples.extend(_emit_shadow_hunt_ring(b, stats))
                if not ((b.get('_rx_shadow_hunt_effect') or {}).get('kind') == 'elven_ring_eye'
                        and b.get('die_side') == 'SHADOW'):
                    samples.extend(_emit_elven_rings(b, pools, stats))
                samples.extend(_emit_token_uses(b, stats))
                samples.extend(_emit_card_interiors(b, stats))
                samples.extend(_emit_combat_family(b, stats))
                samples.extend(_emit_siege_withdrawal(b, stats))
                samples.extend(_emit_dwarven_rings(b, pools, stats))
                samples.extend(_emit_hunt_resolution(b, stats))
                from rendered_reactive_cards import emit_crebain
                samples.extend(emit_crebain(b, stats, _REACTIVE_TABLE))
                if _CLOSURE:
                    samples.extend(_emit_hunt_cards(b, stats))
            except _Quarantine:
                raise
            except Exception as e:
                stats['drop_reasons']['extra_exc:%s' % str(e)[:16]] += 1
        if b.get('_rx_hunt_exclusion'):
            source_batch['_rx_hunt_exclusion'] = dict(b['_rx_hunt_exclusion'])
            for key in ('_rx_huntres', '_rx_reveal_predecision', '_rx_hunt_guide_predecision', '_rx_hunt_reduction_owned'):
                source_batch.pop(key, None)
        elif b is not source_batch and b.get('_rx_hunt_guide_predecision'):
            source_batch['_rx_hunt_guide_predecision'] = b['_rx_hunt_guide_predecision']
        # W4-033 (P meter): `move_comp_unmatched` ticks in _emit_bookkeeping
        # BEFORE the List-2 extras see the batch; when they emit the
        # batch's army_attack row (or count their own attack-family drop)
        # the tick is re-bucketed - see _rebucket_mcu.  Meters only.
        _rebucket_mcu(stats, samples[_n_smp0:], _mcu0, _afd0)
        # v31: drop the per-batch marginal-match context so game-level
        # emitters (declares, eyes, passes) never face-filter against a
        # stale batch state
        global _MM_FACE, _MM_SIDE, _MM_STATE
        _MM_FACE = _MM_SIDE = _MM_STATE = None
        _RX_RINGS_CUR = None
        _RX_LINE_CUR = None
        _RX_HUNT_HISTORY.current_batch = None
    # eye allocation is a per-TURN Shadow decision (pre-roll), emitted once per
    # turn from the turn's first-batch reconstruction (incl. the 0-eye case).
    if _LIST2:
        try:
            samples.extend(_emit_eye_allocations(batches, eye_counts, stats))
        except Exception as e:
            stats['drop_reasons']['eye_exc:%s' % str(e)[:16]] += 1
        try:
            samples.extend(_emit_passes(batches, pools, stats, rendered_path))
        except Exception as e:
            stats['drop_reasons']['pass_exc:%s' % str(e)[:16]] += 1
        try:
            samples.extend(_emit_discards(batches, stats))
        except Exception as e:
            stats['drop_reasons']['discard_exc:%s' % str(e)[:16]] += 1
        try:
            samples.extend(_emit_guides(batches, stats))
        except Exception as e:
            stats['drop_reasons']['guide_exc:%s' % str(e)[:16]] += 1
        if _CLOSURE:
            try:
                _s9decl, declare_cov = _emit_declarations(batches, stats)
                samples.extend(_s9decl)
                samples.extend(_emit_reveal_placements(batches, stats))
                samples.extend(_emit_cardhunt_arms(batches, stats))
            except _Quarantine:
                raise
            except Exception as e:
                stats['drop_reasons']['declare_exc:%s' % str(e)[:16]] += 1

    # Gate 5: engine resync rate < 1% per game (rendered_census definition:
    # gate-counting divergences / TOTAL batches).  Each batch is independently
    # resync'd to the client-authoritative mirror, so an isolated divergent
    # batch NEVER contaminates the rest â€” its own sample is already dropped
    # above (sample hygiene); the per-game rate gate quarantines only a
    # SYSTEMATICALLY divergent game (a broken parse / pervasive engine gap).
    driven = stats['drove']
    total_b = len(batches)
    resync_rate = (stats['diverged'] / total_b) if total_b else 0.0
    if total_b >= 30 and resync_rate >= 0.01:
        raise _Quarantine('gate5_resync',
                          {'diverged': stats['diverged'], 'driven': driven,
                           'total_batches': total_b,
                           'rate': round(resync_rate, 4)})

    # ---- Gate 7: z (CSV authoritative + board cross-check) ------------
    final_snap = batches[-1]['truth']
    z_tag, z_source, z_conflict = _z_from_csv_board(final_snap, meta)
    for s in samples:
        s['z'] = 1 if s['side'] == z_tag else -1

    # Ring holding-window truth guard (USER ruling 2026-08-09): no
    # sample may offer a ring option the acting side cannot hold â€”
    # drops misstated menus, reports the inverse gap, adds nothing
    # HUNT-A F6: guard failure must be LOUD â€” swallowing it ships an
    # UN-guarded game (the zero-hold truth constraint silently void),
    # the exact silent-fallback class the phi emitter already killed
    pool_kept = []
    pool_refusals = []
    for sample in samples:
        refusal = sample.get('hunt_pool_refusal')
        if refusal:
            stats['drop_reasons']['hunt_pool:' + refusal] += 1
            pool_refusals.append({'phase': sample.get('phase'),
                                 'turn': sample.get('turn'), 'reason': refusal,
                                 **(sample.get('hunt_pool_provenance') or {})})
        else:
            pool_kept.append(sample)
    samples = _filter_ai_policy_samples(pool_kept, stats)
    samples = _ring_truth_guard(samples, stats)
    stats['stored_phases'] = Counter(str(s.get('phase', '')) for s in samples)

    line = {
        'game': stem, 'kept': len(samples),
        'drop_counts_complete': True,
        'batches': len(batches), 'turns': g['final_turn'],
        'driven': driven, 'diverged': stats['diverged'],
        'resync_rate': round(resync_rate, 4),
        'nomatch': stats['nomatch'], 'ambiguous': stats['ambiguous'],
        'emit_by_kind': dict(stats['kinds_emit']),
        'stored_by_phase': dict(stats['stored_phases']),
        'detected': dict(stats['detected']),
        # Keep rare failures: truncating each game's counters hides systematic
        # losses when manifests are aggregated across the corpus.
        'drop_by_reason': dict(stats['drop_reasons']),
        # every 1.149 removal this game, with WHICH cards paid for it
        # (USER 2026-08-20) - full list, not a top-N: these are rare
        # and each one is provenance for a training label
        'tablekill_paid': dict(stats['tablekill_paid']),
        # EMISSION COMPLETENESS (DET census 2026-08-30): pendings the
        # replay OPENED with a real menu, vs emitted / ruled-automatic
        'pend_table': {
            'open': dict(stats.get('pend_open') or {}),
            'emit': dict(stats.get('pend_emit') or {}),
            'auto': dict(stats.get('pend_auto') or {})},
        'g2_catalog': dict(g2cat), 'g3': g3note,
        'z_tag': z_tag, 'z_source': z_source, 'z_conflict': z_conflict,
        'reported': '%s %s T%s' % (meta.get('winner_side'),
                                   meta.get('victory'), meta.get('turns')),
        'ring_ledger': ring_audit,
        'token_ledger': token_audit,
        'hunt_pool_refusals': pool_refusals,
        'secs': round(time.time() - t0, 1), 'error': None,
    }
    led = _LEDGER
    if led is not None:
        amb_decks = sum(1 for sd in ('fp', 'sh') for dk in ('C', 'S')
                        if led.amb[sd][dk] > 0)
        line['hand_ledger'] = {
            'draws': led.n_draws, 'named_out': led.n_named_out,
            'named_in': led.n_named_in, 'rand_disc': led.n_rand_disc,
            'rand_unresolved': led.n_rand_unresolved,
            'excess_disc': led.n_excess_disc, 'soft_desync': led.n_soft_desync,
            'cas_disc': led.n_cas_disc,
            'disposal_kills': led.n_disposal,       # W1-002
            'undone_kills': led.n_undone_kills,     # W1-002
            'undone_draws': led.n_undone_draws,     # W1-024
            'views': led.n_views, 'amb_decks': amb_decks,
            # W1-002: every ambiguity window ever opened, [open, close]
            # (close None = still open at game end)
            'amb_windows': {'%s_%s' % (sd, dk): [list(w) for w in ws]
                            for sd in ('fp', 'sh') for dk in ('C', 'S')
                            for ws in [led.amb_win[sd][dk]] if ws},
            'fp_final': len(led._fp), 'sh_final': len(led._sh)}
    else:
        line['hand_ledger'] = None
    # ---- gate-(e) per-turn declare coverage (v32, USER 2026-08-03) ----
    _cat_names = {v: k for k, v in _COV_CATS.items()}
    _cc9 = Counter(_cat_names[c['cat']] for c in declare_cov)
    _turns_seen = len({b['turn'] for b in batches})
    _inmordor9 = stats['detected'].get('declare_cov_inmordor', 0)
    line['declare_cov'] = {
        'eligible': len(declare_cov),
        'inmordor_turns': _inmordor9,
        # turns seen in batches that produced neither a coverage
        # record nor an in-Mordor exclusion (loop exception path)
        'missing_turns': max(0, _turns_seen - len(declare_cov)
                             - _inmordor9),
        # turns absent from the batch stream entirely (parse gaps)
        'gap_turns': max(0, int(g['final_turn'] or 0) - _turns_seen),
        'heal0_declared': stats['detected'].get('declare_heal0', 0),
        **dict(_cc9)}
    cov_extras = None
    line['local_decision_exclusions'] = [dict(b['_rx_hunt_exclusion'])
                                        for b in batches if b.get('_rx_hunt_exclusion')]
    if declare_cov:
        import ai as _ai9
        _D9 = _ai9.get_feature_dim()
        _cx = np.zeros((len(declare_cov), _D9), dtype=np.float16)
        _cok = np.zeros(len(declare_cov), dtype=np.uint8)
        for _i9, c in enumerate(declare_cov):
            if c['x'] is not None and len(c['x']) == _D9:
                _cx[_i9] = c['x']
                _cok[_i9] = 1
        cov_extras = {
            'declare_cov_x': _cx,
            'declare_cov_xok': _cok,
            'declare_cov_cat': np.asarray(
                [c['cat'] for c in declare_cov], dtype=np.uint8),
            'declare_cov_turn': np.asarray(
                [c['turn'] for c in declare_cov], dtype=np.int16),
            'declare_cov_reg': np.asarray(
                [c['reg'] for c in declare_cov], dtype=np.int32)}
    return samples, line, cov_extras


# ==========================================================================
# worker / driver / pilot  (mirrors extract_corpus)
# ==========================================================================
def _quarantine(stem, reason, detail):
    with open(os.path.join(OUT_DIR, 'quarantine.jsonl'), 'a',
              encoding='utf-8') as f:
        f.write(json.dumps({'game': stem, 'reason': reason,
                            'detail': detail}) + '\n')


def _game_iter(source):
    """Yield (rendered_path, stem, meta) for the requested source."""
    zmeta = _z_meta()
    if source == 'canary':
        for p in _canary_files():
            stem = _canary_stem(p)
            yield p, stem, zmeta.get(stem, {})
    else:
        for stem in _ok_stems():
            if stem not in zmeta:
                continue        # not a base game / missing passwords / curated
            yield (os.path.join(RLOG_CORPUS, stem + '.rendered.txt'),
                   stem, zmeta[stem])


def run_worker(shard, nshards, source='corpus', manifest_path=None,
               limit=None, game_list=None):
    import rendered_replay  # noqa: warm heavy imports once
    import rendered_census  # noqa
    import ai  # noqa
    os.makedirs(OUT_DIR, exist_ok=True)
    if game_list is not None:
        games = game_list
    else:
        games = list(_game_iter(source))[shard::nshards]
    if limit:
        games = games[:limit]
    mp = manifest_path or os.path.join(OUT_DIR,
                                       'manifest-shard%d.jsonl' % shard)
    n_done = 0
    for rendered_path, stem, meta in games:
        npz = os.path.join(OUT_DIR, stem + '.npz')
        err = os.path.join(OUT_DIR, stem + '.error.json')
        if os.path.exists(npz) or os.path.exists(err):
            continue
        try:
            samples, line, cov_extras = extract_one(rendered_path, stem,
                                                    meta)
            if samples:
                import extract_corpus as ec
                ec.pack_npz(samples, npz, extras=cov_extras)
            else:
                np.savez_compressed(npz, empty=np.zeros(1))
        except _Quarantine as q:
            _quarantine(stem, q.reason, q.detail)
            line = {'game': stem, 'kept': 0, 'quarantined': q.reason,
                    'error': None}
            np.savez_compressed(npz, empty=np.zeros(1))  # mark done
        except KeyboardInterrupt:
            raise
        except BaseException as e:
            tb = traceback.format_exc()
            line = {'game': stem, 'kept': 0,
                    'error': ('%s: %s' % (type(e).__name__, e))[:200]}
            with open(err, 'w', encoding='utf-8') as f:
                json.dump({'error': line['error'], 'traceback': tb[-1500:]},
                          f, indent=1)
        with open(mp, 'a', encoding='utf-8') as f:
            f.write(json.dumps(line) + '\n')
        n_done += 1
        if n_done % 25 == 0:
            gc.collect()
        print('[shard %d] %-46s kept=%s q=%s err=%s'
              % (shard, stem[:46], line.get('kept'),
                 line.get('quarantined'), line.get('error')), flush=True)
    print('[shard %d] DONE (%d games this run)' % (shard, n_done), flush=True)


def aggregate_manifests():
    agg = {'games': 0, 'kept': 0, 'quarantined': 0, 'errors': 0,
           'quarantine_by_reason': Counter(), 'emit_by_kind': Counter(),
           'detected': Counter(), 'declare_cov': Counter(),
           'pend_open': Counter(), 'pend_emit': Counter(),
           'pend_auto': Counter(),
           'drop_by_reason': Counter(), 'admitted': 0,
           'ledger_games': 0, 'ledger_draws': 0, 'ledger_named_out': 0,
           'ledger_rand_disc': 0, 'ledger_rand_unresolved': 0,
           'ledger_excess_disc': 0, 'ledger_soft_desync': 0,
           'ledger_views': 0, 'ledger_amb_decks': 0,
           'ring_games': 0, 'ring_fp_uses': 0, 'ring_sh_uses': 0,
           'ring_anom_fp': 0, 'ring_anom_sh': 0, 'ring_anom_turn_dup': 0}
    for fn in os.listdir(OUT_DIR):
        if not fn.startswith('manifest-') or not fn.endswith('.jsonl'):
            continue
        with open(os.path.join(OUT_DIR, fn), encoding='utf-8') as f:
            for ln in f:
                try:
                    d = json.loads(ln)
                except Exception:
                    continue
                agg['games'] += 1
                agg['kept'] += d.get('kept', 0) or 0
                if d.get('quarantined'):
                    agg['quarantined'] += 1
                    agg['quarantine_by_reason'][
                        str(d['quarantined']).split(':')[0].split('(')[0]] += 1
                elif d.get('error'):
                    agg['errors'] += 1
                else:
                    agg['admitted'] += 1
                for k, c in (d.get('emit_by_kind') or {}).items():
                    agg['emit_by_kind'][k] += c
                for k, c in (d.get('detected') or {}).items():
                    agg['detected'][k] += c
                for k, c in (d.get('drop_by_reason') or {}).items():
                    agg['drop_by_reason'][k] += c
                for k, c in (d.get('declare_cov') or {}).items():
                    agg['declare_cov'][k] += c
                pt = d.get('pend_table') or {}
                for kk in ('open', 'emit', 'auto'):
                    for k, c in (pt.get(kk) or {}).items():
                        agg['pend_' + kk][k] += c
                rl = d.get('ring_ledger')
                if rl:
                    agg['ring_games'] += 1
                    agg['ring_fp_uses'] += rl.get('fp_uses', 0)
                    agg['ring_sh_uses'] += rl.get('sh_uses', 0)
                    agg['ring_anom_fp'] += rl.get('anom_fp', 0)
                    agg['ring_anom_sh'] += rl.get('anom_sh', 0)
                    agg['ring_anom_turn_dup'] += rl.get('anom_turn_dup', 0)
                hl = d.get('hand_ledger')
                if hl:
                    agg['ledger_games'] += 1
                    agg['ledger_draws'] += hl.get('draws', 0)
                    agg['ledger_named_out'] += hl.get('named_out', 0)
                    agg['ledger_rand_disc'] += hl.get('rand_disc', 0)
                    agg['ledger_rand_unresolved'] += hl.get(
                        'rand_unresolved', 0)
                    agg['ledger_excess_disc'] += hl.get('excess_disc', 0)
                    agg['ledger_soft_desync'] += hl.get('soft_desync', 0)
                    agg['ledger_views'] += hl.get('views', 0)
                    agg['ledger_amb_decks'] += hl.get('amb_decks', 0)
    return agg


def run_pilot(n, source='corpus'):
    games = list(_game_iter(source))[:n]
    print('pilot: %d games (source=%s)' % (len(games), source), flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    run_worker(0, 1, source=source,
               manifest_path=os.path.join(OUT_DIR, 'manifest-pilot.jsonl'),
               game_list=games)
    dt = time.time() - t0
    print('\npilot wall %.0fs (%.1fs/game)' % (dt, dt / max(1, len(games))))
    _print_agg()


def run_canary():
    games = list(_game_iter('canary'))
    print('canary: %d games' % len(games), flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    run_worker(0, 1, source='canary',
               manifest_path=os.path.join(OUT_DIR, 'manifest-canary.jsonl'),
               game_list=games)
    _print_agg()


def _print_agg():
    agg = aggregate_manifests()
    print('\n=== aggregate ===')
    print('games=%d admitted=%d quarantined=%d errors=%d kept_samples=%d'
          % (agg['games'], agg['admitted'], agg['quarantined'],
             agg['errors'], agg['kept']))
    print('quarantine_by_reason: %s' % dict(agg['quarantine_by_reason']))
    print('emit_by_kind: %s' % dict(agg['emit_by_kind']))
    print('detected(list2 census): %s' % dict(agg['detected']))
    _new2 = ('eye_allocate', 'draw_card', 'elven_ring')
    print('v27-new emitted: %s'
          % {k: agg['emit_by_kind'].get(k, 0) for k in _new2})
    _closure = ('hunt_reduction', 'hunt_reduction_use',
                'hunt_reduction_decline', 'wizardstaff', 'wizardstaff_prevent',
                'wizardstaff_decline', 'mithril', 'mithril_redraw',
                'mithril_keep', 'fellowship_declare', 'declare_placement',
                'combat_effect')
    print('CLOSURE emitted: %s'
          % {k: agg['emit_by_kind'].get(k, 0) for k in _closure})
    print('CLOSURE census(detected): %s'
          % {k: agg['detected'].get(k, 0) for k in
             ('huntcard_axebow_horn', 'huntcard_wizardstaff',
              'huntcard_mithril', 'declare_progress_reset',
              'cc_onslaught_backcalc', 'cc_effect_choice')})
    if agg['admitted']:
        print('samples/admitted-game: %.1f' % (agg['kept'] / agg['admitted']))
    print('--- hand ledger ---')
    print('ledger_games=%d draws=%d named_out=%d rand_disc=%d '
          'rand_unresolved=%d excess_disc=%d soft_desync=%d views=%d '
          'amb_decks=%d'
          % (agg['ledger_games'], agg['ledger_draws'], agg['ledger_named_out'],
             agg['ledger_rand_disc'], agg['ledger_rand_unresolved'],
             agg['ledger_excess_disc'], agg['ledger_soft_desync'],
             agg['ledger_views'], agg['ledger_amb_decks']))
    _pe = agg['emit_by_kind'].get('play_event', 0)
    print('play_event emitted=%d' % _pe)
    # ---- gate-(e) per-turn declare coverage ---------------------------
    dc = agg['declare_cov']
    _el9 = dc.get('eligible', 0)
    _cov9 = (dc.get('declared', 0) + dc.get('declined', 0)
             + dc.get('revealed', 0) + dc.get('declined_nowindow', 0))
    print('--- declare coverage (gate e) ---')
    print('eligible_turns=%d covered=%d (%.2f%%) declared=%d '
          '(heal0 %d) declined=%d declined_nowindow=%d revealed=%d '
          'ambiguous=%d missing=%d gap_turns=%d inmordor_turns=%d'
          % (_el9, _cov9, 100.0 * _cov9 / max(1, _el9),
             dc.get('declared', 0), dc.get('heal0_declared', 0),
             dc.get('declined', 0), dc.get('declined_nowindow', 0),
             dc.get('revealed', 0), dc.get('ambiguous', 0),
             dc.get('missing_turns', 0), dc.get('gap_turns', 0),
             dc.get('inmordor_turns', 0)))
    # ---- ring-inventory ledger audit ---------------------------------
    _rg = max(1, agg['ring_games'])
    print('--- ring ledger audit ---')
    print('games=%d fp_uses=%d (%.4f/game) sh_uses=%d (%.4f/game) '
          'anom_fp=%d anom_sh=%d anom_turn_dup=%d ledger_clamp=%d'
          % (agg['ring_games'], agg['ring_fp_uses'],
             agg['ring_fp_uses'] / _rg, agg['ring_sh_uses'],
             agg['ring_sh_uses'] / _rg, agg['ring_anom_fp'],
             agg['ring_anom_sh'], agg['ring_anom_turn_dup'],
             agg['detected'].get('ring_ledger_clamp', 0)))
    top_drops = agg['drop_by_reason'].most_common(12)
    print('top drop reasons: %s' % dict(top_drops))
    # ---- EMISSION COMPLETENESS (DET census 2026-08-30) ---------------
    # every pending the replay OPENED with a real (>=2-option) menu, per
    # family: opened / emitted / ruled-automatic / dropped.
    fams = sorted(set(agg['pend_open']) | set(agg['pend_emit'])
                  | set(agg['pend_auto']))
    if fams:
        print('--- emission completeness (pendings opened during '
              'replay) ---')
        print('%-26s %9s %9s %9s %9s' % ('family', 'opened', 'emitted',
                                         'auto', 'dropped'))
        for f in fams:
            o = agg['pend_open'].get(f, 0)
            e = agg['pend_emit'].get(f, 0)
            au = agg['pend_auto'].get(f, 0)
            print('%-26s %9d %9d %9d %9d'
                  % (f, o, e, au, max(0, o - e - au)))


def run_driver(workers, source='corpus'):
    total = len(list(_game_iter(source)))
    os.makedirs(OUT_DIR, exist_ok=True)
    # phi provenance (SEARCH countersign cond. i): the profile pair,
    # rho and PROFILE_WEIGHT every sample's phi was computed under ride
    # the corpus itself - a wrong-profile phi at training time is
    # silent currency corruption, and rho is the ONE env knob that can
    # silently skew it (shard workers inherit this env; ai reads it at
    # import).  Consumer: train_corpus.consolidate() hard-fails on any
    # mismatch (HUNT-A F7).
    import json as _json9
    import ai as _ai9
    _rho9 = float(os.environ.get('WOTR_PHI_RINGS_RHO', '0') or 0)
    with open(os.path.join(OUT_DIR, 'phi_profiles.json'), 'w',
              encoding='utf-8') as _f9:
        _json9.dump({'fp': 'ring', 'sh': 'military',
                     'currency': 'residual z-phi',
                     'rho': _rho9,
                     'profile_weight': float(_ai9.PROFILE_WEIGHT),
                     'order': 'USER 2026-08-10 currency coherence'},
                    _f9)
    print('corpus source=%s: %d games' % (source, total), flush=True)
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK='TRUE',
               PYTHONIOENCODING='utf-8', PYTHONHASHSEED='0',
               CUDA_VISIBLE_DEVICES='-1')
    procs = {}

    def spawn(i):
        logf = open(os.path.join(OUT_DIR, 'worker%d.log' % i), 'a',
                    encoding='utf-8', errors='replace')
        p = subprocess.Popen(
            [sys.executable, '-u', os.path.abspath(__file__),
             '--shard', str(i), '--nshards', str(workers),
             '--source', source],
            stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        procs[i] = (p, logf)
        print('spawned worker %d (pid %d)' % (i, p.pid), flush=True)

    for i in range(workers):
        spawn(i)
    t0 = time.time()
    while True:
        time.sleep(120)
        files = os.listdir(OUT_DIR)
        done = sum(1 for f in files if f.endswith('.npz'))
        agg = aggregate_manifests()
        alive = sum(1 for p, _ in procs.values() if p.poll() is None)
        rate = done / max(1e-9, (time.time() - t0) / 3600)
        with open(PROGRESS, 'w', encoding='utf-8') as f:
            f.write('# v27 rendered extraction\n\nupdated: %s\n\n'
                    '- games done: %d / %d\n- admitted: %d quarantined: %d '
                    'errors: %d\n- kept samples: %d\n- rate: %.0f games/h; '
                    'ETA ~%.1f h\n- workers alive: %d/%d\n'
                    % (time.strftime('%Y-%m-%d %H:%M:%S'), done, total,
                       agg['admitted'], agg['quarantined'], agg['errors'],
                       agg['kept'], rate, (total - done) / max(rate, 1e-9),
                       alive, workers))
        if done >= total or all(p.poll() is not None
                                for p, _ in procs.values()):
            break
    for _, logf in procs.values():
        try:
            logf.close()
        except Exception:
            pass
    _print_agg()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', type=int, default=0)
    ap.add_argument('--canary', action='store_true')
    ap.add_argument('--driver', action='store_true')
    ap.add_argument('--workers', type=int, default=7)
    ap.add_argument('--shard', type=int, default=None)
    ap.add_argument('--nshards', type=int, default=None)
    ap.add_argument('--source', default='corpus', choices=['corpus', 'canary'])
    args = ap.parse_args()
    if args.canary:
        run_canary()
    elif args.pilot:
        run_pilot(args.pilot, source=args.source)
    elif args.driver:
        run_driver(args.workers, source=args.source)
    elif args.shard is not None:
        run_worker(args.shard, args.nshards or 1, source=args.source)
    else:
        ap.error('pick --canary, --pilot N, --driver, or --shard I --nshards K')


if __name__ == '__main__':
    main()
