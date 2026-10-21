"""RENDERED-LOG replay parser + truth mirror  (rendered-log parity census).

The CLIENT-rendered replay logs (data/rendered_logs/GAMEnn_*.rendered.txt) are
the public description of VASSAL gestures. Rendered text can retain stale
reserve origins and repeated drags; supported identity normalization checks
the corresponding raw physical IDs before updating the mirror. This module
parses the settled records into

  * a batch-segmented DECISION stream (one action-die usage per batch),
  * the RNG stream (action rolls, combat/hunt rolls, hunt-tile draws, deck
    draws, random companion pulls), and
  * a lightweight MIRROR STATE evolved line-by-line, producing a truth board
    snapshot per batch in exactly closed_loop_test._board_snapshot() schema.

No rules are applied here - the mirror is pure bookkeeping (it is the client's
resolved truth).  The engine drive + divergence census lives in
rendered_census.py.

Doctrine: no state forcing, no fabricated objects.  Companions inside the
fellowship do not exist on the board (they live only in fel.comps).
"""
import os
import re
import sys
from raw_log_reader import RawLogEncodingError, read_raw_log_text

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import board  # noqa: E402  (os.chdir to src on import)
from board import sanitize  # noqa: E402

# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------
# rendered nation adjective/name  ->  Nation enum NAME (matches _board_snapshot)
NATION_NAME = {
    'sauron': 'SAURON', 'isengard': 'ISENGARD', 'gondor': 'GONDOR',
    'rohan': 'ROHAN', 'elven': 'ELVES', 'elves': 'ELVES',
    'dwarven': 'DWARVES', 'dwarves': 'DWARVES', 'dwarf': 'DWARVES',
    'north': 'THE_NORTH', 'the north': 'THE_NORTH',
    's&e': 'SOUTHRON_EASTERLINGS', 'southron': 'SOUTHRON_EASTERLINGS',
    'southrons': 'SOUTHRON_EASTERLINGS', 'southron & easterling':
    'SOUTHRON_EASTERLINGS',
}

# companion display-name aliases -> canonical engine name (setup_initial_game)
COMP_ALIAS = {
    'merry': 'Meriadoc', 'meriadoc': 'Meriadoc',
    'pippin': 'Peregrin', 'peregrin': 'Peregrin',
    'gandalf the grey': 'Gandalf the Grey', 'strider': 'Strider',
    'legolas': 'Legolas', 'gimli': 'Gimli', 'boromir': 'Boromir',
    'gandalf the white': 'Gandalf the White', 'aragorn': 'Aragorn',
    'gollum': 'Gollum',
}
START_COMPANIONS = ['Gandalf the Grey', 'Strider', 'Legolas', 'Gimli',
                    'Boromir', 'Meriadoc', 'Peregrin']
# W3-018: companion Levels (rb p.10) for the guide rule fallback, and the
# VASSAL zone that holds the Guide figure
COMP_LEVEL = {'Gandalf the Grey': 3, 'Strider': 3, 'Legolas': 2, 'Gimli': 2,
              'Boromir': 2, 'Meriadoc': 1, 'Peregrin': 1}
GUIDE_ZONE = 'guide of the fellowship'
NAMED_MINIONS = {'saruman': 'Saruman', 'witch king': 'Witch King',
                 'the witch-king': 'Witch King', 'witch-king': 'Witch King',
                 'mouth of sauron': 'Mouth of Sauron',
                 'the mouth of sauron': 'Mouth of Sauron'}

# rendered piece names that carry an INTERNAL comma (would corrupt group
# splitting) -> a comma-free canonical token, normalised before any parse.
COMMA_NAME_FIX = [
    ('The Witch-King, The Black Captain', 'Witch King'),
    ('the Witch-King, The Black Captain', 'Witch King'),
    # card titles whose TITLE half carries an internal ' from '/' in ' (the
    # move/kill regexes anchor on those words as zone separators, so the rest
    # of the title would leak into the from/to zone).  Hyphenate the internal
    # separators; the '/' survives so parse_piece still classifies it a card.
    # ' to ' inside a title is safe (it precedes the real ' from <zone>' and
    # is captured whole by the non-greedy object group), so those need no fix.
    ('Horde from the East/Deadly Strife', 'Horde-from-the-East/Deadly Strife'),
    ('Foul Thing From the Deep/They are Terrible',
     'Foul-Thing-From-the-Deep/They are Terrible'),
    ('Swords in Eriador/Advantageous Position',
     'Swords-in-Eriador/Advantageous Position'),
]

# off-board / non-region zones
FELLOWSHIP_ZONES = {'fellowship of the ring', 'guide of the fellowship'}
CARD_ZONES = {'current fp card', 'current sa card', 'shadow hand',
              'free peoples hand', 'shadow hand (chosen)',
              'free peoples hand (chosen)'}
DICE_ZONES = {'sa dice', 'fp dice', 'current action',
              'the hunt for the ring'}
CAS_ZONES = {'sa casualties', 'fp casualties', 'casualties', 'dead'}
REINF_ZONES = {'sa reinforcements', 'fp reinforcements', 'reinforcements'}
# ('the hunt for the ring' classifies as 'dice' via DICE_ZONES — the
# eye-allocation handling relies on that; it must not appear here)
HUNT_ZONES = {'hunt pool', 'used hunt tiles', 'removed hunt tiles'}


_STRONGHOLD_RE = re.compile(r'^(.*)\s+stronghold$')


def _region_key(name):
    """Resolve display-name articles against the actual board region table."""
    key = sanitize(name)
    if key in board.regions:
        return key
    alternate = key[4:] if key.startswith('the ') else 'the ' + key
    return alternate if alternate in board.regions else key


def zone_kind(z):
    """Classify a from/to zone string."""
    zl = z.strip().lower()
    if zl in FELLOWSHIP_ZONES:
        return ('fellowship', None)
    if zl in CARD_ZONES:
        return ('card', None)
    if zl in DICE_ZONES:
        return ('dice', None)
    if zl in CAS_ZONES:
        return ('cas', None)
    if zl in REINF_ZONES:
        return ('reinf', None)
    if zl in HUNT_ZONES:
        return ('hunt', None)
    # compound pool zones ('reinforcements(1)/casualties(1)'): pieces
    # leaving them enter the board from the pool — classify reinf-first
    if 'reinforcement' in zl:
        return ('reinf', None)
    if 'casualt' in zl:
        return ('cas', None)
    if zl.startswith('political track'):
        m = re.search(r'(\d+)', zl)
        return ('political', int(m.group(1)) if m else None)
    if zl.startswith('fellowship track'):
        m = re.search(r'(\d+)', zl)
        return ('feltrack', int(m.group(1)) if m else None)
    if zl.startswith('mordor'):
        m = re.search(r'mordor\s+(\d+)', zl)
        return ('mordor', int(m.group(1)) if m else 0)
    # table-card slots ('Board1'..'BoardN') and ring tracks are off-board
    if re.match(r'^board\d*$', zl):
        return ('table', None)
    if 'elven ring' in zl or 'dark ring' in zl:
        return ('ring', None)
    m = _STRONGHOLD_RE.match(zl)
    if m:
        return ('stronghold', _region_key(m.group(1)))  # besieged garrison box
    # otherwise a board region
    return ('region', _region_key(z))


def _missplit_card_zone(z):
    """A from/to zone that carries a '/' AND is not a Mordor-track box is the
    tail of a card TITLE that leaked past a ' from '/' in ' the move regex
    mistook for a zone separator (e.g. an unlisted card 'X from Y/Z').  Real
    board zones never contain '/' except the Mordor boxes (handled by
    zone_kind), so this is a safe general detector for a mis-split card play.
    """
    if '/' not in z:
        return False
    zl = z.strip().lower()
    if zl.startswith('mordor') or 'cirith ungol' in zl or 'morgai' in zl:
        return False
    # real pool zones DO carry '/': 'reinforcements(1)/casualties(1)'
    # (GAME05 L408) — dropping those moves silently desynced the mirror
    if 'reinforcement' in zl or 'casualt' in zl:
        return False
    return True


def parse_piece(tok):
    """Parse one rendered piece token -> ('unit',nation,kind,count) /
    ('nazgul',n,None,count) / ('leader',nation,None,count) /
    ('named',name,None,1) / ('die',face,None,1) / ('card',name,None,1) /
    ('tile',name,None,1) / ('unknown',tok,None,1)."""
    t = tok.strip()
    if not t:
        return None
    m = re.match(r'^\[(\d+)\]\s*(.*)$', t)
    count = 1
    if m:
        count = int(m.group(1))
        t = m.group(2).strip()
    tl = t.lower()
    # event/combat card (title/subtitle) or ring name -> off-board play
    if '/' in t:
        return ('card', t, None, count)
    if tl.endswith('control marker'):
        side = 'SHADOW' if tl.startswith('shadow') else 'FREE_PEOPLES'
        return ('ctrlmarker', side, None, count)
    if tl.startswith('victory points marker'):
        return ('vp', None, None, count)
    # action die
    md = re.match(r'^(shadow|free)\s+action\s+die\s+\(([acehmpw])\)$', tl)
    if md:
        return ('die', md.group(2).upper(), None, count)
    if tl == 'eye':
        return ('tile', 'Eye', None, count)
    # nazgul
    if tl == 'nazgul' or tl == 'nazgûl':
        return ('nazgul', 'SAURON', None, count)
    # named minions / characters
    if tl in NAMED_MINIONS:
        return ('named', NAMED_MINIONS[tl], None, count)
    if tl in COMP_ALIAS:
        return ('named', COMP_ALIAS[tl], None, count)
    # "<Nation> Leader"
    m = re.match(r'^(.*)\s+leader$', tl)
    if m and m.group(1) in NATION_NAME:
        return ('leader', NATION_NAME[m.group(1)], None, count)
    # "<Nation> Regular/Elite"
    m = re.match(r'^(.*)\s+(regular|elite)$', tl)
    if m and m.group(1) in NATION_NAME:
        return ('unit', NATION_NAME[m.group(1)],
                m.group(2).upper(), count)
    # bare hunt tile ("3","1r", and EVERY special tile name the client
    # can draw - ONE answerer, log_replay.HUNT_TILE_PIECES; D-T2-9:
    # 'Phial of Galadriel' etc. parsed as 'unknown' and the AI's
    # Mordor-track move starved on its tile forever)
    if re.match(r'^\d+r?$', tl) or tl in _SPECIAL_TILE_NAMES:
        return ('tile', t, None, count)
    # bare nation (political marker handled by caller via zone)
    if tl in NATION_NAME:
        return ('nation', NATION_NAME[tl], None, count)
    return ('unknown', t, None, count)


def _special_tile_names():
    """Case-folded special hunt tile names (the ONE table:
    log_replay.HUNT_TILE_PIECES; numeric/Eye faces excluded)."""
    from log_replay import HUNT_TILE_PIECES
    out = set()
    for name in HUNT_TILE_PIECES.values():
        nl = str(name).lower()
        if re.match(r'^\d+r?$', nl) or nl == 'eye':
            continue
        out.add(nl)
    return frozenset(out)


_SPECIAL_TILE_NAMES = _special_tile_names()


def split_group(inner):
    """Split a 'group (...)' body on top-level commas (piece tokens have
    no nested commas in the rendered grammar)."""
    return [p.strip() for p in inner.split(',') if p.strip()]


def _settlement_owner_side(region_key):
    """Original faction side of a SETTLEMENT region (city/stronghold/town),
    from board metadata - replicates game_logic.is_shadow_region /
    is_fp_region without importing the (in-flux) engine.  Returns 'SHADOW',
    'FREE_PEOPLES', or None for a non-settlement / factionless region."""
    meta = board.regions.get(sanitize(region_key), {})
    if not (meta.get('is_city') or meta.get('is_stronghold')
            or meta.get('is_town')):
        return None
    pid = meta.get('parent_id')
    if not pid:
        return None
    fname = board.fid_to_name.get(pid)
    if not fname:
        return None
    fmeta = board.name_to_meta.get(fname, {})
    return 'SHADOW' if fmeta.get('isShadows', False) else 'FREE_PEOPLES'


# --------------------------------------------------------------------------
# Mirror state  (pure bookkeeping = client-authoritative truth)
# --------------------------------------------------------------------------
class Mirror:
    def __init__(self):
        # armies: region_key -> {'units': {(nat,kind):count}, 'leaders':[desc]}
        #   leader desc = ('NAMED',name) | ('NAZGUL','SAURON') | ('LEADER',nat)
        self.armies = {}
        self.pol = {}      # NATION -> [box, active]
        self._init_political()
        self.comps = set(START_COMPANIONS)   # companions in the fellowship
        # W3-018: occupants of the VASSAL Guide Of The Fellowship zone in
        # entry order (setup: Gandalf the Grey); the earliest occupant
        # still in the company is the declared Guide, an empty zone
        # falls back to the rule in _guide_name
        self.guide_box = ['Gandalf the Grey']
        self.fel_region = 'rivendell'
        self.fel_progress = 0
        self.fel_corruption = 0
        self.fel_hidden = True
        self.fel_mordor = False
        self.fel_step = 0
        self.ctrl = {'FREE_PEOPLES': set(), 'SHADOW': set()}
        # regions whose current control is RULE-DERIVED (amendment 1.55a
        # walk-in capture: an army entering an undefended enemy settlement
        # conquers it on entry - the engine does this immediately, the
        # human's physical control-marker gesture lags several records).
        # Flagged with provenance so a census diff separates gesture-truth
        # from rule-derived truth (like pol_derived).
        self.ctrl_derived = set()
        # regions whose control marker was placed by an EXPLICIT rendered
        # gesture (per side) - used to retire a derived flag once the human
        # catches up, and to keep a genuine gesture from re-flagging derived
        self._ctrl_gesture_SHADOW = set()
        self._ctrl_gesture_FREE_PEOPLES = set()
        self.besieged = set()
        # per-region count of UNITS inside the stronghold box: the
        # besieged flag was ACCRETIVE (GAME19 T8 - Minas Morgul stayed
        # flagged after FP wiped the garrison and captured it, so the
        # NEW Shadow siege recon skipped defender_siege_choice); a
        # siege exists only while the box holds >= 1 unit
        self._box_n = {}
        self.fpc = {}      # NATION -> {'regular':n,'elite':n,'leader':n}
        self.minion_history = {name: {'entered': False, 'eliminated': False}
                               for name in ('saruman', 'witch_king', 'mouth_of_sauron')}
        self._setup_armies()

    def _init_political(self):
        self.pol = {
            'ROHAN': [0, False], 'THE_NORTH': [0, False], 'ELVES': [0, True],
            'DWARVES': [0, False], 'GONDOR': [1, False],
            'SOUTHRON_EASTERLINGS': [1, True], 'ISENGARD': [2, True],
            'SAURON': [2, True],
        }
        # PURE-GESTURE shadow of the track (box + active) - what the human's
        # physical marker/flip gestures have shown so far.  self.pol carries
        # the RULE-TRUE state (gestures + the p.35 advance-on-attack the
        # marker lags), so a snapshot can flag which steps are rule-derived.
        self.pol_gesture = {n: v[0] for n, v in self.pol.items()}
        self.pol_gesture_active = {n: v[1] for n, v in self.pol.items()}
        self.pol_derived = set()   # nations whose current box/active is
        #                            ahead of the gesture (rule-derived)

    def _army(self, key):
        return self.armies.setdefault(
            key, {'units': {}, 'leaders': []})

    def _add_unit(self, key, nat, kind, n=1):
        a = self._army(key)
        a['units'][(nat, kind)] = a['units'].get((nat, kind), 0) + n

    def _del_unit(self, key, nat, kind, n=1):
        a = self.armies.get(key)
        if not a:
            return 0
        have = a['units'].get((nat, kind), 0)
        take = min(have, n)
        if take:
            a['units'][(nat, kind)] = have - take
            if a['units'][(nat, kind)] <= 0:
                del a['units'][(nat, kind)]
        return take

    def _add_leader(self, key, desc, n=1):
        a = self._army(key)
        for _ in range(n):
            a['leaders'].append(desc)

    def _del_leader(self, key, match, n=1):
        """Remove up to n leaders matching predicate `match` (desc)->bool."""
        a = self.armies.get(key)
        if not a:
            return 0
        removed = 0
        out = []
        for d in a['leaders']:
            if removed < n and match(d):
                removed += 1
                continue
            out.append(d)
        a['leaders'] = out
        return removed

    # ---- setup ----------------------------------------------------------
    def _setup_armies(self):
        def A(name, nat, reg=0, elite=0, leaders=0, nazgul=0):
            k = sanitize(name)
            for _ in range(reg):
                self._add_unit(k, nat, 'REGULAR')
            for _ in range(elite):
                self._add_unit(k, nat, 'ELITE')
            for _ in range(leaders):
                self._add_leader(k, ('LEADER', nat))
            for _ in range(nazgul):
                self._add_leader(k, ('NAZGUL', 'SAURON'))
        A('Erebor', 'DWARVES', 1, 2, 1)
        A('Ered Luin', 'DWARVES', 1)
        A('Iron Hills', 'DWARVES', 1)
        A('Grey Havens', 'ELVES', 1, 1, 1)
        A('Rivendell', 'ELVES', 0, 2, 1)
        A('Woodland Realm', 'ELVES', 1, 1, 1)
        A('Lorien', 'ELVES', 1, 2, 1)
        A('Minas Tirith', 'GONDOR', 3, 1, 1)
        A('Dol Amroth', 'GONDOR', 3)
        A('Osgiliath', 'GONDOR', 2)
        A('Pelargir', 'GONDOR', 1)
        A('Bree', 'THE_NORTH', 1)
        A('Carrock', 'THE_NORTH', 1)
        A('Dale', 'THE_NORTH', 1, 0, 1)
        A('North Downs', 'THE_NORTH', 0, 1)
        A('The Shire', 'THE_NORTH', 1)
        A('Edoras', 'ROHAN', 1, 1)
        A('Fords of Isen', 'ROHAN', 2, 0, 1)
        A("Helm's Deep", 'ROHAN', 1)
        A('Orthanc', 'ISENGARD', 4, 1)
        A('North Dunland', 'ISENGARD', 1)
        A('South Dunland', 'ISENGARD', 1)
        A('Barad-Dur', 'SAURON', 4, 1, 0, 1)
        A('Dol Guldur', 'SAURON', 5, 1, 0, 1)
        A('Gorgoroth', 'SAURON', 3)
        A('Minas Morgul', 'SAURON', 5, 0, 0, 1)
        A('Moria', 'SAURON', 2)
        A('Mount Gundabad', 'SAURON', 2)
        A('Nurn', 'SAURON', 2)
        A('Morannon', 'SAURON', 5, 0, 0, 1)
        A('Far Harad', 'SOUTHRON_EASTERLINGS', 3, 1)
        A('Near Harad', 'SOUTHRON_EASTERLINGS', 3, 1)
        A('North Rhun', 'SOUTHRON_EASTERLINGS', 2)
        A('South Rhun', 'SOUTHRON_EASTERLINGS', 3, 1)
        A('Umbar', 'SOUTHRON_EASTERLINGS', 3)

    # ---- snapshot (closed_loop_test._board_snapshot schema) -------------
    def snapshot(self):
        armies = {}
        for k, a in self.armies.items():
            units = []
            for (nat, kind), n in a['units'].items():
                units.extend([(nat, kind)] * n)
            leaders = list(a['leaders'])
            if not units and not leaders:
                continue
            armies[k] = (units, leaders)
        # At War (box 3) implies active by rule, even absent an explicit flip
        pol = {n: (v[0], (v[1] or v[0] >= 3)) for n, v in self.pol.items()}
        # Mordor-track boxes alias to 'gorgoroth' (task-documented); the
        # engine's fellowship_region keeps a sanitized region string, never
        # None -- report the alias so in-Mordor snapshots are comparable.
        fel = {'region': ('gorgoroth' if self.fel_mordor else self.fel_region),
               'progress': self.fel_progress, 'hidden': self.fel_hidden,
               'corruption': self.fel_corruption, 'mordor': self.fel_mordor,
               'step': self.fel_step,
               'comps': sorted(self.comps),
               'guide': self._guide_name(),
               # A fallback tie is useful for mirror display, but is not
               # evidence that the expert nominated that companion.
               'guide_provenance': self._guide_provenance()}
        ctrl = {'FREE_PEOPLES': sorted(self.ctrl['FREE_PEOPLES']),
                'SHADOW': sorted(self.ctrl['SHADOW'])}
        return {'armies': armies, 'pol': pol, 'fel': fel,
                'fpc': {n: dict(v) for n, v in self.fpc.items()},
                'minion_history': {name: dict(value) for name, value in self.minion_history.items()},
                'besieged': sorted(self.besieged), 'ctrl': ctrl,
                # provenance: nations whose current political box/active is
                # rule-derived (p.35 advance-on-attack), not yet gestured -
                # lets census diffs separate gesture-truth from rule-truth
                'pol_derived': sorted(self.pol_derived),
                'pol_gesture': {n: [self.pol_gesture[n],
                                    self.pol_gesture_active[n]]
                                for n in self.pol_gesture},
                # provenance: regions whose control is rule-derived (1.55a
                # walk-in capture) rather than an explicit marker gesture
                'ctrl_derived': sorted(self.ctrl_derived)}

    # ---- record application --------------------------------------------
    def _move_piece(self, piece, from_kind, to_kind):
        typ, a, b, cnt = piece
        fk, fmeta = from_kind
        tk, tmeta = to_kind
        # The casualty pool is a physical ledger, distinct from reserves.
        # FP Regulars reclaimed for Elite reductions leave this ledger;
        # ordinary Leaders sent there enter it just like army units.
        if typ in ('unit', 'leader') and a in self._fp_nations():
            kind = 'leader' if typ == 'leader' else b.lower()
            if fk == 'cas' and tk != 'cas':
                losses = self.fpc.setdefault(a, {'regular': 0, 'elite': 0, 'leader': 0})
                losses[kind] = max(0, losses.get(kind, 0) - cnt)
            elif tk == 'cas' and fk != 'cas':
                self._record_fp_cas(a, kind, cnt)
        # figures on the board
        if typ == 'unit':
            if fk in ('region', 'stronghold'):
                self._del_unit(fmeta, a, b, cnt)
                if fk == 'stronghold':
                    n = self._box_n.get(fmeta, 0) - (cnt or 1)
                    if n <= 0:
                        self._box_n.pop(fmeta, None)
                        self.besieged.discard(fmeta)
                    else:
                        self._box_n[fmeta] = n
            if tk in ('region', 'stronghold'):
                self._add_unit(tmeta, a, b, cnt)
                if tk == 'stronghold':
                    self._box_n[tmeta] = \
                        self._box_n.get(tmeta, 0) + (cnt or 1)
                    self.besieged.add(tmeta)
        elif typ == 'nazgul':
            if fk in ('region', 'stronghold'):
                self._del_leader(fmeta,
                                 lambda d: d == ('NAZGUL', 'SAURON'), cnt)
            if tk in ('region', 'stronghold'):
                self._add_leader(tmeta, ('NAZGUL', 'SAURON'), cnt)
        elif typ == 'leader':
            if fk in ('region', 'stronghold'):
                self._del_leader(fmeta,
                                 lambda d: d == ('LEADER', a), cnt)
            if tk in ('region', 'stronghold'):
                self._add_leader(tmeta, ('LEADER', a), cnt)
        elif typ == 'named':
            self._move_named(a, from_kind, to_kind)
        # everything else (die/card/tile/nation/unknown) = off-board
        # bookkeeping, no board effect

    def _fp_nations(self):
        return {'ELVES', 'DWARVES', 'GONDOR', 'THE_NORTH', 'ROHAN'}

    def _nation_side(self, nat):
        return 'FREE_PEOPLES' if nat in self._fp_nations() else 'SHADOW'

    def advance_on_attack(self, battle_regions, attacker_side, before_snap,
                          gestured=()):
        """Rulebook p.35 (RULE-derived truth, legitimate in a truth mirror):
        a nation whose army is ATTACKED activates AND advances one step
        immediately, at attack declaration - the human's physical marker
        move lags by several records or a whole turn.  Fires once per combat
        die-batch for every non-attacker nation with units in the battle
        region(s) (mirrors game_logic.advance_defenders_political, incl. the
        co-located-siege attacker-side filter).  The advance is recorded in
        self.pol (rule-true) and, when ahead of the gesture, flagged in
        self.pol_derived so a snapshot distinguishes gesture-truth from
        rule-derived truth."""
        if not attacker_side or not battle_regions:
            return
        armies = (before_snap or {}).get('armies', {})
        for reg in battle_regions:
            nats = set()
            for k, (units, _l) in armies.items():
                if sanitize(k) == reg:
                    nats |= set(n for (n, _kind) in units)
            for nat in nats:
                if nat not in self.pol or self._nation_side(nat) == \
                        attacker_side:
                    continue
                # W4-009: the rule-derived step only fills the LAG - when
                # the human moved this nation's marker inside the batch
                # (2024-04-30 L122 in the L119 attack batch) the gesture
                # IS the advance; deriving it again double-counts.
                if nat in gestured:
                    continue
                box, active = self.pol[nat]
                newbox = min(3, box + 1)
                if newbox == box and active:
                    continue   # already at war / no change
                self.pol[nat] = [newbox, True]
                if newbox > self.pol_gesture.get(nat, box) or \
                        not self.pol_gesture_active.get(nat, False):
                    self.pol_derived.add(nat)

    def enforce_settlement_control(self):
        """Rulebook amendment 1.55a (RULE-derived truth): a SETTLEMENT
        region occupied by exactly ONE side's Army units belongs to that
        side.  Mirrors game_logic.enforce_settlement_control - the engine
        conquers an undefended enemy settlement the instant an Army enters
        it (no dice), while the human's physical control-marker gesture
        lags.  Two-side (besieged) and empty regions are untouched;
        recapture and neutral settlements follow _check_settlement_capture.
        Any control this derives ahead of the marker gesture is flagged in
        self.ctrl_derived so a snapshot distinguishes gesture-truth from
        rule-derived truth (like advance_on_attack / pol_derived)."""
        for key, a in list(self.armies.items()):
            units = a.get('units') or {}
            if not any(c > 0 for c in units.values()):
                continue
            owner = _settlement_owner_side(key)
            if owner is None:
                continue
            sides = set(self._nation_side(nat) for (nat, _k), c
                        in units.items() if c > 0)
            if len(sides) != 1:
                continue          # besieged / mixed occupation - untouched
            occ = next(iter(sides))
            if occ == 'SHADOW' and owner == 'FREE_PEOPLES':
                if key not in self.ctrl['SHADOW'] and \
                        key not in self._ctrl_gestured('SHADOW'):
                    self.ctrl_derived.add(key)
                self.ctrl['SHADOW'].add(key)
                self.ctrl['FREE_PEOPLES'].discard(key)
            elif occ == 'FREE_PEOPLES' and owner == 'SHADOW':
                if key not in self.ctrl['FREE_PEOPLES'] and \
                        key not in self._ctrl_gestured('FREE_PEOPLES'):
                    self.ctrl_derived.add(key)
                self.ctrl['FREE_PEOPLES'].add(key)
                self.ctrl['SHADOW'].discard(key)
            elif occ == 'FREE_PEOPLES' and owner == 'FREE_PEOPLES':
                if key in self.ctrl['SHADOW']:
                    self.ctrl_derived.add(key)      # recapture, marker lags
                self.ctrl['SHADOW'].discard(key)
            elif occ == 'SHADOW' and owner == 'SHADOW':
                if key in self.ctrl['FREE_PEOPLES']:
                    self.ctrl_derived.add(key)
                self.ctrl['FREE_PEOPLES'].discard(key)

    def _ctrl_gestured(self, side):
        """Regions whose control marker for `side` was placed by an EXPLICIT
        rendered gesture (not rule-derived)."""
        return getattr(self, '_ctrl_gesture_' + side, set())

    def _record_fp_cas(self, nat, kind_l, cnt):
        d = self.fpc.setdefault(nat, {'regular': 0, 'elite': 0, 'leader': 0})
        if kind_l in ('regular', 'elite', 'leader'):
            d[kind_l] = d.get(kind_l, 0) + cnt

    def _guide_name(self):
        """W3-018: the Guide of the Fellowship as the log declares it.
        Gollum guides an empty company (rb p.39); otherwise the earliest
        occupant of the Guide Of The Fellowship zone still in the company
        ('moves <Companion> ... to Guide Of The Fellowship'); otherwise the
        rule (rb p.10: a new guide is chosen at once among the highest-
        Level companions - ties broken in setup order, which reproduces
        the 24-game declarations: GJB L133 Legolas over Gimli)."""
        real = [c for c in self.comps if c != 'Gollum']
        if not real:
            return 'Gollum'
        for c in self.guide_box:
            if c in real:
                return c
        lvl = max(COMP_LEVEL.get(c, 0) for c in real)
        for c in START_COMPANIONS:
            if c in real and COMP_LEVEL.get(c, 0) == lvl:
                return c
        return sorted(real)[0]

    def _guide_provenance(self):
        real = [c for c in self.comps if c != 'Gollum']
        occupants = [c for c in self.guide_box if c in real]
        if len(occupants) == 1:
            return 'declared'
        if len(occupants) > 1:
            return 'ambiguous_box'
        if not real:
            return 'forced'
        top = max(COMP_LEVEL.get(c, 0) for c in real)
        return ('forced' if sum(COMP_LEVEL.get(c, 0) == top for c in real) == 1
                else 'inferred_tie')

    def _leave_guide_box(self, name):
        if name in self.guide_box:
            self.guide_box.remove(name)

    def _track_guide(self, rec):
        """W3-018: guide-zone occupancy.  A companion dragged INTO the
        Guide Of The Fellowship zone joins the occupants (a group dragged
        in behind the sitting guide is a staging shuffle - MadRobin L339,
        BluPickle L566 - and the earlier occupant stays the guide; a
        stray drag in-and-out - MadRobin L66/67 Legolas - leaves the
        sitting guide untouched); every exit removes the occupant."""
        to = str(rec.get('to') or '').strip().lower()
        fr = str(rec.get('from') or '').strip().lower()
        if to != GUIDE_ZONE and fr != GUIDE_ZONE:
            return
        obj = str(rec.get('obj') or '').strip()
        mg = re.match(r'^group\s*\((.*)\)$', obj, re.I)
        toks = split_group(mg.group(1)) if mg else [obj]
        names = []
        for t in toks:
            p = parse_piece(t)
            if p and p[0] == 'named':
                names.append(COMP_ALIAS.get(p[1].lower(), p[1]))
        if fr == GUIDE_ZONE:
            self.guide_box = [c for c in self.guide_box if c not in names]
        if to == GUIDE_ZONE:
            for n in names:
                if n not in self.guide_box:
                    self.guide_box.append(n)

    def _move_named(self, name, from_kind, to_kind):
        fk, fmeta = from_kind
        tk, tmeta = to_kind
        nl = name.lower()
        attr = self._minion_attr(name)
        if attr is not None and (fk in ('region', 'stronghold') or tk in ('region', 'stronghold')):
            self.minion_history[attr]['entered'] = True
            if fk in ('region', 'stronghold') and tk not in ('region', 'stronghold'):
                self.minion_history[attr]['eliminated'] = True
        # leaving/entering the fellowship changes the companion set
        if fk == 'fellowship' and tk == 'fellowship':
            return  # guide<->company shuffle, no set change
        if fk == 'fellowship' and tk != 'fellowship':
            self.comps.discard(COMP_ALIAS.get(nl, name))
            self._leave_guide_box(COMP_ALIAS.get(nl, name))   # W3-018
            # a companion separated onto the map becomes an army leader
            if tk in ('region', 'stronghold'):
                self._add_leader(tmeta, ('NAMED', COMP_ALIAS.get(nl, name)))
            return
        if tk == 'fellowship' and fk != 'fellowship':
            self.comps.add(COMP_ALIAS.get(nl, name))
            if fk in ('region', 'stronghold'):
                self._del_leader(fmeta, lambda d: d[0] == 'NAMED'
                                 and d[1] == COMP_ALIAS.get(nl, name))
            return
        # board <-> board (named minion or crowned figure marching)
        if fk in ('region', 'stronghold'):
            self._del_leader(fmeta, lambda d: d[0] == 'NAMED'
                             and _named_eq(d[1], name))
        if tk in ('region', 'stronghold'):
            self._add_leader(tmeta, ('NAMED', name))

    @staticmethod
    def _minion_attr(name):
        name = re.sub(r'[^a-z]', '', str(name).lower()).removeprefix('the')
        return {'saruman': 'saruman', 'witchking': 'witch_king',
                'mouthofsauron': 'mouth_of_sauron'}.get(name)

    def _political_marker_move(self, obj, tk):
        """One political-marker gesture: obj (a nation name) moved to the
        VASSAL zone tk = ('political', box).  W4-009: follow the track
        LITERALLY - the rendered text is the truth.  VASSAL enforces nothing
        and the rulebook has no backward political move, so a backward
        marker move is the human's correction of a mis-drag (2023-01-24 L178
        Elves 2->0 overshoot, L181 0->1).  The old monotonic max kept the
        overshoot (Elves At War in the mirror for the rest of the game) and,
        together with advance_on_attack, stacked the p.35 rule-derived step
        on top of the human's own marker move for the same attack
        (2024-04-30 L122 inside the L119 attack batch, L135 inside L129:
        Elves reached At War while the track showed 1, and the human's real
        advance at L500 dropped as political_unmatched)."""
        ol = obj.strip().lower()
        p = parse_piece(obj)
        nat = None
        if p and p[0] in ('nation', 'leader', 'unit'):
            nat = p[1]
        elif ol in NATION_NAME:
            nat = NATION_NAME[ol]
        if nat and tk[0] == 'political' and tk[1] is not None:
            gbox = 3 - tk[1]               # vassal->internal
            self.pol_gesture[nat] = gbox
            self.pol[nat][0] = gbox
            if self.pol_gesture_active.get(nat, False) >= self.pol[nat][1]:
                self.pol_derived.discard(nat)

    def apply_move(self, rec):
        obj = rec['obj']
        # a mis-split card play (unlisted title with an internal ' from '/
        # ' in ') leaks the title tail into from/to; off-board no-op
        if _missplit_card_zone(rec.get('from', '')) or \
                _missplit_card_zone(rec.get('to', '')):
            return
        fk = zone_kind(rec['from'])
        tk = zone_kind(rec['to'])
        self._track_guide(rec)              # W3-018 guide-zone occupancy
        # fellowship pawn
        ol = obj.strip().lower()
        # VASSAL 'moves nothing from X to Y' = empty retreat-into-siege gesture
        if ol == 'nothing':
            return
        if ol == 'the fellowship':
            if tk[0] == 'region':
                self.fel_region = tk[1]
                # A pawn region move alone NEVER reveals (spec 1.216
                # "every reveal must have a source"): the sourced
                # 'Previously on Fellowship Track N' flip is the only
                # reveal evidence (apply_flip).  A DECLARATION moves
                # the pawn and keeps the Fellowship HIDDEN (v32, USER
                # 2026-08-03).
            # USER truth-table (game01_verification): a Fellowship
            # move whose DESTINATION is a 'Mordor N - ...' zone is the
            # Mordor entry / a track step - set fel_mordor on the MOVE
            # gesture, not only on flips.
            _mm = re.match(r'^\s*Mordor\s+(\d+)\s*-', str(rec.get('to')
                                                          or ''))
            if _mm:
                self.fel_mordor = True
                self.fel_step = int(_mm.group(1))
                # W3-017: Mordor ENTRY does NOT re-hide (engine
                # _check_mordor_entry, FINAL ruling 2026-08-16 "NO
                # re-hide"); the hidden state changes only on a flip.
            return
        if ol == 'fsp':
            if tk[0] == 'feltrack':
                self.fel_progress = tk[1]
            return
        if ol == 'corruption':
            if tk[0] == 'feltrack':
                self.fel_corruption = tk[1]
            return
        # USER truth-table fix (GAME01 T2 L172-173): the client renders
        # combined marker walks as 'moves group (FSP, Corruption) from
        # Fellowship Track A to Track B' - the exact-name handlers above
        # never matched the group form, so corruption/progress silently
        # desynced (T2 corruption stayed 2 while truth was 4).
        if ol.startswith('group (') and tk[0] == 'feltrack' \
                and ('fsp' in ol or 'corruption' in ol):
            if 'fsp' in ol:
                self.fel_progress = tk[1]
            if 'corruption' in ol:
                self.fel_corruption = tk[1]
            return
        # USER truth-table fix (game01_verification T2): a hunt-drawn
        # companion PLACED on the map (draw -> Current -> region walk,
        # never exiting via 'Fellowship Of The Ring') stayed in comps -
        # any named companion landing on a board region leaves the
        # fellowship company.
        _cn9 = COMP_ALIAS.get(ol, None)
        if _cn9 and tk[0] in ('region', 'stronghold') \
                and _cn9 in self.comps:
            self.comps.discard(_cn9)
            self._leave_guide_box(_cn9)                        # W3-018
        if ol.endswith('control marker'):
            side = 'SHADOW' if ol.startswith('shadow') else 'FREE_PEOPLES'
            other = 'FREE_PEOPLES' if side == 'SHADOW' else 'SHADOW'
            if tk[0] in ('region', 'stronghold'):
                self.ctrl[side].add(tk[1])
                self.ctrl[other].discard(tk[1])
                # explicit gesture: record it and retire any rule-derived
                # flag (gesture-truth has caught up to rule-truth)
                getattr(self, '_ctrl_gesture_' + side).add(tk[1])
                self.ctrl_derived.discard(tk[1])
            return
        if ol.startswith('victory points marker'):
            return  # VP not in board snapshot
        # bare nation political move: "Gondor from Political Track A to B"
        if tk[0] == 'political' or fk[0] == 'political':
            # W4-009: a GROUP drag of several markers ("moves group (Gondor,
            # Elves) from Political Track 1 to Political Track 0", 2023-01-24
            # L223, L42) is one marker gesture per nation - replay each
            # literally (the human's correction follows the same path).
            _mgp = re.match(r'^group\s*\((.*)\)$', obj, re.I)
            _toks = split_group(_mgp.group(1)) if _mgp else [obj]
            for _tok in _toks:
                self._political_marker_move(_tok.strip(), tk)
            return
        # group move
        mg = re.match(r'^group\s*\((.*)\)$', obj, re.I)
        if mg:
            toks = split_group(mg.group(1))
            fel_in_group = any(t.strip().lower() == 'the fellowship'
                               for t in toks)
            for tok in toks:
                tl2 = tok.strip().lower()
                if tl2 == 'the fellowship':
                    if tk[0] == 'region':
                        self.fel_region = tk[1]
                        # moves never reveal - sourced flips only
                        # (see the single-pawn handler above).
                    # Mordor entry via GROUP move: same destination
                    # handling as the single-pawn branch (GAME11 L2416)
                    _mmg9 = re.match(r'^\s*Mordor\s+(\d+)\s*-',
                                     str(rec.get('to') or ''))
                    if _mmg9:
                        self.fel_mordor = True
                        self.fel_step = int(_mmg9.group(1))
                        # W3-017: no re-hide on entry (see above)
                    continue
                p = parse_piece(tok)
                if not p:
                    continue
                # named companions travelling WITH the fellowship pawn are
                # inside the company (not board figures) - key on LIVE company
                # membership (self.comps), NOT the static START_COMPANIONS set:
                # a companion that has SEPARATED (removed from self.comps, now a
                # board leader) and rejoins the pawn's group move is a BOARD
                # figure that must FOLLOW the move to the destination, else it
                # lags at its old region and the reconstructed attacking army
                # loses its leadership (snoopymate25 b49: L412 group(The
                # Fellowship, Gandalf the Grey, Legolas, Merry, [3]Elven Elite)
                # Moria->North Dunland - the 3 separated companions were
                # stranded at Moria, zeroing the North Dunland army's
                # Leadership and mis-skipping the attack).  rb p.28/p.30
                # (Leadership = participating Characters' ratings; Leader
                # re-roll).
                if fel_in_group and p[0] == 'named' and \
                        COMP_ALIAS.get(p[1].lower()) in self.comps:
                    continue
                self._move_piece(p, fk, tk)
            return
        p = parse_piece(obj)
        if p:
            self._move_piece(p, fk, tk)

    def apply_kill(self, rec):
        # a single card kill whose TITLE half has an internal ' in ' leaks the
        # subtitle into the zone (which then carries a '/'); off-board no-op
        if _missplit_card_zone(rec.get('zone', '')):
            return
        zk = zone_kind(rec['zone'])
        for tok in rec['pieces']:
            p = parse_piece(tok)
            if not p:
                continue
            typ, a, b, cnt = p
            if zk[0] in ('region', 'stronghold'):
                key = zk[1]
                if typ == 'unit':
                    self._del_unit(key, a, b, cnt)
                    if zk[0] == 'stronghold':
                        # same box-exit bookkeeping as _move_piece: a
                        # garrison wiped by KILL lines must clear the
                        # besieged flag too (GAME19 accretive class)
                        _n9 = self._box_n.get(key, 0) - (cnt or 1)
                        if _n9 <= 0:
                            self._box_n.pop(key, None)
                            self.besieged.discard(key)
                        else:
                            self._box_n[key] = _n9
                    if a in self._fp_nations():
                        self._record_fp_cas(a, (b or '').lower(), cnt)
                elif typ == 'nazgul':
                    self._del_leader(
                        key, lambda d: d == ('NAZGUL', 'SAURON'), cnt)
                elif typ == 'leader':
                    self._del_leader(
                        key, lambda d: d == ('LEADER', a), cnt)
                    if a in self._fp_nations():
                        self._record_fp_cas(a, 'leader', cnt)
                elif typ == 'named':
                    attr = self._minion_attr(a)
                    if attr is not None:
                        self.minion_history[attr].update(entered=True, eliminated=True)
                    self._del_leader(key, lambda d: d[0] == 'NAMED'
                                     and _named_eq(d[1], a), cnt)
                elif typ == 'ctrlmarker':
                    # 'kills Shadow Control Marker in R' -> R no longer
                    # controlled by that side (recapture / liberation);
                    # retire provenance like the move-gesture path
                    self.ctrl[a].discard(key)
                    self.ctrl_derived.discard(key)
                    getattr(self, '_ctrl_gesture_' + a, set()).discard(
                        key)
            # kills in Current Action / off-board = die/card/tile consume
            # (bookkeeping only, no board effect) -- but a companion killed
            # off a fellowship zone leaves the company
            if typ == 'named':
                cn = COMP_ALIAS.get(a.lower(), a)
                if cn in self.comps and zk[0] not in ('region', 'stronghold'):
                    self.comps.discard(cn)
                    self._leave_guide_box(cn)              # W3-018 killed

    def apply_flip(self, rec):
        obj = rec['obj'].strip()
        ol = obj.lower()
        if ol == 'the fellowship':
            if rec.get('mordor') is not None:
                self.fel_mordor = True
                self.fel_step = rec['mordor']
                # W3-017: a Fellowship flip inside Mordor is the same
                # hidden/revealed TOGGLE as on the map (a hunt reveal in
                # Mordor is rendered by exactly this flip: GenericPie
                # 2023-01-02 L909 after the Eye draw, FateevVV
                # 2023-01-05 L461/L463 reveal then hide); forcing
                # hidden=True made the mirror unable to show a revealed
                # Fellowship in Mordor.
                self.fel_hidden = not self.fel_hidden
                return
            reg = rec.get('region')
            if reg:
                self.fel_region = sanitize(reg)
            # reveal (has "Previously on Fellowship Track") vs hide
            self.fel_hidden = not self.fel_hidden
            if rec.get('revealed'):
                self.fel_hidden = False
                # USER truth-table fix (game01_verification): the
                # 'Previously on Fellowship Track N' flip carries the
                # PAWN RESET - progress returns to 0 (both on real
                # reveals and on declaration progress-reset flips).
                # The mirror never applied it, so declarations were
                # undetectable (T3: progress stayed 2 through the
                # declare) and progress drifted for whole turns.
                self.fel_progress = 0
            return
        # W3-001 (COORD C-BO-1): the FSP-counter dialect.  Five of the 24
        # card-cover logs flip the progress COUNTER instead of the
        # Fellowship figure: '<FP> flips FSP in Fellowship Track N' is the
        # hidden/revealed TOGGLE (a die staged + flip + die killed = the
        # hide; after a hunt tile / revealing card = the reveal) and the
        # '. Previously on Fellowship Track N' form is VASSAL's report of
        # the counter being sent back to Track 0 (the reveal's pawn reset;
        # SlickRick L1413 shows it on a Mordor HIDE after a mis-dragged
        # counter, so it never forces 'revealed').  The mirror ignored the
        # object entirely, so every such game ran with an inverted or
        # stale hidden flag (gate L69..L708 inverted) and 35 gestures had
        # no row.
        if ol == 'fsp' and 'fellowship track' in str(
                rec.get('zone') or rec.get('region') or '').lower():
            self.fel_hidden = not self.fel_hidden
            if rec.get('revealed'):
                self.fel_progress = 0
            return
        # political nation flip = activation.  A single flip line can list
        # SEVERAL nations ("flips North, Dwarves in Political Track 3").
        if 'political' in (rec.get('zone', '') or '').lower():
            hit = False
            for part in obj.split(','):
                pl = part.strip().lower()
                nat = NATION_NAME.get(pl)
                if nat is None:
                    pp = parse_piece(part.strip())
                    if pp and pp[0] in ('nation', 'leader', 'unit'):
                        nat = pp[1]
                if nat is not None:
                    self.pol[nat][1] = True
                    self.pol_gesture_active[nat] = True   # gesture-observed
                    # gesture caught up to (or past) any rule-derived advance
                    if self.pol_gesture.get(nat, 0) >= self.pol[nat][0]:
                        self.pol_derived.discard(nat)
                    hit = True
            if hit:
                return
        # "flips Aragorn in <region>" = the CROWNING: the Strider figure at
        # <region> (separated on the board, or still the fellowship guide)
        # becomes Aragorn.  Without this the mirror keeps a phantom Strider
        # forever and every later "moves Aragorn ..." fails to match it.
        if ol == 'aragorn':
            reg = rec.get('region')
            if not reg:
                return
            # crowning while in an off-board zone (Casualties/Reinforcements):
            # the figure is placed on the board by the following move gesture,
            # never create an off-board pseudo-region army
            if zone_kind(reg)[0] not in ('region', 'stronghold'):
                self.comps.discard('Strider')
                return
            # W1-026: key by the ZONE key (the besieged-stronghold sub-box
            # '<X> Stronghold' is the region X; sanitize() kept the suffix
            # and the figure was looked up / parked under a phantom key)
            rk = zone_kind(reg)[1]
            had_board = self._del_leader(
                rk, lambda d: d[0] == 'NAMED' and _named_eq(d[1], 'Strider'))
            had_comp = 'Strider' in self.comps
            self.comps.discard('Strider')
            a = self.armies.get(rk)
            already = a and any(d[0] == 'NAMED' and _named_eq(d[1], 'Aragorn')
                                for d in a['leaders'])
            if (had_board or had_comp) and not already:
                self._add_leader(rk, ('NAMED', 'Aragorn'))
            return
        # "flips Gandalf the White in <region>" = the transformation: the Grey
        # figure (on the board or still in the fellowship) becomes White.
        # Remove any lingering Grey, then place White once (never duplicate).
        if ol.startswith('gandalf the white'):
            reg = rec.get('region')
            if not reg:
                return
            # transformation while in the Casualties box (Grey fell, White is
            # summoned): remove any lingering Grey, but let the FOLLOWING move
            # gesture place White on the board - never create an off-board
            # pseudo-region army (GAME17 'flips Gandalf the White in FP
            # Casualties' then 'moves ... to Fangorn')
            if zone_kind(reg)[0] not in ('region', 'stronghold'):
                for k in list(self.armies):
                    self._del_leader(k, lambda d: d[0] == 'NAMED'
                                     and _named_eq(d[1], 'Gandalf the Grey'))
                self.comps.discard('Gandalf the Grey')
                return
            # W1-026: key by the ZONE key (the besieged-stronghold sub-box
            # '<X> Stronghold' is the region X; sanitize() kept the suffix
            # and the figure was looked up / parked under a phantom key)
            rk = zone_kind(reg)[1]
            for k in list(self.armies):
                self._del_leader(k, lambda d: d[0] == 'NAMED'
                                 and _named_eq(d[1], 'Gandalf the Grey'))
            self.comps.discard('Gandalf the Grey')
            a = self.armies.get(rk)
            already = a and any(
                d[0] == 'NAMED' and _named_eq(d[1], 'Gandalf the White')
                for d in a['leaders'])
            if not already:
                self._add_leader(rk, ('NAMED', 'Gandalf the White'))
            return

    def apply(self, rec):
        op = rec['op']
        if op == 'move':
            self.apply_move(rec)
        elif op == 'kill':
            self.apply_kill(rec)
        elif op == 'flip':
            self.apply_flip(rec)
        # roll/draw/add/allocate/choose/chat = RNG or bookkeeping (no board)


def _named_eq(a, b):
    if a == b:
        return True
    al = COMP_ALIAS.get(str(a).lower(), str(a))
    bl = COMP_ALIAS.get(str(b).lower(), str(b))
    return al == bl


# --------------------------------------------------------------------------
# Line parsing
# --------------------------------------------------------------------------
_MOVE_RE = re.compile(
    r'^(\S+)\s+moves\s+(.*?)\s+from\s+(.+?)\s+to\s+(.+?)\.\s*$')
_AUTO_MOVE_RE = re.compile(
    r'^auto>\s+moves\s+(.*?)\s+from\s+(.+?)\s+to\s+(.+?)\.\s*$')
_KILL_GROUP_RE = re.compile(
    r'^(\S+)\s+kills\s+group\s+\((.*)\)\s+in\s+(.+?)\.\s*$')
_KILL_ONE_RE = re.compile(
    r'^(\S+)\s+kills\s+(.+?)\s+in\s+(.+?)\.\s*$')
_FLIP_RE = re.compile(r'^(\S+)\s+flips\s+(.*?)\.?\s*$')
_ROLL_LET_RE = re.compile(r'^(\S+)\s+rolls\s+([ACEHMPW ]+)\.?\s*$')
_ROLL_NUM_RE = re.compile(r'^<game>\s+(\S+)\s+rolls\s+([\d ]+?)\s*$')
_DRAW_RE = re.compile(r'^(\S+)\s+draws\s+(.+?)\.\s*$')
_ADD_EYE_RE = re.compile(r'^(\S+)\s+adds\s+(\d+)\s+eyes?\s+to\s+the\s+hunt')
_ALLOC_RE = re.compile(r'^(\S+)\s+allocates\s+an\s+eye')
_CHOOSE_RE = re.compile(r'^(\S+)\s+chooses\s+a\s+card')
_UNDO_RE = re.compile(r'^(\S+)\s+undoes\s+last\s+action\s*:\s*%?(.*)$')
_REDO_RE = re.compile(r'^(\S+)\s+redoes\s+last\s+action\s*:\s*%?(.*)$')
_TURN_RE = re.compile(r'^<replay>\s+Turn\s+(\d+)')
_DIE_STAGE_RE = re.compile(
    r'^(?:\S+)\s+moves\s+(Shadow|Free)\s+Action\s+Die\s+\(([ACEHMPW])\)'
    r'\s+from\s+(SA Dice|FP Dice)\s+to\s+Current Action', re.I)
# FP fellowship-move SHORTCUT rendering: some client versions stage the Free
# Action die straight from FP Dice to The Hunt For The Ring (skipping the
# "Current Action" step), then move the Fellowship - the die is placed in the
# Hunt Box after moving (rb p.37).  Without this the FP move-die glued into the
# preceding Shadow batch (die_side SHADOW), stranding the fellowship move as an
# un-drivable hunt (the shadow-tile-card / Kindred / Orcs class).  Shadow-die-
# to-Hunt is Hunt Allocation (Phase 3, turn start), NOT an Action-Resolution
# stage, so this is scoped to the FREE die from the FP pool only.  Proven in an
# isolated tree: 41-game driven-diverged stays 0, skipped 31->26, driven +8.
_FP_HUNT_STAGE_RE = re.compile(
    r'^(?:\S+)\s+moves\s+Free\s+Action\s+Die\s+\(([ACEHMPW])\)'
    r'\s+from\s+FP Dice\s+to\s+The Hunt For The Ring', re.I)
# W4-018 via-card staging dialect: the die is dragged from its pool onto the
# Current SA/FP Card zone first and from there to Current Action (2023-01-20
# L1058-1059, 2023-08-15 L63-64, 2025-01-18 L1203-1204).  The second drag is
# the staging; without a boundary there the die's action is glued into the
# previous batch (other side's fellowship move / another die).  Gated in
# parse_game on the die having LEFT the pool into a card zone first, so a die
# that merely visits the card zone (new minion dice from Reinforcements) never
# opens a phantom batch.
_DIE_TO_CARD_RE = re.compile(
    r'^(?:\S+)\s+moves\s+(Shadow|Free)\s+Action\s+Die\s+\(([ACEHMPW])\)'
    r'\s+from\s+(?:SA Dice|FP Dice)\s+to\s+Current (?:SA|FP) Card', re.I)
_DIE_VIA_CARD_STAGE_RE = re.compile(
    r'^(?:\S+)\s+moves\s+(Shadow|Free)\s+Action\s+Die\s+\(([ACEHMPW])\)'
    r'\s+from\s+Current (?:SA|FP) Card\s+to\s+Current Action', re.I)


_NAMED_AUTO_RE = re.compile(r'^(?:[$%])?WOTR_AI_(?:Shadow|FP) \[auto\](?=\s|$)')


def canonical_automatic_line(line):
    """Keep native automatic/undo semantics when the UI names its AI actor."""
    return _NAMED_AUTO_RE.sub('auto>', line, count=1)


def parse_line(raw):
    """Parse ONE rendered line (leading '%' already stripped) into a record
    dict, or None for pure chat/noise.  Records carry op + fields."""
    s = canonical_automatic_line(raw.strip())
    if not s:
        return None
    for a, b in COMMA_NAME_FIX:
        if a in s:
            s = s.replace(a, b)
    # turn boundary
    m = _TURN_RE.match(s)
    if m:
        return {'op': 'turn', 'turn': int(m.group(1))}
    # chat / game tags (skip, but detect roll chunks tagged <game>)
    mnum = _ROLL_NUM_RE.match(s)
    if mnum:
        ints = [int(x) for x in mnum.group(2).split()]
        return {'op': 'roll_num', 'player': mnum.group(1), 'ints': ints}
    if s.startswith('<'):
        route = re.fullmatch(r'<([^<>]+)>\s*(?:via\s+)?([A-Za-z][A-Za-z\s\'-]*?)\s*[.!]?\s*', s, re.I)
        if route and sanitize(route.group(2)) in board.regions:
            # Inert evidence only. A later Hunt/reveal matcher must establish
            # actor, timing, legal route and extra-draw consequences.
            return {'op': 'route_hint', 'player': route.group(1),
                    'region': sanitize(route.group(2))}
        return None   # chat/address/game/replay noise
    # auto> card play
    ma = _AUTO_MOVE_RE.match(s)
    if ma:
        return {'op': 'move', 'player': 'auto', 'obj': ma.group(1),
                'from': ma.group(2), 'to': ma.group(3)}
    # undo / redo (net-resolution: amendment 1.216 undo/redo)
    mu = _UNDO_RE.match(s)
    if mu:
        return {'op': 'undo', 'player': mu.group(1),
                'inner': mu.group(2).strip()}
    mr = _REDO_RE.match(s)
    if mr:
        return {'op': 'redo', 'player': mr.group(1),
                'inner': mr.group(2).strip()}
    # move
    m = _MOVE_RE.match(s)
    if m:
        rec = {'op': 'move', 'player': m.group(1), 'obj': m.group(2),
               'from': m.group(3), 'to': m.group(4)}
        md = _DIE_STAGE_RE.match(s)
        if md:
            rec['die_stage'] = (
                'SHADOW' if md.group(1).lower() == 'shadow'
                else 'FREE_PEOPLES', md.group(2))
        else:
            mh = _FP_HUNT_STAGE_RE.match(s)
            if mh:
                rec['die_stage'] = ('FREE_PEOPLES', mh.group(1))
        return rec
    # kill group
    m = _KILL_GROUP_RE.match(s)
    if m:
        return {'op': 'kill', 'player': m.group(1),
                'pieces': split_group(m.group(2)), 'zone': m.group(3)}
    # kill single
    m = _KILL_ONE_RE.match(s)
    if m:
        return {'op': 'kill', 'player': m.group(1),
                'pieces': [m.group(2)], 'zone': m.group(3)}
    # flip
    m = _FLIP_RE.match(s)
    if m:
        body = m.group(2)
        rec = {'op': 'flip', 'player': m.group(1)}
        # "The Fellowship in Mordor N - Label"
        mm = re.match(r'^(.*?)\s+in\s+Mordor\s+(\d+)\s*-\s*(.*)$', body)
        if mm:
            rec['obj'] = mm.group(1)
            rec['mordor'] = int(mm.group(2))
            rec['zone'] = 'mordor'
            return rec
        # "The Fellowship in X. Previously on Fellowship Track N"
        mm = re.match(r'^(.*?)\s+in\s+(.+?)\.\s*Previously on '
                      r'Fellowship Track\s+(\d+)', body)
        if mm:
            rec.update({'obj': mm.group(1), 'region': mm.group(2),
                        'revealed': True,
                        'prev_track': int(mm.group(3))})
            return rec
        # "X in Political Track N"  /  "The Fellowship in X"
        mm = re.match(r'^(.*?)\s+in\s+(.+?)\.?\s*$', body)
        if mm:
            zone = mm.group(2)
            rec['obj'] = mm.group(1)
            rec['zone'] = zone
            if not zone.lower().startswith(('political', 'mordor')):
                rec['region'] = zone
            return rec
        rec['obj'] = body
        return rec
    # roll (letters = action dice)
    m = _ROLL_LET_RE.match(s)
    if m:
        return {'op': 'roll_dice', 'player': m.group(1),
                'faces': m.group(2).split()}
    # draw
    m = _DRAW_RE.match(s)
    if m:
        return {'op': 'draw', 'player': m.group(1), 'what': m.group(2)}
    m = _ADD_EYE_RE.match(s)
    if m:
        return {'op': 'add_eye', 'player': m.group(1),
                'n': int(m.group(2))}
    if _ALLOC_RE.match(s):
        return {'op': 'alloc_eye',
                'player': _ALLOC_RE.match(s).group(1)}
    if _CHOOSE_RE.match(s):
        return {'op': 'choose', 'player': _CHOOSE_RE.match(s).group(1)}
    return {'op': 'other', 'raw': s}




def resolve_undo_redo(records):
    """Net physical snapshots; preserve non-snapshot records.

    Native undo restores piece positions, not dice rolls or chosen-card UI.
    The rendered client emits one quoted label per automatic snapshot too.
    Every target is validated before reconstruction; new snapshots alone
    truncate redo, matching ay_0.k/o/p and m_0.b.
    """
    snapshot_ops = {'move', 'kill', 'flip', 'draw', 'alloc_eye', 'add_eye'}
    stack, redo, removed = [], [], set()
    def label(value):
        value = str(value or '').strip().lstrip('%$')
        for old, new in COMMA_NAME_FIX:
            value = value.replace(old, new)
        return value.replace('<auto>', 'auto>')
    for index, rec in enumerate(records):
        op = rec.get('op')
        if op in ('undo', 'redo'):
            source = stack if op == 'undo' else redo
            target_index = source[-1] if source else None
            target = records[target_index] if target_index is not None else None
            if target is None or label(target.get('body')) != label(rec.get('inner')):
                raise ClientUndoHistoryError({'detail': 'quoted_snapshot_target_mismatch',
                    'line': rec.get('line'), 'operation': op, 'quoted_target': rec.get('inner'),
                    'resolver_target_line': target.get('line') if target else None})
            source.pop()
            (redo if op == 'undo' else stack).append(target_index)
            if op == 'undo': removed.add(target_index)
            else: removed.remove(target_index)
            removed.add(index)
        elif op in snapshot_ops or (op == 'other' and re.fullmatch(
                r'\S+ randomly discards a (?:Free|Shadow) (?:Character|Strategy) Card\.',
                str(rec.get('body') or ''))):
            stack.append(index)
            redo.clear()
    return [rec for index, rec in enumerate(records) if index not in removed]


class ClientUndoHistoryError(ValueError):
    """The rendered history exceeds the supported client undo contract."""
    def __init__(self, detail):
        self.reason = 'coverage_client_undo_history_unproven'
        self.detail = detail
        super().__init__(self.reason, detail)



def _paired_raw_path(rendered_path, raw_path, refuse):
    if raw_path is None:
        rendered_dir = os.path.dirname(os.path.abspath(rendered_path))
        stem = os.path.basename(str(rendered_path)).removesuffix('.rendered.txt')
        # GAME-prefixed legacy fixtures use rendered_census.raw_stem's
        # established mapping to the exact original game-log basename.
        canonical = re.sub(r'^GAME[0-9]+_', '', stem)
        folder = os.path.join(os.path.dirname(rendered_dir), 'game_logs')
        candidates = []
        for candidate in (os.path.join(rendered_dir, stem + '.log'),
                          os.path.join(folder, canonical + '.log')):
            if os.path.isfile(candidate):
                candidates.append(os.path.realpath(candidate))
        # A spelling variant is permitted only for a timestamped corpus
        # name and a unique file in this same game's sibling raw folder.
        if re.match(r'^\d{4}-\d{2}-\d{2}_\d{6}_FP_.+_SP_.+$', canonical) and os.path.isdir(folder):
            key = re.sub(r'[ _]+', '_', canonical).lower()
            candidates += [os.path.realpath(os.path.join(folder, name))
                          for name in os.listdir(folder)
                          if name.endswith('.log') and 'replaynav' not in name.lower()
                          and re.sub(r'[ _]+', '_', name[:-4]).lower() == key]
        candidates = sorted(set(candidates))
        if len(candidates) > 1:
            refuse('raw_log_ambiguous')
        raw_path = candidates[0] if candidates else os.path.join(folder, canonical + '.log')
    return raw_path


def _validate_client_undo_history(records, rendered_path=None, raw_path=None):
    if not any(record.get('op') in ('undo', 'redo') for record in records):
        return
    resolve_undo_redo(records)
    def refuse(detail):
        raise ClientUndoHistoryError({'detail': detail})
    if rendered_path is None and raw_path is None:
        refuse('paired_raw_log_missing')
    raw_path = _paired_raw_path(rendered_path, raw_path, refuse)
    if not os.path.isfile(raw_path):
        refuse('paired_raw_log_missing')
    from client_snapshot_net import validate_rendered_pair, SnapshotPairError
    try:
        validate_rendered_pair(records, read_raw_log_text(raw_path).splitlines())
    except SnapshotPairError as exc:
        raise ClientUndoHistoryError(exc.detail) from exc


def _parse_all(path):
    """Parse every %-line of a rendered log into records (in file order),
    tagging each with its 1-based source line and raw body.  Returns
    (records, n_lines)."""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        raw_lines = [ln.rstrip('\n') for ln in f]
    recs = []
    for i, raw in enumerate(raw_lines, 1):
        s = raw.strip()
        if not s.startswith('%'):
            continue
        body = s[1:].strip()
        rec = parse_line(body)
        if rec is None:
            continue
        rec['line'] = i
        rec['body'] = body
        recs.append(rec)
    return recs, len(raw_lines)


# --------------------------------------------------------------------------
# Full game parse: batches + RNG + truth snapshots
# --------------------------------------------------------------------------
def _draw_kind(what, records=()):
    w = what.strip()
    wl = w.lower()
    if wl in ('free character card', 'free strategy card',
              'shadow character card', 'shadow strategy card'):
        return ('deck', w)
    if wl == 'eye' or re.match(r'^\d+r?$', wl) or wl in (
            "shelob's lair", 'elven rope'):
        return ('tile', w)
    if wl in _SPECIAL_TILE_NAMES:
        # Bare special-tile names can also caption a named Event-card draw.
        # Require a preceding public Hunt cause in this paid-action prefix;
        # never use hidden hands or a later drawn outcome to classify it.
        hunt_events = ('foul-thing-from-the-deep', 'foul thing from the deep',
                       'orc patrol', "isildur's bane", 'the breaking of the fellowship',
                       'there is another way', 'challenge of the king')
        for record in reversed(records):
            if record.get('op') != 'move':
                continue
            obj = str(record.get('obj', '')).strip().lower()
            origin = str(record.get('from', '')).strip().lower()
            target = str(record.get('to', '')).strip().lower()
            if obj in ('the fellowship', 'fsp') and (
                    target.startswith(('mordor ', 'fellowship track '))
                    or origin.startswith(('mordor ', 'fellowship track '))):
                return ('tile', w)
            if (obj.split('/')[0].strip() in hunt_events
                    and origin in ('free peoples hand', 'shadow hand')
                    and target in ('current fp card', 'current sa card')):
                return ('tile', w)
    # A random COMPANION pull always NAMES a real Companion (Foul Thing / a
    # hunt casualty).  Any other drawn title is a CARD (deck) draw - e.g. the
    # Shadow Character card 'Give it to Uss!' (ppoma b77).  Classify as
    # companion ONLY when the name is a recognised Companion; otherwise it is
    # a deck/card draw.
    if wl in COMP_ALIAS:
        return ('companion', w)
    return ('deck', w)


class PhysicalRecruitIdentityError(ValueError):
    """A repeated reserve gesture cannot be assigned a physical identity."""
    def __init__(self, detail):
        self.reason = 'coverage_physical_recruit_identity_unproven'
        self.detail = detail
        super().__init__(self.reason, detail)


def _normalize_physical_recruit_runs(records, rendered_path, raw_path=None, _raw_lines=None):
    """Resolve contiguous stale reserve-origin gestures using actual IDs.

    Identical rendered unit names do not prove duplicate recruitment.
    Only aligned, contiguous gestures of the same complete physical set
    can collapse to their final destination. All original records survive
    in the retained record's provenance.
    """
    import copy
    from collections import Counter

    def refuse(detail, line=None):
        raise PhysicalRecruitIdentityError({'detail': detail, 'line': line})

    def rendered_recruit(record):
        if record.get('op') != 'move' or str(record.get('from', '')).lower() not in ('fp reinforcements', 'sa reinforcements'):
            return None
        target_kind, target = zone_kind(record.get('to', ''))
        if target_kind not in ('region', 'stronghold') or target not in board.regions:
            return None
        cargo = str(record.get('obj', ''))
        tokens = split_group(cargo[7:-1]) if cargo.startswith('group (') and cargo.endswith(')') else [cargo]
        counter = Counter()
        for token in tokens:
            piece = parse_piece(token)
            if piece is None or piece[0] not in ('unit', 'leader'):
                return None
            kind, nation, unit, count = piece
            counter[(nation, unit if kind == 'unit' else 'LEADER')] += count
        return counter

    recruits = [(i, r, rendered_recruit(r)) for i, r in enumerate(records)]
    recruits = [(i, r, pieces) for i, r, pieces in recruits if pieces]
    by_index = {i: (r, pieces) for i, r, pieces in recruits}
    suspects = []
    for i, record, pieces in recruits:
        previous = by_index.get(i - 1)
        if previous is None:
            continue
        prior, prior_pieces = previous
        if (prior_pieces == pieces and prior.get('player') == record.get('player')
                and prior.get('from') == record.get('from')):
            suspects.append((i - 1, i))
    def signature(player, origin, pieces):
        return (player, str(origin).lower(), tuple(sorted(pieces.items())))
    relevant = {signature(records[i]['player'], records[i]['from'], by_index[i][1])
                for pair in suspects for i in pair}
    prefixes = {}
    turn, rolls, stages = 1, 0, 0
    for i, record in enumerate(records):
        if record.get('op') == 'turn':
            turn = int(record['turn'])
        if record.get('op') == 'roll_dice':
            rolls += 1
        if record.get('die_stage'):
            stages += 1
        prefixes[i] = (turn, rolls, stages)

    raw_path = _paired_raw_path(rendered_path, raw_path, refuse)
    if not os.path.isfile(raw_path):
        if suspects:
            refuse('raw_log_missing')
        return records
    import log_census as lc
    from piece_table_authoritative import AUTH_ARMY_PIECES, AUTH_PIECE_LABELS
    from location_codes_authoritative import AUTH_LOCATION_CODES
    raw_lines = (lc.undo_redo_net(read_raw_log_text(raw_path).splitlines())
                 if _raw_lines is None else _raw_lines)
    raw_recruits = []
    raw_prefixes = {}
    turn, rolls, stages = 1, 0, 0
    fp_dice = lc.BASE_FP_DIE_PIDS | lc.KNOWN_BONUS_FP
    sh_dice = lc.BASE_SH_DIE_PIDS | lc.KNOWN_BONUS_SH
    for line, text in enumerate(raw_lines, 1):
        turn_match = re.search(r'<~Game\.7~>(\d+)\b', text)
        if turn_match:
            turn = int(turn_match.group(1))
        if lc.ROLL_RE.search(text):
            rolls += 1
        raw_prefixes[line] = (turn, rolls, stages)
        match = lc.MOVE_RE.search(text)
        if not match:
            continue
        ids, origin, target = match.group(1).split(), match.group(2), match.group(3)
        # Match the rendered stage grammar: a physical die may have been
        # parked in the other player's pool before its actual paid gesture.
        # The ID still identifies its owner; this counts transport stages,
        # while dice legality remains checked by extraction.
        if len(ids) == 1 and ((ids[0] in fp_dice | sh_dice
                              and origin in ('Game.2042', 'Game.2043')
                              and target == 'Game.2035')
                             or (ids[0] in fp_dice and origin == 'Game.2042'
                                 and target == 'Game.1997')):
            stages += 1
        raw_prefixes[line] = (turn, rolls, stages)
        if origin not in ('Game.2038', 'Game.2040'):
            continue
        target_name = AUTH_LOCATION_CODES.get(target, '')
        target_kind, region = zone_kind(target_name)
        if target_kind not in ('region', 'stronghold') or region not in board.regions:
            continue
        pieces = [AUTH_ARMY_PIECES.get(pid) for pid in ids]
        if not all(p is not None and p[1] in ('REGULAR', 'ELITE', 'LEADER') for p in pieces):
            continue
        if len(set(ids)) != len(ids):
            refuse('duplicate_id_inside_gesture', line)
        player = text.split('<~', 1)[0].lstrip('$')
        raw_recruits.append({'line': line, 'ids': ids, 'from': AUTH_LOCATION_CODES[origin],
                             'to': target_name, 'pieces': Counter(pieces),
                             'player': player, 'raw': text, 'prefix': (turn, rolls, stages)})
    for prior, current in zip(raw_recruits, raw_recruits[1:]):
        if (prior['line'] + 1 == current['line']
                and prior['player'] == current['player'] and prior['from'] == current['from']
                and set(prior['ids']) == set(current['ids'])):
            relevant.add(signature(current['player'], current['from'], current['pieces']))
    if not relevant:
        return records
    # Include every reserve gesture touching a candidate physical ID,
    # even a singleton by another player. Iterate because a larger group
    # can connect additional IDs to the same lifecycle under review.
    relevant_ids = set()
    while True:
        old_size = (len(relevant), len(relevant_ids))
        for raw in raw_recruits:
            key = signature(raw['player'], raw['from'], raw['pieces'])
            if key in relevant or relevant_ids.intersection(raw['ids']):
                relevant.add(key)
                relevant_ids.update(raw['ids'])
        if old_size == (len(relevant), len(relevant_ids)):
            break
    recruits = [(i, r, pieces) for i, r, pieces in recruits
                if signature(r['player'], r['from'], pieces) in relevant]
    raw_recruits = [r for r in raw_recruits if signature(r['player'], r['from'], r['pieces']) in relevant]
    if len(raw_recruits) != len(recruits):
        refuse('raw_rendered_recruit_count')
    aligned = {}
    for (index, record, pieces), raw in zip(recruits, raw_recruits):
        if (pieces != raw['pieces'] or record.get('player') != raw['player']
                or zone_kind(record['from']) != zone_kind(raw['from'])
                or zone_kind(record['to']) != zone_kind(raw['to'])
                or prefixes[index] != raw['prefix']):
            refuse('raw_rendered_recruit_alignment', record.get('line'))
        aligned[index] = raw

    def returned_to_reserve(prior_i, next_i, overlap):
        """Prove an intervening physical return in both public streams.

        A later recruit can reuse a Shadow casualty. Match each intervening
        move/kill of the reused IDs and require their last physical zone to
        be the declared reserve. A chat line or an unmatched raw return is
        not evidence of a new recruitment episode.
        """
        prior, current = aligned[prior_i], aligned[next_i]
        def physical_zone(name):
            kind = zone_kind(name)
            return (kind, None if kind[0] in ('region', 'stronghold') else str(name).strip().lower())
        locations = {pid: physical_zone(prior['to']) for pid in overlap}
        match_index = prior_i + 1

        def cargo_counter(tokens):
            result = Counter()
            for token in tokens:
                piece = parse_piece(token)
                if piece is None:
                    return None
                kind, nation, unit, count = piece
                if kind in ('unit', 'leader'):
                    result[('army', nation, unit if kind == 'unit' else 'LEADER')] += count
                elif kind in ('named', 'nazgul'):
                    result[(kind, str(nation).casefold())] += count
                elif kind == 'die':
                    side = re.search(r'(shadow|free)\s+action\s+die', token, re.I)
                    if side is None:
                        return None
                    result[('die', 'SH' if side.group(1).lower() == 'shadow' else 'FP')] += count
                elif kind == 'card' and '/' in str(nation):
                    result[('card', ' '.join(str(nation).split()).casefold())] += count
                else:
                    return None
            return result

        def raw_cargo_counter(ids):
            result = Counter()
            for pid in ids:
                if pid in AUTH_ARMY_PIECES:
                    nation, kind = AUTH_ARMY_PIECES[pid]
                    if kind == 'NAZGUL':
                        result[('nazgul', 'sauron')] += 1
                    else:
                        result[('army', nation, kind)] += 1
                    continue
                label = AUTH_PIECE_LABELS.get(pid, '')
                if label in ('SH action die', 'FP action die'):
                    result[('die', label[:2])] += 1
                elif '/' in label:
                    # Card PIDs are shuffled slots. Their construction
                    # labels are not identities in this game: require a
                    # preceding public reveal, never hidden decoding.
                    from card_decoder import (_FP_PID_TYPES, _SA_PID_TYPES,
                        FP_CHARACTER_CARDS, FP_STRATEGY_CARDS,
                        SA_CHARACTER_CARDS, SA_STRATEGY_CARDS)
                    ipid = int(pid)
                    if 106 <= ipid <= 157:
                        deck = _FP_PID_TYPES[ipid - 106]
                        names = FP_CHARACTER_CARDS if deck == 'C' else FP_STRATEGY_CARDS
                    elif 158 <= ipid <= 209:
                        deck = _SA_PID_TYPES[ipid - 158]
                        names = SA_CHARACTER_CARDS if deck == 'C' else SA_STRATEGY_CARDS
                    else:
                        return None
                    if deck not in ('C', 'S'):
                        return None
                    title = None
                    for earlier in reversed(raw_lines[:raw_line - 1]):
                        killed = lc.KILL_RE.search(earlier) or re.search(r'<~Controls\.662~>([\d\s]+?)<~Controls\.663~>', earlier)
                        if killed and pid in killed.group(1).split():
                            break
                        revealed = re.search(r'<~Controls\.\d+~>reveal\s+' + re.escape(pid) + r'\s+[0-9A-Fa-f]{40}\s+(\d+)\s*<~', earlier) if earlier.startswith(('$', '<auto>')) else None
                        if revealed:
                            title = names.get(int(revealed.group(1)))
                            break
                    if title is None:
                        return None
                    options = {v for v in AUTH_PIECE_LABELS.values() if '/' in v and v.split('/', 1)[0].casefold() == title.casefold()}
                    if len(options) != 1:
                        return None
                    caption = options.pop()
                    for original, protected in COMMA_NAME_FIX:
                        caption = caption.replace(original, protected)
                    result[('card', ' '.join(caption.split()).casefold())] += 1
                else:
                    parsed = parse_piece(label)
                    if parsed is None or parsed[0] not in ('named', 'nazgul'):
                        return None
                    result[(parsed[0], str(parsed[1]).casefold())] += 1
            return result

        for raw_line in range(prior['line'] + 1, current['line']):
            text = raw_lines[raw_line - 1]
            move = lc.MOVE_RE.search(text)
            kill = lc.KILL_RE.search(text) or re.search(r'<~Controls\.662~>([\d\s]+?)<~Controls\.663~>', text)
            swap_remove = lc.SWAP_REMOVE_RE.search(text)
            swap_place = lc.SWAP_PLACE_RE.search(text)
            if move:
                ids = move.group(1).split()
                origin = AUTH_LOCATION_CODES.get(move.group(2), '')
                target = AUTH_LOCATION_CODES.get(move.group(3), '')
                op = 'move'
            elif swap_place:
                ids = swap_place.group(1).split()
                if swap_place.group(2) != '307':
                    if overlap.intersection(ids):
                        return False
                    continue
                origin = 'reinforcements'
                target = AUTH_LOCATION_CODES.get(swap_place.group(3), '')
                if zone_kind(target)[0] not in ('region', 'stronghold'):
                    return False
                op = 'move'
            elif kill or swap_remove:
                ids = (kill or swap_remove).group(1).split()
                if swap_remove:
                    origin = AUTH_LOCATION_CODES.get(swap_remove.group(2), '')
                else:
                    zone = re.match(r'<~([\w.]+)~>', text[kill.end():])
                    if zone is None:
                        return False
                    origin = AUTH_LOCATION_CODES.get(zone.group(1), '')
                target = None
                op = 'kill'
            else:
                continue
            touched = overlap & set(ids)
            if not touched:
                continue
            if len(set(ids)) != len(ids):
                return False
            pieces = raw_cargo_counter(ids)
            if pieces is None:
                return False
            player = text.split('<~', 1)[0].lstrip('$')
            found = None
            for j in range(match_index, next_i):
                record = records[j]
                if record.get('op') != op or record.get('player') != player:
                    continue
                if op == 'move':
                    cargo = str(record.get('obj', ''))
                    tokens = split_group(cargo[7:-1]) if cargo.startswith('group (') and cargo.endswith(')') else [cargo]
                    if physical_zone(record.get('from', '')) != physical_zone(origin) or \
                            physical_zone(record.get('to', '')) != physical_zone(target):
                        continue
                else:
                    tokens = record.get('pieces', [])
                    if physical_zone(record.get('zone', '')) != physical_zone(origin):
                        continue
                if cargo_counter(tokens) == pieces and prefixes[j] == raw_prefixes[raw_line]:
                    found = j
                    break
            if found is None:
                return False
            match_index = found + 1
            for pid in touched:
                cleanup = bool(kill and ('<~Controls.644~>' in text or '<~Controls.662~>' in text) and origin == 'Current Action')
                boxes = ('Current FP Card', 'Current SA Card') if '<~Controls.662~>' in text else ('Current Action', 'Current FP Card', 'Current SA Card')
                current_boxes = {physical_zone(name) for name in boxes}
                nation = AUTH_ARMY_PIECES[pid][0]
                reserve = 'SA Reinforcements' if nation in ('SAURON', 'ISENGARD', 'SOUTHRON_EASTERLINGS') else 'FP Reinforcements'
                replacement = bool(swap_place and locations[pid] == physical_zone(reserve))
                if cleanup and locations[pid] not in current_boxes:
                    return False
                if locations[pid] != physical_zone(origin) and not (cleanup and locations[pid] in current_boxes) and not replacement:
                    return False
                if op == 'move':
                    locations[pid] = physical_zone(target)
                else:
                    nation = AUTH_ARMY_PIECES[pid][0]
                    # Controls296/304 share the client's kill handler:
                    # ordinary Shadow units return directly to reserve;
                    # Free Peoples units go to their casualty zone.
                    target = 'SA Reinforcements' if nation in ('SAURON', 'ISENGARD', 'SOUTHRON_EASTERLINGS') else 'FP Casualties'
                    locations[pid] = physical_zone(target)
        return all(loc == physical_zone(current['from']) for loc in locations.values())

    # Check each physical piece against its last reserve gesture. A
    # different piece between two uses cannot erase the first use.
    links = {}
    last_use = {}
    for next_i, _, _ in recruits:
        current = aligned[next_i]
        previous = {}
        for pid in current['ids']:
            if pid in last_use:
                previous.setdefault(last_use[pid], set()).add(pid)
        for prior_i, overlap in previous.items():
            prior = aligned[prior_i]
            if returned_to_reserve(prior_i, next_i, overlap):
                continue
            if prior['player'] != current['player'] or prior['from'] != current['from']:
                refuse('cross_player_recruit_identity', records[next_i].get('line'))
            if set(prior['ids']) != set(current['ids']):
                refuse('partially_overlapping_recruit_ids', records[next_i].get('line'))
            if (next_i != prior_i + 1 or current['line'] != prior['line'] + 1
                    or int(records[next_i]['line']) != int(records[prior_i]['line']) + 1):
                refuse('interposed_recruit_gesture', records[next_i].get('line'))
            links[prior_i] = next_i
        for pid in current['ids']:
            last_use[pid] = next_i
    result = []
    i = 0
    while i < len(records):
        run = [i]
        while run[-1] in links:
            run.append(links[run[-1]])
        if len(run) == 1:
            result.append(records[i])
            i += 1
            continue
        final = copy.deepcopy(records[run[-1]])
        final['_physical_recruit_run'] = {
            'version': 'aligned_contiguous_reserve_ids_v1',
            'ids': list(aligned[i]['ids']),
            'rendered_records': copy.deepcopy([records[j] for j in run]),
            'raw_records': [{k: v for k, v in aligned[j].items() if k != 'pieces'} for j in run],
            'final_destination': final['to']}
        result.append(final)
        i = run[-1] + 1
    return result
def parse_game(path, raw_path=None):
    """Parse a rendered log into batches with truth snapshots + RNG stream.

    Returns dict:
      'batches': [ {idx,turn,die_side,die_face,start_line,end_line,
                    records,kind,rng,truth,truth_before} ],
      'roll_chunks_num': [...], 'final_turn': int, 'gaps': [...]
    A batch spans one action-die staging up to (excluding) the next staging or
    turn boundary; truth = mirror snapshot AFTER the batch's consequences.
    """
    records, n_lines = _parse_all(path)
    _validate_client_undo_history(records, path, raw_path)
    # Resolve only histories admitted by the client snapshot guard above.
    net = resolve_undo_redo(records)
    _physical_recruit_raw = None
    def _transport_refuse(detail, line=None):
        raise PhysicalRecruitIdentityError({'detail': detail, 'line': line})
    _transport_path = _paired_raw_path(path, raw_path, _transport_refuse)
    if os.path.isfile(_transport_path):
        from army_transport_normalize import normalize as _normalize_army_transport
        import sys
        net, _physical_recruit_raw = _normalize_army_transport(
            net, records, read_raw_log_text(_transport_path).splitlines(), sys.modules[__name__])
        from physical_region_transport import normalize as normalize_region_transport
        net, _ = normalize_region_transport(net, records,
            read_raw_log_text(_transport_path).splitlines(), sys.modules[__name__])
    # Equal printed groups can contain distinct physical figures. Duplicate
    # movement is removed only by paired physical-ID transport normalization.
    dd = net
    net = _normalize_physical_recruit_runs(dd, path, raw_path, _raw_lines=_physical_recruit_raw)
    from rendered_pool_transport import mark_duplicates
    net = mark_duplicates(net, path, raw_path, raw_lines=_physical_recruit_raw)

    if any(record.get('op') in ('undo', 'redo') for record in records):
        from undo_board_origins import normalize as normalize_undo_origins
        def refuse_undo_raw(detail):
            raise ClientUndoHistoryError({'detail': detail})
        paired_raw = _paired_raw_path(path, raw_path, refuse_undo_raw)
        net = normalize_undo_origins(net, records, read_raw_log_text(paired_raw).splitlines(), sys.modules[__name__])

    mir = Mirror()
    batches = []
    gaps = []
    turn = 1
    cur = None          # current open batch
    pending_pre = []    # records before first die-stage of a turn (draws/roll)
    pending_truth = mir.snapshot()
    via_card = set()    # W4-018: (side, face) dice parked pool -> card zone
    guide_phase_start = None
    guide_phase_closed = False

    def apply_record(rec):
        before_comps = set(mir.comps)
        mir.apply(rec)
        if set(mir.comps) != before_comps:
            rec['_fellowship_composition_changed'] = True
        if rec.get('_guide_gesture'):
            rec['guide_truth_after'] = mir.snapshot()
            rec['guide_truth_after']['_rx_hunt_observation'] = (int(rec['line']) + 1, 'snapshot')

    def open_batch(die_side, die_face, line_no):
        return {'idx': len(batches), 'turn': turn, 'die_side': die_side,
                'die_face': die_face, 'start_line': line_no, 'end_line': None,
                'records': [], 'kind': None, 'rng': [], 'record_truth_version': 1,
                'truth_before': dict(mir.snapshot(), _rx_hunt_observation=(line_no, 'snapshot')),
                '_guide_phase_start': guide_phase_start}

    def close_batch(b, line_no):
        b['end_line'] = line_no
        b['kind'] = classify_batch(b)
        # rule-implicit political advance-on-attack (p.35): derive it from the
        # combat BEFORE the after-snapshot so the mirror stops lagging there
        if b['kind'] in ('combat', 'combat?'):
            mir.advance_on_attack(_battle_regions(b), b.get('die_side'),
                                  b.get('truth_before'),
                                  gestured=_political_gestured(b))
        # rule-implicit walk-in capture (amendment 1.55a): a settlement now
        # occupied by exactly one side belongs to that side - derive it at
        # the batch boundary (stable point) so the mirror stops lagging the
        # human's control-marker gesture (RB-08 marker-lag family)
        mir.enforce_settlement_control()
        b['truth'] = mir.snapshot()
        b['truth']['_rx_hunt_observation'] = (line_no, 'snapshot')
        batches.append(b)

    for rec in net:
        i = rec.get('line', 0)
        body = rec.get('body', '')
        op = rec['op']
        # Hand decisions can occur after a card's board effects but before a
        # later bonus draw in the same die batch. Preserve the actual board
        # immediately before those records; neither batch endpoint is safe.
        hand_zone = str(rec.get('from') if op == 'move' else rec.get('zone', '')).lower()
        hand_exit = op in ('move', 'kill') and hand_zone in (
            'free peoples hand', 'free peoples hand (chosen)',
            'shadow hand', 'shadow hand (chosen)')
        card_draw = op == 'draw' and _draw_kind(rec.get('what', ''), cur['records'] if cur is not None else pending_pre)[0] == 'deck'
        card_cleanup = op == 'kill' and any(
            (piece := parse_piece(str(token))) and piece[0] == 'card'
            for token in rec.get('pieces', []))
        ring_counter = op in ('move', 'kill') and any(
            str(token).strip().lower() in ('narya', 'nenya', 'vilya', 'dwarven ring')
            for token in ([rec.get('obj', '')] if op == 'move'
                          else rec.get('pieces', [])))
        legacy_ring_prefix = (op == 'move'
            and 'free control marker' in str(rec.get('obj', '')).lower()
            and any(str(rec.get(zone, '')).strip().lower() == 'fp dice'
                    for zone in ('from', 'to')))
        guide_gesture = op == 'move' and any(
            str(rec.get(key, '')).strip().lower() == 'guide of the fellowship'
            for key in ('from', 'to'))
        fellowship_gesture = (op in ('move', 'flip') and
                              str(rec.get('obj', '')).strip().lower()
                              in ('the fellowship', 'fsp'))
        eye_move = (op == 'move'
                    and str(rec.get('obj', '')).lower() == 'shadow action die (e)'
                    and str(rec.get('from', '')).lower() == 'sa dice'
                    and str(rec.get('to', '')).lower() == 'the hunt for the ring')
        phase_close = (cur is None and not guide_phase_closed
                       and (op in ('alloc_eye', 'roll_dice') or eye_move))
        if guide_gesture:
            rec['_guide_gesture'] = True
        if phase_close:
            rec['_guide_phase_close'] = True
            guide_phase_closed = True
        if hand_exit or card_draw or card_cleanup or ring_counter or legacy_ring_prefix or guide_gesture or phase_close or fellowship_gesture:
            rec['decision_truth'] = mir.snapshot()
            rec['decision_truth']['_rx_hunt_observation'] = (i, 'snapshot')
        if op == 'turn':
            if cur is not None:
                close_batch(cur, i)
                cur = None
            turn = rec['turn']
            pending_pre = []
            pending_truth = mir.snapshot()
            via_card.clear()
            guide_phase_start = mir.snapshot()
            guide_phase_start['_rx_hunt_observation'] = (i + 1, 'snapshot')
            guide_phase_closed = False
            continue
        # undo/redo already resolved away by resolve_undo_redo
        # W4-018 via-card staging: pool -> Current X Card -> Current Action.
        # The first drag parks the die (not a boundary); the second is the
        # staging of THAT die and opens its own batch.
        if op == 'move' and not rec.get('die_stage'):
            mvc = _DIE_TO_CARD_RE.match(body)
            if mvc:
                via_card.add(('SHADOW' if mvc.group(1).lower() == 'shadow'
                              else 'FREE_PEOPLES', mvc.group(2).upper()))
            else:
                mvs = _DIE_VIA_CARD_STAGE_RE.match(body)
                if mvs:
                    key = ('SHADOW' if mvs.group(1).lower() == 'shadow'
                           else 'FREE_PEOPLES', mvs.group(2).upper())
                    if key in via_card:
                        via_card.discard(key)
                        rec['die_stage'] = key
        if op == 'move' and rec.get('die_stage'):
            via_card.discard(rec['die_stage'])
        # a new action-die staged -> boundary
        if op == 'move' and rec.get('die_stage'):
            if cur is not None:
                close_batch(cur, i)
            cur = open_batch(rec['die_stage'][0], rec['die_stage'][1], i)
            cur['records'].extend(pending_pre)
            pending_pre = []
            cur['records'].append(rec)
            apply_record(rec)
            continue
        # RNG records
        if op in ('roll_num', 'roll_dice', 'draw', 'add_eye', 'alloc_eye'):
            tgt = cur if cur is not None else None
            if op == 'roll_num':
                item = {'kind': 'roll_num', 'player': rec['player'],
                        'ints': rec['ints'], 'line': i}
            elif op == 'roll_dice':
                item = {'kind': 'action_roll', 'player': rec['player'],
                        'faces': rec['faces'], 'line': i}
            elif op == 'draw':
                dk = _draw_kind(rec['what'], cur['records'] if cur is not None else pending_pre)
                item = {'kind': 'draw_' + dk[0], 'value': dk[1],
                        'player': rec['player'], 'line': i}
            elif op == 'add_eye':
                item = {'kind': 'add_eye', 'n': rec['n'], 'line': i}
            else:
                item = {'kind': 'alloc_eye', 'line': i}
            if tgt is not None:
                tgt['rng'].append(item)
                tgt['records'].append(rec)
            else:
                pending_pre.append(rec)
            apply_record(rec)
            continue
        # ordinary board record
        if cur is not None:
            cur['records'].append(rec)
        else:
            pending_pre.append(rec)
        if rec['op'] in ('move', 'kill', 'flip'):
            _census_gaps(rec, i, body, gaps)
            apply_record(rec)

    if cur is not None:
        close_batch(cur, n_lines + 1)
    elif pending_pre:
        # EOF can follow a completed free decision or the next turn's setup.
        # Preserve its literal records without fabricating a paid die stage.
        start = pending_pre[0]['line']
        tail = {'idx': len(batches), 'turn': turn, 'die_side': None,
                'die_face': None, 'start_line': start, 'end_line': n_lines + 1,
                'records': pending_pre, 'kind': 'phase_tail', 'rng': [],
                'record_truth_version': 1, '_phase_only': True,
                'truth_before': dict(pending_truth, _rx_hunt_observation=(start, 'snapshot')),
                'truth': dict(mir.snapshot(), _rx_hunt_observation=(n_lines + 1, 'snapshot')),
                '_guide_phase_start': guide_phase_start}
        for rec in pending_pre:
            if rec['op'] == 'roll_dice':
                tail['rng'].append({'kind': 'action_roll', 'player': rec['player'],
                                    'faces': rec['faces'], 'line': rec['line']})
        batches.append(tail)

    return {'batches': batches, 'final_turn': turn, 'gaps': gaps,
            'n_lines': n_lines, 'net_records': net}


_FIGURE_TYPES = ('unit', 'nazgul', 'leader', 'named')


def _figure_keys(tokens):
    """(type, nation-or-name, kind) of every figure token in `tokens`
    (count dropped) - the identity a later off-board kill is matched on."""
    keys = []
    for tok in tokens:
        p = parse_piece(tok)
        if p and p[0] in _FIGURE_TYPES:
            keys.append((p[0], p[1], p[2]))
    return keys


def _offboard_casualty_regions(b):
    """W2-030: regions whose figures LEFT THE BOARD inside die-batch b
    through the client's off-board casualty dialects - board casualties
    that never render as 'kills <figures> in <region>':
      (a) a figure MOVED from a region / Stronghold box straight into a
          Casualties box ('[3]Sauron Regular from Folde to SA Casualties',
          2023-12-25 L808) or - Shadow casualties return to the pool -
          into a Reinforcements box ('[2]Sauron Regular from North
          Ithilien to SA Reinforcements', 2024-10-18 L189);
      (b) a figure MOVED from a region / Stronghold box onto a Current
          FP/SA Card slot (or the Current Action slot) and KILLED off-board
          later in the same batch ('Rohan Regular from Fords of Isen to
          Current FP Card' + 'kills group (Shadow Action Die (C), Rohan
          Regular) in Current Action', 2024-10-18 L625/L627).
    Both are the very board event the canonical in-region kill renders
    (the mirror's _move_piece / apply_kill already remove the figure);
    classify_batch and _battle_regions read them here so a battle fought
    in either dialect classifies 'combat' and names its defender region
    instead of falling to the zero-kill family with the dead defenders
    still on the board.  A figure killed off-board WITHOUT an in-batch exit
    from a region (a hunt casualty leaving the Guide / Fellowship boxes,
    2023-01-20 L60-61) is not a board casualty and is ignored; so is a
    non-figure drag to Casualties (rings, dice, control markers).  A HUNT
    batch (tile drawn, or the Fellowship / FSP moved) fights no battle -
    its figure drags are hunt bookkeeping or self-corrections (2023-09-18
    L115/L124: Gandalf the White placed, refused, dragged back to the
    Casualties box before the hunt roll) - so it yields nothing here; a
    NAMED figure (companion / minion) dragged to Casualties on its own is
    not read as a battle casualty either (no 24-game instance is one); and
    a die-batch whose die owner never rolled fought no battle (2023-03-16
    L1014: a Stormcrow play whose only roll is the victim's - the
    canonical kill+roll rule would label it 'combat' and the combat family,
    finding no event play, fabricates a siege attack), so the dialect
    yields nothing there either - fail closed, never a wrong action."""
    if any(c.get('kind') == 'draw_tile' for c in (b.get('rng') or [])):
        return set()
    for r in b['records']:
        if r.get('op') == 'move' and str(r.get('obj', '') or '').strip(
                ).lower() in ('the fellowship', 'fsp'):
            return set()
    owner = next((r.get('player') for r in b['records']
                  if r.get('die_stage')), None)
    if owner is not None and not any(
            c.get('kind') == 'roll_num' and c.get('player') == owner
            for c in (b.get('rng') or [])):
        return set()
    regs = set()
    parked = []                       # (line, region, figure key)
    for r in b['records']:
        op = r.get('op')
        if op == 'move':
            fk = zone_kind(r.get('from', '') or '')
            if fk[0] not in ('region', 'stronghold'):
                continue
            tk = zone_kind(r.get('to', '') or '')
            if tk[0] not in ('cas', 'reinf', 'card', 'dice'):
                continue
            obj = str(r.get('obj', '') or '').strip()
            mg = re.match(r'^group\s*\((.*)\)$', obj, re.I)
            keys = _figure_keys(split_group(mg.group(1)) if mg else [obj])
            if not keys:
                continue
            if tk[0] in ('cas', 'reinf'):
                if any(k[0] != 'named' for k in keys):
                    regs.add(fk[1])
            else:
                ln = int(r.get('line', 0) or 0)
                parked.extend((ln, fk[1], k) for k in keys)
        elif op == 'kill':
            zk = zone_kind(r.get('zone', '') or '')
            if zk[0] not in ('card', 'dice'):
                continue
            ln = int(r.get('line', 0) or 0)
            for k in _figure_keys(r.get('pieces') or []):
                for (pl, preg, pk) in parked:
                    if pl < ln and pk == k:
                        regs.add(preg)
    return regs


def _battle_regions(b):
    """Region(s) under attack in a combat die-batch = regions carrying a
    figure-kill (units/leaders), in the canonical 'kills ... in <region>'
    rendering OR an off-board casualty dialect (W2-030,
    _offboard_casualty_regions).  The defender's region is the p.35
    advance-on-attack site; advance_on_attack's non-attacker filter drops the
    attacker's own casualties in a co-located siege."""
    regs = set()
    for r in b['records']:
        if r['op'] != 'kill':
            continue
        zk = zone_kind(r.get('zone', ''))
        if zk[0] not in ('region', 'stronghold'):
            continue
        for tok in r['pieces']:
            p = parse_piece(tok)
            if p and p[0] in ('unit', 'nazgul', 'leader', 'named'):
                regs.add(zk[1])
                break
    regs |= _offboard_casualty_regions(b)
    return regs


def _political_gestured(b):
    """Nations whose political marker the human MOVED inside die-batch b
    (W4-009: such a gesture is the p.35 advance itself, so advance_on_attack
    must not derive it a second time).  Flips are activations, not moves."""
    nats = set()
    for r in b['records']:
        if r.get('op') != 'move':
            continue
        if zone_kind(r.get('to', ''))[0] != 'political':
            continue
        obj = str(r.get('obj', '')).strip()
        mg = re.match(r'^group\s*\((.*)\)$', obj, re.I)
        for tok in (split_group(mg.group(1)) if mg else [obj]):
            tok = tok.strip()
            p = parse_piece(tok)
            if p and p[0] in ('nation', 'leader', 'unit'):
                nats.add(p[1])
            elif tok.lower() in NATION_NAME:
                nats.add(NATION_NAME[tok.lower()])
    return nats


_HANDLED_OBJ = ('the fellowship', 'fsp', 'corruption')


def _census_gaps(rec, line_no, body, gaps):
    """Flag only genuinely-unhandled pieces that COULD carry a board effect
    (a move/kill touching a region whose piece token parse_piece can't
    classify).  Off-board card/die/ring plays and marker moves are handled
    no-ops, not gaps."""
    if rec['op'] == 'move':
        ol = rec['obj'].strip().lower()
        if ol in _HANDLED_OBJ or ol == 'nothing' or \
                ol.endswith('control marker') or \
                ol.startswith('victory points marker'):
            return
        # mis-split card play (title tail leaked into a from/to zone)
        if _missplit_card_zone(rec.get('from', '')) or \
                _missplit_card_zone(rec.get('to', '')):
            return
        fk = zone_kind(rec['from'])[0]
        tk = zone_kind(rec['to'])[0]
        if 'region' not in (fk, tk) and 'stronghold' not in (fk, tk):
            return  # purely off-board (card play, ring, die, political)
        mg = re.match(r'^group\s*\((.*)\)$', rec['obj'], re.I)
        toks = split_group(mg.group(1)) if mg else [rec['obj']]
        for t in toks:
            if t.strip().lower() == 'the fellowship':
                continue
            p = parse_piece(t)
            if p and p[0] == 'unknown':
                gaps.append((line_no, 'move? ' + body[:90]))
                return
    elif rec['op'] == 'kill':
        if _missplit_card_zone(rec.get('zone', '')):
            return
        if zone_kind(rec['zone'])[0] not in ('region', 'stronghold'):
            return
        for t in rec['pieces']:
            p = parse_piece(t)
            if p and p[0] == 'unknown':
                gaps.append((line_no, 'kill? ' + body[:90]))
                return


def classify_batch(b):
    """Heuristic label from a batch's records (for the census driver)."""
    recs = b['records']
    has_num_roll = any(r.get('kind') == 'roll_num' for r in b['rng'])
    has_fel_move = False
    has_tile = any(r.get('kind') == 'draw_tile' for r in b['rng'])
    has_muster = False
    has_political = False
    kill_in_region = False
    for r in recs:
        if r['op'] == 'kill':
            zk = zone_kind(r.get('zone', ''))
            if zk[0] in ('region', 'stronghold'):
                # only count figure kills (units/leaders), not tile/die/card
                for tok in r['pieces']:
                    p = parse_piece(tok)
                    if p and p[0] in ('unit', 'nazgul', 'leader', 'named'):
                        kill_in_region = True
        if r['op'] == 'move':
            ol = r['obj'].strip().lower()
            if ol == 'the fellowship' or ol == 'fsp' or ol == 'corruption':
                has_fel_move = True
            zk = zone_kind(r.get('from', ''))
            if zk[0] == 'reinf':
                has_muster = True
            if zone_kind(r.get('to', ''))[0] == 'political' or \
                    zone_kind(r.get('from', ''))[0] == 'political':
                has_political = True
        if r['op'] == 'flip':
            if 'political' in (r.get('zone', '') or '').lower():
                has_political = True
    # W2-030: a casualty dragged to a Casualties box, or parked on a
    # Current card slot and killed there, is the same board kill as the
    # canonical 'kills ... in <region>' line (2023-12-25 L808, 2024-10-18
    # L625/L627: rolled battles that classified other/political/muster
    # and lost the whole combat family)
    if not kill_in_region and _offboard_casualty_regions(b):
        kill_in_region = True
    if kill_in_region and has_num_roll:
        return 'combat'
    if has_tile or (has_fel_move and has_num_roll):
        return 'hunt'
    # A kill-in-region with NO combat roll but WITH a muster (reinf -> region)
    # is the elite-UPGRADE muster (kill N regulars, muster N elites into the
    # same region - e.g. Isengard at Orthanc), a pure recruitment gesture with
    # no combat resolver and no RNG, resolved faithfully by the mirror from the
    # log.  It is NOT a combat missing its roll (a real combat always rolls in
    # its own die-batch), so classify it a muster (bookkeeping), not combat?.
    # A no-roll kill-in-region is likewise NEVER a dice combat; besides the
    # muster-upgrade it is either a CARD-EFFECT elimination (an event card
    # played this batch removes units/leaders - Stormcrow, Denethor's Folly) or
    # a STACKING-OVERFLOW trim (rb p.27 region cap 10 / p.31 siege cap 5: excess
    # units removed when a friendly army moves into the region or a garrison
    # withdraws into a besieged stronghold).  Both are faithful bookkeeping the
    # mirror applies from the log, so do NOT route them to the combat resolver.
    # Only a no-roll kill with NEITHER a card nor a friendly move-in stays a
    # suspicious 'combat?' (a battle that truly lost its dice - a real gap).
    if kill_in_region and not has_muster:
        _card = any(
            r['op'] == 'move' and '/' in str(r.get('obj', ''))
            and ('card' in str(r.get('to', '')).lower()
                 or 'board' in str(r.get('to', '')).lower())
            for r in recs)
        _movein = any(
            r['op'] == 'move'
            and zone_kind(r.get('to', ''))[0] in ('region', 'stronghold')
            and zone_kind(r.get('from', ''))[0] in ('region', 'stronghold',
                                                     'reinf')
            for r in recs)
        if _card:
            return 'card'
        if _movein:
            return 'overstack'
        return 'combat?'   # casualties without a clear roll chunk
    if has_muster:
        return 'muster'
    if has_fel_move:
        return 'fellowship'
    if has_political:
        return 'political'
    return 'other'


if __name__ == '__main__':
    import glob
    args = sys.argv[1:]
    if args and args[0] == '--survey':
        D = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'data', 'rendered_logs')
        files = sorted(glob.glob(os.path.join(D, 'GAME*.rendered.txt')))
        files = [f for f in files if 'branstonpickle_Compote.rendered'
                 not in f]
        for f in files:
            g = parse_game(f)
            from collections import Counter
            kinds = Counter(b['kind'] for b in g['batches'])
            print('%-30s batches=%3d turns=%2d gaps=%3d %s'
                  % (os.path.basename(f)[:30], len(g['batches']),
                     g['final_turn'], len(g['gaps']), dict(kinds)))
    else:
        path = args[0]
        g = parse_game(path)
        print('batches', len(g['batches']), 'turns', g['final_turn'],
              'gaps', len(g['gaps']))
        for b in g['batches'][:40]:
            print('#%3d T%-2d %-13s die=%s(%s) L%d-%s recs=%d rng=%d'
                  % (b['idx'], b['turn'], b['kind'],
                     (b['die_side'] or '?')[:2], b['die_face'],
                     b['start_line'], b['end_line'], len(b['records']),
                     len(b['rng'])))
