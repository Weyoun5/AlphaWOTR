"""RENDERED-LOG replay parser + truth mirror  (rendered-log parity census).

The CLIENT-rendered replay logs (data/rendered_logs/GAMEnn_*.rendered.txt) are
the authoritative resolution of every VASSAL gesture: the app already collapsed
drag jitter, multi-hop retreats, stale sources and undo chains into readable
sentences.  This module parses those sentences into

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
        return ('stronghold', sanitize(m.group(1)))  # besieged garrison box
    # otherwise a board region
    return ('region', sanitize(z))


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
               'comps': sorted(self.comps)}
        ctrl = {'FREE_PEOPLES': sorted(self.ctrl['FREE_PEOPLES']),
                'SHADOW': sorted(self.ctrl['SHADOW'])}
        return {'armies': armies, 'pol': pol, 'fel': fel,
                'fpc': {n: dict(v) for n, v in self.fpc.items()},
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
            elif tk in ('cas', 'reinf'):
                if a in self._fp_nations():
                    self._record_fp_cas(a, b.lower(), cnt)
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

    def advance_on_attack(self, battle_regions, attacker_side, before_snap):
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

    def _move_named(self, name, from_kind, to_kind):
        fk, fmeta = from_kind
        tk, tmeta = to_kind
        nl = name.lower()
        # leaving/entering the fellowship changes the companion set
        if fk == 'fellowship' and tk == 'fellowship':
            return  # guide<->company shuffle, no set change
        if fk == 'fellowship' and tk != 'fellowship':
            self.comps.discard(COMP_ALIAS.get(nl, name))
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

    def apply_move(self, rec):
        obj = rec['obj']
        # a mis-split card play (unlisted title with an internal ' from '/
        # ' in ') leaks the title tail into from/to; off-board no-op
        if _missplit_card_zone(rec.get('from', '')) or \
                _missplit_card_zone(rec.get('to', '')):
            return
        fk = zone_kind(rec['from'])
        tk = zone_kind(rec['to'])
        # fellowship pawn
        ol = obj.strip().lower()
        # VASSAL 'moves nothing from X to Y' = empty retreat-into-siege gesture
        if ol == 'nothing':
            return
        if ol == 'the fellowship':
            if tk[0] == 'region':
                self.fel_region = tk[1]
                # v32 declare-calibration fix (2026-08-03): a pawn
                # region move alone NEVER reveals.  Per spec 1.216
                # "every reveal must have a source" — the sourced
                # 'Previously on Fellowship Track N' flip is the only
                # reveal evidence (apply_flip).  A DECLARATION moves
                # the pawn and keeps the Fellowship HIDDEN; the old
                # `fel_hidden = False` here marked every no-flip
                # declare as a reveal, so the mirror reported
                # hidden=False for turns afterwards (census: 85% of
                # post-noflip-declare turn starts) and the v29 decline
                # emission silently dropped those windows at its
                # hidden gate -> corpus decline:declare collapsed to
                # ~50:50 vs the real human conditional.
            # USER truth-table fix (game01_verification): a Fellowship
            # move whose DESTINATION is a 'Mordor N - ...' zone is the
            # Mordor entry / a track step - the mirror never set
            # fel_mordor on the MOVE gesture (only on flips), so
            # truth.mordor stayed False for whole Mordor turns.
            _mm = re.match(r'^\s*Mordor\s+(\d+)\s*-', str(rec.get('to')
                                                          or ''))
            if _mm:
                _was_mordor9 = self.fel_mordor
                self.fel_mordor = True
                self.fel_step = int(_mm.group(1))
                if not _was_mordor9:
                    # Mordor ENTRY re-hides (mirrors apply_flip's
                    # mordor branch; the old test read fel_mordor
                    # AFTER setting it and never fired)
                    self.fel_hidden = True
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
            p = parse_piece(obj)
            nat = None
            if p and p[0] in ('nation', 'leader', 'unit'):
                nat = p[1]
            elif ol in NATION_NAME:
                nat = NATION_NAME[ol]
            if nat and tk[0] == 'political' and tk[1] is not None:
                gbox = 3 - tk[1]               # vassal->internal
                self.pol_gesture[nat] = gbox
                # monotonic: the marker never retreats; keep the rule-true box
                # at least as advanced as the gesture
                self.pol[nat][0] = max(self.pol[nat][0], gbox)
                if self.pol[nat][0] <= gbox and \
                        self.pol_gesture_active.get(nat, False) >= \
                        self.pol[nat][1]:
                    self.pol_derived.discard(nat)
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
                        # v32 declare-calibration fix (2026-08-03):
                        # moves never reveal — sourced flips only
                        # (see the single-pawn handler above).
                    # Mordor entry via GROUP move: same destination
                    # handling as the single-pawn branch (GAME11 L2416
                    # left truth.mordor False for the entry batch)
                    _mmg9 = re.match(r'^\s*Mordor\s+(\d+)\s*-',
                                     str(rec.get('to') or ''))
                    if _mmg9:
                        _was_mordor9 = self.fel_mordor
                        self.fel_mordor = True
                        self.fel_step = int(_mmg9.group(1))
                        if not _was_mordor9:
                            self.fel_hidden = True
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

    def apply_flip(self, rec):
        obj = rec['obj'].strip()
        ol = obj.lower()
        if ol == 'the fellowship':
            if rec.get('mordor') is not None:
                self.fel_mordor = True
                self.fel_step = rec['mordor']
                self.fel_hidden = True
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
            rk = sanitize(reg)
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
            rk = sanitize(reg)
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


def parse_line(raw):
    """Parse ONE rendered line (leading '%' already stripped) into a record
    dict, or None for pure chat/noise.  Records carry op + fields."""
    s = raw.strip()
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




# RECONSTRUCTED (2026-07-27): the stale-sweep deletion of the dead
# invert_for_undo also ate these two module constants (net-resolution
# 1.216).  Rebuilt from resolve_undo_redo's contract: 'turn' markers
# are preserved in place (undo never pops them); any REAL action
# appended after an undo clears the redo stack.
# FIX (full-pass review 2026-07-27): the first reconstruction carried a
# phantom 'roll' token (real ops are 'roll_num'/'roll_dice') and missed
# 'flip' — an undo -> roll/flip -> redo sequence could resurrect an
# already-undone record into the settled stream.
_UR_STICKY = {'turn'}
# 'other' included: unclassified REAL gestures (e.g. elven-ring die
# changes) must clear the redo stack too — any real action after an
# undo invalidates the redo chain
_UR_CLEARS = {'move', 'draw', 'roll_num', 'roll_dice', 'flip', 'kill',
              'add_eye', 'alloc_eye', 'choose', 'other'}


def resolve_undo_redo(records):
    """Collapse a parsed line-record stream (in file order, incl. 'turn',
    'undo', 'redo') into the NET linear action sequence.  Undone-and-not-redone
    actions are removed; redone actions are restored.  Turn markers preserved
    in place.  A stream with no undo/redo returns unchanged (byte-identical)."""
    net = []
    redo_stack = []                 # LIFO of (orig_index, record)
    for rec in records:
        op = rec.get('op')
        if op == 'undo':
            for j in range(len(net) - 1, -1, -1):
                if net[j].get('op') not in _UR_STICKY:
                    redo_stack.append((j, net.pop(j)))
                    break
            continue
        if op == 'redo':
            if redo_stack:
                j, r = redo_stack.pop()
                net.insert(min(j, len(net)), r)
            continue
        net.append(rec)
        if op in _UR_CLEARS:
            redo_stack.clear()
    return net


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
def _draw_kind(what):
    w = what.strip()
    wl = w.lower()
    if wl in ('free character card', 'free strategy card',
              'shadow character card', 'shadow strategy card'):
        return ('deck', w)
    if wl == 'eye' or re.match(r'^\d+r?$', wl) or wl in (
            "shelob's lair", 'elven rope'):
        return ('tile', w)
    # A random COMPANION pull always NAMES a real Companion (Foul Thing / a
    # hunt casualty).  Any other drawn title is a CARD (deck) - e.g. a Shadow
    # Character card 'Give it to Uss!' whose draw the old fall-through mislabeled
    # a companion pull, minting a phantom draw_companion that wrongly tripped the
    # 'companion-stays' hunt skip (ppoma b77: Isildur's Bane Mordor tile -> the
    # tile-hunt driver never ran).  Classify as companion ONLY when the name is
    # a recognised Companion; otherwise it is a deck/card draw.
    if wl in COMP_ALIAS:
        return ('companion', w)
    return ('deck', w)


def parse_game(path):
    """Parse a rendered log into batches with truth snapshots + RNG stream.

    Returns dict:
      'batches': [ {idx,turn,die_side,die_face,start_line,end_line,
                    records,kind,rng,truth,truth_before} ],
      'roll_chunks_num': [...], 'final_turn': int, 'gaps': [...]
    A batch spans one action-die staging up to (excluding) the next staging or
    turn boundary; truth = mirror snapshot AFTER the batch's consequences.
    """
    records, n_lines = _parse_all(path)
    # UNDO/REDO net-resolution (1.216): collapse to the settled action stream
    # BEFORE segmentation + mirror, so undone gestures never touch the board and
    # the snapshots reflect the net result (redo games resolve, not quarantine).
    net = resolve_undo_redo(records)
    # CLIENT RENDER-DUP dedup (session-35c methodology, applied to the
    # mirror 2026-07-24): the chat pane occasionally renders the SAME
    # move gesture twice on consecutive lines (GAME01 T3 L288-289:
    # '[2]Nazgul, Sauron Elite, [9]Sauron Regular Osgiliath->Minas
    # Tirith' x2 -> mirror held 37>36 Sauron regulars -> gate6
    # quarantined the user's own reference game).  Dedup applies to
    # MOVE lines ONLY (kill lines legitimately repeat: per-figure
    # casualties).  Two identical consecutive MOVES of the same
    # multi-piece group from the same source are physically impossible
    # (the pieces already left); identical single-piece repeats ARE
    # possible for fungible pools (Reinforcements->region recruits),
    # so dedup only when the source is a BOARD REGION and the move
    # names >1 piece or a named/leader figure.
    dd = []
    for r in net:
        if (dd and r.get('op') == 'move'
                and dd[-1].get('op') == 'move'
                and r.get('body') == dd[-1].get('body')
                and int(r.get('line', 0)) == int(dd[-1].get('line', -9)) + 1):
            src = sanitize(str(r.get('from', '') or ''))
            obj = str(r.get('obj', '') or '')
            # dedup criterion: >1 piece OR any NON-fungible figure — a
            # named companion/minion cannot legitimately repeat an
            # identical consecutive move (GAME21 double-rendered Merry
            # minted a second Meriadoc leader)
            _pp9 = parse_piece(obj)
            multi = (obj.startswith('group ')
                     or (_pp9 is not None
                         and _pp9[0] in ('named', 'nazgul', 'leader')))
            if src in board.regions and multi:
                continue
        dd.append(r)
    net = dd

    mir = Mirror()
    batches = []
    gaps = []
    turn = 1
    cur = None          # current open batch
    pending_pre = []    # records before first die-stage of a turn (draws/roll)

    def open_batch(die_side, die_face, line_no):
        return {'idx': len(batches), 'turn': turn, 'die_side': die_side,
                'die_face': die_face, 'start_line': line_no, 'end_line': None,
                'records': [], 'kind': None, 'rng': [],
                'truth_before': mir.snapshot()}

    def close_batch(b, line_no):
        b['end_line'] = line_no
        b['kind'] = classify_batch(b)
        # rule-implicit political advance-on-attack (p.35): derive it from the
        # combat BEFORE the after-snapshot so the mirror stops lagging there
        if b['kind'] in ('combat', 'combat?'):
            mir.advance_on_attack(_battle_regions(b), b.get('die_side'),
                                  b.get('truth_before'))
        # rule-implicit walk-in capture (amendment 1.55a): a settlement now
        # occupied by exactly one side belongs to that side - derive it at
        # the batch boundary (stable point) so the mirror stops lagging the
        # human's control-marker gesture (RB-08 marker-lag family)
        mir.enforce_settlement_control()
        b['truth'] = mir.snapshot()
        batches.append(b)

    for rec in net:
        i = rec.get('line', 0)
        body = rec.get('body', '')
        op = rec['op']
        if op == 'turn':
            if cur is not None:
                close_batch(cur, i)
                cur = None
            turn = rec['turn']
            pending_pre = []
            continue
        # undo/redo already resolved away by resolve_undo_redo
        # a new action-die staged -> boundary
        if op == 'move' and rec.get('die_stage'):
            if cur is not None:
                close_batch(cur, i)
            cur = open_batch(rec['die_stage'][0], rec['die_stage'][1], i)
            cur['records'].extend(pending_pre)
            pending_pre = []
            cur['records'].append(rec)
            mir.apply(rec)
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
                dk = _draw_kind(rec['what'])
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
            mir.apply(rec)
            continue
        # ordinary board record
        if cur is not None:
            cur['records'].append(rec)
        else:
            pending_pre.append(rec)
        if rec['op'] in ('move', 'kill', 'flip'):
            _census_gaps(rec, i, body, gaps)
            mir.apply(rec)

    if cur is not None:
        close_batch(cur, n_lines + 1)

    return {'batches': batches, 'final_turn': turn, 'gaps': gaps,
            'n_lines': n_lines, 'net_records': net}


def _battle_regions(b):
    """Region(s) under attack in a combat die-batch = regions carrying a
    figure-kill (units/leaders).  The defender's region is the p.35
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
    return regs


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
