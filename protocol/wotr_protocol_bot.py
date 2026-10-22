# -*- coding: ascii -*-
"""wotr_protocol_bot.py -- AlphaWOTR as a client of the WotR Bot Protocol.

The WotR Bot Protocol (draft 0.1, github.com/behindmedia/wotr-protocol)
lets War of the Ring bots play each other through a REFEREE: one rules
engine runs the game, rolls every die, draws every card and tile, and at
each decision sends the deciding bot its view of the position and a
closed list of legal options.  The bot answers with one option.

This program puts AlphaWOTR in the bot's seat.  It adds NO rules of its
own: every rule is the engine's (ai.get_legal_actions, ai._execute_action,
the battle and hunt machinery) and every choice is the engine's search
(ai.mcts_action_with_policy with the packaged corpus_live nets and the
EngineSession defaults: 300 simulations, temperature 0.02 -- 0 for the
Fellowship phase and the eye allocation --, no noise, the per-kind
budget of engine_play._sims_for_kind).

One decision of the referee:

  1. the engine's GameState is rebuilt from the referee's VIEW (the view
     is the authority, as the protocol says; the event stream fills in
     two per-turn flags the view does not carry);
  2. the engine's pending decision is set up to match the question (the
     battle phase and its hits, the hunt damage, the hand-limit
     discard, the declare phase ...);
  3. the engine's menu (get_legal_actions) is matched against the
     referee's options; actions without a counterpart are kept out of
     the search, so the engine picks only among actions the referee
     allows;
  4. the search picks an action, which becomes the referee's option.

The referee asks in ATOMIC steps (an Army move: the die, then the Army
and its destination, then the units if not all of them); the engine's
actions are compound.  An engine action therefore becomes a PLAN -- the
option for the first question and the answers for the ones that follow,
read off a copy of the engine's state with the action applied.  Chains
(the Army die's second move, the Muster die's second recruit, further
Companion groups and Nazgul flights, a card's own choices) go on on
that copy, the engine's search answering each link.

When nothing of the engine's matches (a map border only one engine has,
a rule the two engines read differently, a question the engine never
asks), a plain rule answers; the fallback is counted and logged.  The
answer is always one of the referee's options.

Usage (from the repository root; Python 3.10+ with requirements.txt):

  As the referee's child process (stdin and stdout carry the protocol,
  stderr is the log):
      python -I protocol/wotr_protocol_bot.py

  Dialling a referee's WebSocket with a seat token:
      python -I protocol/wotr_protocol_bot.py \\
          --connect wss://wotr.behindmedia.com/referee --token TOKEN

Options: --sims N (search budget, default 300; also a protocol option
the referee may set), --max-seconds S (cap per decision, besides the
referee's clock), --verbose (each decision on stderr), --log FILE (one
JSON line per decision).

MIT License (see LICENSE).  The WebSocket client is adapted from the
protocol repository's example bot (examples/python/random_bot.py, MIT).
"""

import argparse
import base64
import copy
import json
import os
import random
import socket
import ssl
import struct
import sys
import time
import traceback
import unicodedata
from urllib.parse import urlparse

PROTOCOL = 'wotr-bot/0.1'
BOT_NAME = 'AlphaWOTR'
BOT_VERSION = 'corpus_live-v52/protocol-1'

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.join(os.path.dirname(HERE), 'src')


def log(*args):
    print(*args, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------
# Engine import: the launchers' certified-path environment (CPU only,
# the v52 action generation the corpus_live nets need)
# ---------------------------------------------------------------------

board = gs = gl = ai = engine_play = cards_mod = None
FP = SH = None
BOARD = None


def import_engine(src):
    global board, gs, gl, ai, engine_play, cards_mod, FP, SH, BOARD
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    # as wotr_bridge/launch_head.py's HEAD_ENV (the corpus_live nets' generation), the search in Python
    os.environ.setdefault('WOTR_ACT_GEN', 'v52')
    os.environ.setdefault('WOTR_REWARD_CONTRACT', 'predecision_profile_v1')
    os.environ['WOTR_NATIVE'] = '0'
    sys.path.insert(0, os.path.abspath(src))
    import board as _board          # chdirs into src
    import game_state as _gs
    import game_logic as _gl
    import ai as _ai
    import engine_play as _ep
    import cards as _cards
    board, gs, gl, ai, engine_play, cards_mod = (
        _board, _gs, _gl, _ai, _ep, _cards)
    FP = gs.Side.FREE_PEOPLES
    SH = gs.Side.SHADOW
    BOARD = board.load_board()


# ---------------------------------------------------------------------
# The protocol's vocabulary (vocabulary/vocabulary-0.1.json of the
# protocol repository, MIT): region keys, card keys with their printed
# titles (accents folded).  Matched against the engine's own catalog at
# start, so a name that does not match fails at once, not in a game.
# ---------------------------------------------------------------------

REGION_KEYS = """
    forlindon ered-luin north-ered-luin evendim harlindon grey-havens
    tower-hills south-ered-luin the-shire arnor north-downs buckland
    bree ettenmoors weather-hills south-downs cardolan old-forest
    minhiriath tharbad enedwaith north-dunland south-dunland gap-of-rohan
    orthanc druwaith-iaur andrast trollshaws hollin fords-of-bruinen
    high-pass goblins-gate old-ford gladden-fields dimrill-dale
    eagles-eyrie angmar mount-gram mount-gundabad lorien rivendell moria
    parth-celebrant fangorn eastemnet westemnet folde edoras helms-deep
    fords-of-isen anfalas erech lamedon dol-amroth pelargir lossarnach
    minas-tirith druadan-forest osgiliath dead-marshes western-emyn-muil
    eastern-emyn-muil north-ithilien south-ithilien west-harondor
    east-harondor western-brown-lands eastern-brown-lands
    south-anduin-vale north-anduin-vale carrock rhosgobel
    northern-mirkwood withered-heath western-mirkwood woodland-realm
    old-forest-road narrows-of-the-forest eastern-mirkwood dale erebor
    iron-hills vale-of-the-carnen north-rhun east-rhun south-rhun
    northern-rovanion southern-rovanion vale-of-the-celduin
    northern-dorwinion southern-dorwinion ash-mountains noman-lands
    dagorlad dol-guldur southern-mirkwood umbar near-harad far-harad
    khand nurn gorgoroth minas-morgul barad-dur morannon
""".split()

CARD_TITLES = {
    'fp-char-01': 'Elven Cloaks', 'fp-char-02': 'Elven Rope',
    'fp-char-03': 'Phial of Galadriel',
    'fp-char-04': 'Smeagol Helps Nice Master',
    'fp-char-05': 'Mithril Coat and Sting', 'fp-char-06': 'Axe and Bow',
    'fp-char-07': 'Horn of Gondor', 'fp-char-08': "Wizard's Staff",
    'fp-char-09': 'Athelas', 'fp-char-10': 'There is Another Way',
    'fp-char-11': 'I Will Go Alone', 'fp-char-12': "Bilbo's Song",
    'fp-char-13': 'Mirror of Galadriel',
    'fp-char-14': 'Challenge of the King',
    'fp-char-15': 'Gwaihir the Windlord',
    'fp-char-16': 'We Prove the Swifter',
    'fp-char-17': 'There and Back Again',
    'fp-char-18': 'The Eagles are Coming!',
    'fp-char-19': 'The Ents Awake: Treebeard',
    'fp-char-20': 'The Ents Awake: Huorns',
    'fp-char-21': 'The Ents Awake: Entmoot',
    'fp-char-22': 'Dead Men of Dunharrow',
    'fp-char-23': 'House of the Stewards', 'fp-char-24': 'The Grey Company',
    'fp-strat-01': 'The Last Battle', 'fp-strat-02': 'A Power too Great',
    'fp-strat-03': 'The Power of Tom Bombadil',
    'fp-strat-04': 'Book of Mazarbul', 'fp-strat-05': 'The Spirit of Mordor',
    'fp-strat-06': "Faramir's Rangers", 'fp-strat-07': 'Fear! Fire! Foes!',
    'fp-strat-08': 'Wisdom of Elrond', 'fp-strat-09': 'The Red Arrow',
    'fp-strat-10': 'Help Unlooked For', 'fp-strat-11': 'Paths of the Woses',
    'fp-strat-12': 'Through a Day and a Night',
    'fp-strat-13': "Cirdan's Ships", 'fp-strat-14': 'Guards of the Citadel',
    'fp-strat-15': "Celeborn's Galadhrim",
    'fp-strat-16': 'Riders of Theoden',
    'fp-strat-17': 'Grimbeorn the Old, Son of Beorn',
    'fp-strat-18': 'Imrahil of Dol Amroth',
    'fp-strat-19': "King Brand's Men", 'fp-strat-20': 'Swords of Eriador',
    'fp-strat-21': 'Kindred of Glorfindel',
    'fp-strat-22': "Dain Ironfoot's Guard",
    'fp-strat-23': 'Eomer, Son of Eomund',
    'fp-strat-24': "Thranduil's Archers",
    'sh-char-01': "Shelob's Lair", 'sh-char-02': 'The Ring is Mine!',
    'sh-char-03': 'On, On They Went', 'sh-char-04': 'Give it to Uss!',
    'sh-char-05': 'Orc Patrol', 'sh-char-06': "Isildur's Bane",
    'sh-char-07': 'Foul Thing from the Deep',
    'sh-char-08': 'Candles of Corpses', 'sh-char-09': 'Nazgul Search',
    'sh-char-10': 'Cruel Weather', 'sh-char-11': 'The Nazgul Strike!',
    'sh-char-12': 'Morgul Wound', 'sh-char-13': 'Lure of the Ring',
    'sh-char-14': 'The Breaking of the Fellowship',
    'sh-char-15': 'Worn with Sorrow and Toil',
    'sh-char-16': 'Flocks of Crebain', 'sh-char-17': 'Balrog of Moria',
    'sh-char-18': 'The Lidless Eye', 'sh-char-19': 'Dreadful Spells',
    'sh-char-20': 'Grond, Hammer of the Underworld',
    'sh-char-21': 'The Palantir of Orthanc', 'sh-char-22': 'Wormtongue',
    'sh-char-23': 'The Ringwraiths Are Abroad',
    'sh-char-24': 'The Black Captain Commands',
    'sh-strat-01': 'Return to Valinor', 'sh-strat-02': 'The Fighting Uruk-hai',
    'sh-strat-03': "Denethor's Folly", 'sh-strat-04': 'The Day Without Dawn',
    'sh-strat-05': 'Threats and Promises', 'sh-strat-06': 'Stormcrow',
    'sh-strat-07': 'Shadows Gather', 'sh-strat-08': 'The Shadow Lengthens',
    'sh-strat-09': 'The Shadow is Moving', 'sh-strat-10': 'Corsairs of Umbar',
    'sh-strat-11': 'Rage of the Dunlendings',
    'sh-strat-12': 'Return of the Witch-king',
    'sh-strat-13': 'Half-orcs and Goblin-men', 'sh-strat-14': 'Olog-hai',
    'sh-strat-15': 'Hill-trolls', 'sh-strat-16': 'A New Power is Rising',
    'sh-strat-17': 'Many Kings to the Service of Mordor',
    'sh-strat-18': 'The King is Revealed',
    'sh-strat-19': 'Shadows on the Misty Mountains',
    'sh-strat-20': 'Orcs Multiplying Again',
    'sh-strat-21': 'Horde From the East', 'sh-strat-22': 'Monsters Roused',
    'sh-strat-23': 'Musterings of Long-planned War',
    'sh-strat-24': 'Pits of Mordor',
}

# Titles the engine's card list spells differently.
CARD_TITLE_ALIASES = {
    'We Prove the Swifter': 'We Prove Swifter',
    'Swords of Eriador': 'Swords in Eriador',
    'The Nazgul Strike!': 'Nazgul Strike!',
}
# Region keys whose engine name is spelt differently.
REGION_ALIASES = {
    'northern-rovanion': 'northern rhovanion',
    'southern-rovanion': 'southern rhovanion',
}

CHAR_IN = {
    'gandalf-the-grey': 'Gandalf the Grey', 'strider': 'Strider',
    'boromir': 'Boromir', 'legolas': 'Legolas', 'gimli': 'Gimli',
    'meriadoc': 'Meriadoc', 'peregrin': 'Peregrin', 'gollum': 'Gollum',
    'gandalf-the-white': 'Gandalf the White', 'aragorn': 'Aragorn',
    'witch-king': 'Witch King', 'saruman': 'Saruman',
    'mouth-of-sauron': 'Mouth of Sauron',
}
CHAR_OUT = dict((v, k) for k, v in CHAR_IN.items())
MINION_KEYS = ('witch-king', 'saruman', 'mouth-of-sauron')

SPECIAL_TILES = {
    'elven-cloaks': 'Elven Cloaks', 'elven-rope': 'Elven Rope',
    'phial-of-galadriel': 'Phial of Galadriel',
    'smeagol-helps-nice-master': 'Smeagol Helps Nice Master',
    'shelobs-lair': "Shelob's Lair", 'the-ring-is-mine': 'The Ring is Mine!',
    'on-on-they-went': 'On On They Went', 'give-it-to-uss': 'Give it to Uss!',
}

DECKS = ('character', 'strategy')
GATEWAYS = ('morannon', 'minas morgul')


def plain(s):
    """Lower-case letters and digits only, accents folded: the key for
    matching a name across the two vocabularies."""
    s = unicodedata.normalize('NFD', str(s)).lower()
    return ''.join(ch for ch in s if ch.isalnum() and ord(ch) < 128)


def side_in(name):
    return FP if name == 'FREE_PEOPLES' else SH


def side_out(side):
    return 'FREE_PEOPLES' if side is FP else 'SHADOW'


def tag(side):
    return 'fp' if side is FP else 'sh'


def other(side):
    return SH if side is FP else FP


class Names(object):
    """The protocol's names against the engine's."""

    def __init__(self):
        N = gs.Nation
        self.nation_in = {
            'DWARVES': N.DWARVES, 'ELVES': N.ELVES, 'GONDOR': N.GONDOR,
            'NORTH': N.THE_NORTH, 'ROHAN': N.ROHAN, 'ISENGARD': N.ISENGARD,
            'SAURON': N.SAURON, 'SOUTHRONS_EASTERLINGS': N.SOUTHRON_EASTERLINGS}
        self.nation_out = dict((v, k) for k, v in self.nation_in.items())
        F = gs.ActionDieFace
        self.face_in = {
            'CHARACTER': F.CHARACTER, 'ARMY': F.ARMY, 'MUSTER': F.MUSTER,
            'ARMY_MUSTER': F.MUSTER_ARMY, 'EVENT': F.EVENT,
            'WILL_OF_THE_WEST': F.WILL_OF_WEST, 'EYE': F.EYE}
        self.face_out = dict((v, k) for k, v in self.face_in.items())

        by_plain = dict((plain(k), k) for k in board.regions)
        self.region_in = {}
        for slug in REGION_KEYS:
            want = REGION_ALIASES.get(slug, slug)
            key = by_plain.get(plain(want))
            if key is None:
                raise SystemExit('no engine region for %s' % slug)
            self.region_in[slug] = key
        self.region_out = dict((v, k) for k, v in self.region_in.items())
        if len(self.region_out) != len(board.regions):
            raise SystemExit('engine regions without a protocol key: %s'
                             % sorted(set(board.regions) - set(self.region_out)))

        engine_cards = {}
        for cid, c in cards_mod.get_cards().items():
            engine_cards[(c.side, str(c.deck).lower(), plain(c.card_name))] = cid
        self.card_in = {}
        for key, title in CARD_TITLES.items():
            side = FP if key.startswith('fp-') else SH
            deck = 'character' if '-char-' in key else 'strategy'
            title = CARD_TITLE_ALIASES.get(title, title)
            cid = engine_cards.get((side, deck, plain(title)))
            if cid is None:
                raise SystemExit('no engine card for %s (%s)' % (key, title))
            self.card_in[key] = cid
        self.card_out = dict((v, k) for k, v in self.card_in.items())
        self.cards_of = {}
        for key, cid in self.card_in.items():
            side = FP if key.startswith('fp-') else SH
            deck = 'character' if '-char-' in key else 'strategy'
            self.cards_of.setdefault((side, deck), []).append(cid)

        self.special_tile = {}
        for key, name in SPECIAL_TILES.items():
            for en, t in gs.SPECIAL_HUNT_TILES.items():
                if plain(en) == plain(name):
                    self.special_tile[key] = t
            if key not in self.special_tile:
                raise SystemExit('no engine special tile for %s' % key)
        NAMES[0] = self

    def region(self, slug):
        return self.region_in.get(slug)

    def slug(self, key):
        if key is None:
            return None
        return self.region_out.get(board.sanitize(str(key)))

    def card(self, key):
        return self.card_in.get(key)

    def card_key(self, cid):
        return self.card_out.get(cid)

    def deck_of(self, key):
        return 'character' if '-char-' in str(key) else 'strategy'

    def tile(self, key):
        """A Hunt tile of the engine for a protocol tile key."""
        if key in self.special_tile:
            return copy.copy(self.special_tile[key])
        if key.startswith('eye'):
            return gs.HuntTile(is_eye=True)
        head = key.split('-')[0]
        reveal = head.endswith('r')
        return gs.HuntTile(int(head.rstrip('r')), reveal=reveal)


# ---------------------------------------------------------------------
# The engine's state from the referee's view
# ---------------------------------------------------------------------

class GameMemo(object):
    """What the bot remembers of one game beyond the current view: the
    per-turn flags the view does not carry (from the event stream)."""

    def __init__(self, side, seed):
        self.side = side
        self.rng = random.Random(seed)
        self.turn = 0
        self.messenger_used = False      # Mouth of Sauron: Muster as Army
        self.fellowship_acted = False    # moved or hid on the Mordor Track
        self.in_mordor = False

    def event(self, ev):
        kind = ev.get('kind')
        if kind == 'turn_started':
            self.messenger_used = False
            self.fellowship_acted = False
        elif kind == 'fellowship_entered_mordor':
            self.in_mordor = True
        elif kind in ('fellowship_moved', 'fellowship_hidden'):
            if self.in_mordor:
                self.fellowship_acted = True
        elif kind == 'die_used':
            if (ev.get('side') == 'SHADOW' and ev.get('face') == 'MUSTER'
                    and ev.get('action') in ('move_armies', 'attack')):
                self.messenger_used = True


def character_figure(key, by_name):
    """The engine's figure of a Character on the map."""
    name = CHAR_IN.get(key)
    if name is None:
        return None
    if name in by_name:
        return by_name[name]
    if name == 'Gandalf the White':
        # as game_logic.transform_gandalf_white makes him
        return gs.Companion(name='Gandalf the White', level=3, leadership=1,
                            captain_of_the_west=False, bonus_movement=1,
                            solo_hobbit_movement=True)
    if name == 'Aragorn':
        # as game_logic.crown_aragorn makes him
        return gs.Companion(name='Aragorn', level=3, leadership=2,
                            captain_of_the_west=True, activates_nations=[])
    if key in MINION_KEYS:
        return gl.Minion(nation=gs.Nation.SAURON, name=name)
    return None


def build_state(view, memo, names):
    """A GameState of the engine for the referee's view.  The opponent's
    hidden cards (hand, face-down discards, deck order) are any unseen
    cards of the right decks: the engine's search samples them afresh in
    every simulation (its determinization), so it reads the counts, not
    the stand-ins."""
    me = memo.side
    st = gs.setup_initial_game(BOARD)
    by_name = dict((c.name, c) for c in st.fellowship.companions)
    by_name['Gollum'] = st.gollum
    U = gs.UnitKind

    st.armies = {}
    st.besieged_strongholds = set()
    st.enemy_controlled = {FP: set(), SH: set()}
    for slug, r in (view.get('regions') or {}).items():
        key = names.region(slug)
        if key is None:
            continue
        units = r.get('units') or []
        chars = r.get('characters') or []
        if units or chars:
            army = gl.get_or_create_army(st, key)
            for u in units:
                nat = names.nation_in.get(u.get('nation'))
                n = int(u.get('n', 0))
                kind = u.get('unit')
                for _ in range(n):
                    if kind == 'regular':
                        army.units.append(gs.Unit(nat, U.REGULAR))
                    elif kind == 'elite':
                        army.units.append(gs.Unit(nat, U.ELITE))
                    elif kind == 'leader':
                        army.leaders.append(gs.Leader(nat))
                    elif kind == 'nazgul':
                        army.leaders.append(gs.Nazgul(gs.Nation.SAURON))
            for ck in chars:
                fig = character_figure(ck, by_name)
                if fig is not None:
                    army.leaders.append(fig)
        siege = r.get('siege')
        if siege not in (None, False, '', 'false'):
            st.besieged_strongholds.add(key)
        control = r.get('control')
        nat = ai._region_nation(key)
        if control and nat is not None:
            ctl = side_in(control)
            if ctl is not ai._side_of_nation(nat):
                st.enemy_controlled[ctl].add(key)
    st.armies = dict((k, a) for k, a in st.armies.items()
                     if a.units or a.leaders)

    chars = view.get('characters') or {}

    def where(key):
        return (chars.get(key) or {}).get('where', 'not_in_play')

    f = view.get('fellowship') or {}
    fs = st.fellowship
    fkey = names.region(f.get('region')) or 'rivendell'
    fs.region = BOARD.region(fkey)
    fs.progress = int(f.get('progress', 0))
    fs.corruption = int(f.get('corruption', 0))
    fs.hidden = bool(f.get('hidden', True))
    fs.mordor_entered = f.get('mordor') is not None
    fs.mordor_step = int(f.get('mordor') or 0)
    fs.companions = [by_name[CHAR_IN[k]] for k in (f.get('companions') or [])
                     if k != 'gollum' and CHAR_IN.get(k) in by_name]
    guide = f.get('guide')
    if guide == 'gollum':
        fs.guide = st.gollum
    elif CHAR_IN.get(guide) in by_name:
        fs.guide = by_name[CHAR_IN[guide]]
    elif fs.companions:
        fs.guide = fs.companions[0]
    else:
        fs.guide = st.gollum
    memo.in_mordor = fs.mordor_entered

    for key, minion in (('witch-king', st.witch_king), ('saruman', st.saruman),
                        ('mouth-of-sauron', st.mouth_of_sauron)):
        w = where(key)
        minion.in_play = w == 'region'
        minion.was_ever_in_play = w in ('region', 'eliminated')
    st.gandalf_the_white.in_play = where('gandalf-the-white') != 'not_in_play'
    st.aragorn_crowned.crowned = where('aragorn') != 'not_in_play'

    for n, p in (view.get('politics') or {}).items():
        nat = names.nation_in.get(n)
        if nat is None:
            continue
        st.political.boxes[nat] = 3 - int(p.get('step', 3))
        st.political.active[nat] = bool(p.get('active'))
    st.fp_casualties = {}
    for n, c in (view.get('casualties') or {}).items():
        nat = names.nation_in.get(n)
        if nat is not None and ai._side_of_nation(nat) is FP:
            st.fp_casualties[nat] = {'regular': int(c.get('regular', 0)),
                                     'elite': int(c.get('elite', 0)),
                                     'leader': int(c.get('leader', 0))}

    st.dice = {FP: [], SH: []}
    st.hunt_box = []
    eyes_allocated = 0
    rolled = False
    for sname, side in (('FREE_PEOPLES', FP), ('SHADOW', SH)):
        faces = gs.FP_FACES if side is FP else gs.SH_FACES
        for d in (view.get('dice') or {}).get(sname) or []:
            die = gs.ActionDie(faces, side)
            face = d.get('face')
            die.current = names.face_in.get(face) if face else None
            status = d.get('status')
            if face:
                rolled = True
            if status == 'used':
                die.used = True
            elif status == 'hunt_box':
                die.used = True
                if side is SH:
                    if die.current is None:
                        eyes_allocated += 1
                    die.current = gs.ActionDieFace.EYE
                st.hunt_box.append(die)
            st.dice[side].append(die)
    st.eyes_allocated = eyes_allocated
    st.dice_rolled = rolled
    st.fp_bonus_dice = max(0, len(st.dice[FP]) - 4)
    st.sh_bonus_dice = max(0, len(st.dice[SH]) - 7)

    hunt = view.get('hunt') or {}
    st.hunt_pool = [names.tile(t) for t in hunt.get('pool') or []]
    # drawn tiles and specials played before Mordor: the engine keeps
    # both in its used pile (event_effects.stage_special_hunt_tile)
    st.hunt_discard = [names.tile(t) for t in (hunt.get('drawn') or [])
                       + (hunt.get('ready') or [])]
    st.hunt_removed = [names.tile(t) for t in hunt.get('removed') or []]
    st._fp_used_hunt_box_last_turn = bool(hunt.get('shadow_must_hunt'))

    _build_cards(st, view, memo, names)

    rings = view.get('elven_rings') or {}
    used = view.get('elven_ring_used') or {}
    st.fp_elven_rings = int(rings.get('FREE_PEOPLES', 0))
    st.sh_elven_rings = int(rings.get('SHADOW', 0))
    st.fp_ring_used_this_turn = bool(used.get('FREE_PEOPLES'))
    st.sh_ring_used_this_turn = bool(used.get('SHADOW'))
    st.turn = int(view.get('turn', 1))
    st.sh_mouth_muster_used_this_turn = memo.messenger_used
    st.fellowship_acted_in_mordor = memo.fellowship_acted
    vp = gl.calculate_victory_points(st)
    st.vp.free_peoples = int(vp.get('free_peoples', 0))
    st.vp.shadow = int(vp.get('shadow', 0))
    st.pending_discard = None
    return st


def _build_cards(st, view, memo, names):
    me = memo.side
    opp = other(me)
    c = view.get('cards') or {}
    discards = c.get('discards') or {}
    table = c.get('on_table') or {}
    battle = view.get('battle') or {}
    in_battle = set(k for k in (battle.get('attacker_card'),
                                battle.get('defender_card'),
                                battle.get('event_card')) if k)

    for side in (FP, SH):
        for deck in DECKS:
            st.discard_face_up['%s_%s' % (tag(side), deck)] = [
                names.card(k) for k in discards.get(side_out(side)) or []
                if names.deck_of(k) == deck and names.card(k)]

    # Own cards: all of them is known.
    hand = [names.card(k) for k in c.get('hand') or [] if names.card(k)]
    fdown = [names.card(k) for k in c.get('discarded_face_down') or []
             if names.card(k)]
    seen = set(hand) | set(fdown) | set(names.card(k) for k in in_battle)
    seen |= set(names.card(k) for k in discards.get(side_out(me)) or [])
    seen |= set(names.card(k) for k in table.get(side_out(me)) or [])
    decks = (c.get('decks') or {}).get(side_out(me)) or {}
    for deck in DECKS:
        rest = [cid for cid in names.cards_of[(me, deck)] if cid not in seen]
        memo.rng.shuffle(rest)
        want = int(decks.get(deck, len(rest)))
        setattr(st, '%s_%s_deck' % (tag(me), 'char' if deck == 'character'
                                    else 'strat'), rest[:want])
    st.hands[me].hand = hand
    st.hands[me].discarded_face_down = fdown
    st.hands[me].needs_discard = len(hand) > 6

    # The opponent's: stand-ins from the unseen cards of each deck.
    seen = set(names.card(k) for k in in_battle)
    seen |= set(names.card(k) for k in discards.get(side_out(opp)) or [])
    seen |= set(names.card(k) for k in table.get(side_out(opp)) or [])
    by_deck = c.get('opponent_hand_by_deck') or {}
    odecks = (c.get('decks') or {}).get(side_out(opp)) or {}
    total_fd = int(c.get('opponent_discarded_face_down', 0))
    ohand, ofd = [], []
    for deck in DECKS:
        unseen = [cid for cid in names.cards_of[(opp, deck)]
                  if cid not in seen]
        memo.rng.shuffle(unseen)
        nh = min(int(by_deck.get(deck, 0)), len(unseen))
        nd = min(int(odecks.get(deck, 0)), len(unseen) - nh)
        rest = unseen[nh + nd:]
        take_fd = min(len(rest), max(0, total_fd - len(ofd)))
        ohand += unseen[:nh]
        ofd += rest[:take_fd]
        setattr(st, '%s_%s_deck' % (tag(opp), 'char' if deck == 'character'
                                    else 'strat'), unseen[nh:nh + nd])
    st.hands[opp].hand = ohand
    st.hands[opp].discarded_face_down = ofd
    st.hands[opp].needs_discard = len(ohand) > 6
    st.decks = gs.DeckStatus(len(st.fp_char_deck), len(st.fp_strat_deck),
                             len(st.sh_char_deck), len(st.sh_strat_deck))

    from persistent_effects import create_persistent_card
    st.persistent_cards = gs.PersistentCardState()
    for side in (FP, SH):
        for k in table.get(side_out(side)) or []:
            cid = names.card(k)
            if cid is None:
                continue
            card = cards_mod.get_card(cid)
            try:
                pc = create_persistent_card(cid, card.card_name, side)
            except Exception:
                pc = gs.PersistentCard(card_id=cid, card_name=card.card_name,
                                       side=side, card_type=None)
            st.persistent_cards.add_card(pc)


def state_notes(st, view, names):
    """Differences between the rebuilt state and the view that the
    engine computes itself (reinforcements); for the log only."""
    notes = []
    deployed = ai._count_deployed_all(st)
    for n, r in (view.get('reinforcements') or {}).items():
        nat = names.nation_in.get(n)
        if nat is None:
            continue
        for typ in ('regular', 'elite', 'leader'):
            if typ == 'leader' and n == 'SAURON':
                want = int(r.get('nazgul', 0))
            else:
                want = int(r.get(typ, 0))
            have = ai.figures_available(st, nat, typ, deployed)
            if have != want:
                notes.append('%s %s: engine %d, view %d' % (n, typ, have, want))
    return notes


# ---------------------------------------------------------------------
# Helpers on the engine's state
# ---------------------------------------------------------------------

def freeze(v):
    """A hashable, order-free form of an action's parameters."""
    if isinstance(v, dict):
        return tuple(sorted((str(k), freeze(x)) for k, x in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(freeze(x) for x in v)
    if isinstance(v, (set, frozenset)):
        return tuple(sorted(str(freeze(x)) for x in v))
    if hasattr(v, 'name') and hasattr(v, 'value'):
        return str(v.name)
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(getattr(v, 'name', v))


def act_key(a):
    return (a.action_type, a.die_idx, freeze(a.params or {}))


def act_json(a):
    p = {}
    for k, v in (a.params or {}).items():
        if str(k).startswith('_'):
            continue
        p[str(k)] = freeze(v)
    return {'type': a.action_type, 'die': a.die_idx, 'params': p}


def deep_copy_state(st):
    return copy.deepcopy(st, {id(st.board): st.board})


def army_at(st, key):
    san = board.sanitize(str(key))
    for k, a in st.armies.items():
        if board.sanitize(k) == san:
            return a
    return None


def figures(st, key, side, names):
    """{(nation, unit): n} and the Character keys of [side] in a region."""
    out = {}
    chars = set()
    a = army_at(st, key)
    if a is None:
        return out, chars
    for u in a.units:
        if ai._side_of_nation(u.nation) is not side:
            continue
        unit = 'regular' if u.kind is gs.UnitKind.REGULAR else 'elite'
        k = (names.nation_out[u.nation], unit)
        out[k] = out.get(k, 0) + 1
    for ld in a.leaders:
        tn = type(ld).__name__
        if tn == 'Nazgul':
            if side is SH:
                k = ('SAURON', 'nazgul')
                out[k] = out.get(k, 0) + 1
        elif tn == 'Leader':
            if ai._side_of_nation(ld.nation) is side:
                k = (names.nation_out[ld.nation], 'leader')
                out[k] = out.get(k, 0) + 1
        else:
            ck = CHAR_OUT.get(getattr(ld, 'name', None))
            if ck is None:
                continue
            if (ck in MINION_KEYS) == (side is SH):
                chars.add(ck)
    return out, chars


def moved_figures(before, after, frm, side, names):
    """What left region [frm] between two states of the engine."""
    b, bc = figures(before, frm, side, names)
    a, ac = figures(after, frm, side, names)
    moved = {}
    for k, n in b.items():
        d = n - a.get(k, 0)
        if d > 0:
            moved[k] = d
    return moved, bc - ac, b, bc


class _Phase(object):
    """Sets one of the engine's phase flags for the menu and the search."""

    def __init__(self, st, flag):
        self.st, self.flag = st, flag

    def __enter__(self):
        if self.flag:
            setattr(self.st, self.flag, True)

    def __exit__(self, *exc):
        if self.flag:
            setattr(self.st, self.flag, False)
        return False


# ---------------------------------------------------------------------
# The engine's search
# ---------------------------------------------------------------------

class Searcher(object):
    """ai.mcts_action_with_policy as EngineSession.ai_pick calls it."""

    def __init__(self, sims, max_seconds, threads=1):
        self.sims = int(sims)
        self.max_seconds = max_seconds
        self.rates = {}                 # simulations per second, per kind
        self.recent = {}
        if getattr(ai, 'HAS_TORCH', False):
            try:
                ai.torch.set_num_threads(int(threads))
            except Exception:
                pass
        t0 = time.time()
        self.nets = engine_play.load_nets('corpus_live')
        log('nets loaded in %.1f s' % (time.time() - t0))

    def budget(self, kind, left_ms):
        n = engine_play._sims_for_kind(kind, self.sims)
        limit = self.max_seconds
        if left_ms is not None:
            # the search's time per simulation varies twofold between
            # positions and cannot be stopped midway: aim at 40% of the
            # clock
            room = max(0.5, (left_ms / 1000.0) * 0.4 - 0.2)
            limit = room if limit is None else min(limit, room)
        if limit is not None and self.rates:
            rate = self.rates.get(kind)
            if rate is None:
                # a kind not searched yet: the slowest rate seen, and a
                # third of it for the turn-start decisions (their search
                # runs through the roll)
                rate = min(self.rates.values())
                if kind in ('eye', 'declare'):
                    rate /= 3.0
            n = min(n, max(8, int(rate * limit)))
        return n

    def choose(self, st, side, kind, banned, flag=None, left_ms=None):
        """The search's action for [side] at [st], never one of [banned]
        (keys of actions without a referee counterpart: they are left out
        of this side's menus during the search)."""
        fp_net, sh_net = self.nets
        sims = self.budget(kind, left_ms)
        temp = 0.0 if kind in ('declare', 'eye') else 0.02
        ai._TREE_CACHE['child'] = None
        ai._TREE_CACHE['sig'] = None
        orig = ai.get_legal_actions

        def legal(st9, side9):
            out = orig(st9, side9)
            if side9 is side and banned:
                kept = [a for a in out if act_key(a) not in banned]
                if kept:
                    return kept
            return out

        ai.get_legal_actions = legal
        t0 = time.time()
        try:
            with _Phase(st, flag):
                action, info = ai.mcts_action_with_policy(
                    st, side, fp_net, sh_net, num_simulations=sims,
                    temperature=temp, add_noise=False)
        finally:
            ai.get_legal_actions = orig
        secs = time.time() - t0
        if secs > 0.05 and sims >= 8:
            r = sims / secs
            # the slowest of the last three searches of this kind
            recent = (self.recent.get(kind) or [])[-2:] + [r]
            self.recent[kind] = recent
            self.rates[kind] = min(recent)
        value = None
        try:
            if info is not None:
                value = float(info['q_values'][int(info['played'])])
        except Exception:
            value = None
        return action, {'sims': sims, 'secs': round(secs, 2), 'value': value}


# ---------------------------------------------------------------------
# Options of a decision
# ---------------------------------------------------------------------

def opts(dec, kind=None, **fields):
    """The options of [dec] of option kind [kind] whose fields equal
    [fields] (a field given as None must be absent or null)."""
    out = []
    for o in dec.get('options') or []:
        if kind is not None and o.get('kind') != kind:
            continue
        ok = True
        for k, v in fields.items():
            if o.get(k) != v:
                ok = False
                break
        if ok:
            out.append(o)
    return out


def first(lst):
    return lst[0]['o'] if lst else None


# ---------------------------------------------------------------------
# Plans: the referee's follow-up questions of one engine action
# ---------------------------------------------------------------------

class Plan(object):
    kinds = ()
    name = 'plan'
    near = False       # an answer was the nearest, not the engine's own

    def answer(self, dec):
        """An option for [dec], or None: the plan has nothing to say."""
        return None


class ArmyMovePlan(Plan):
    """army_move / victor_advance: the Army, its destination, the units."""
    name = 'army-move'

    def __init__(self, frm, to, moved, chars, everything, advance=False):
        self.frm, self.to = frm, to
        self.moved, self.chars = moved, chars
        self.everything = everything
        self.advance = advance
        self.started = False
        self.kinds = ('ADVANCE', 'ARMY_SELECTION') if advance else (
            'ARMY_MOVE', 'ARMY_SELECTION')

    def answer(self, dec):
        k = dec['kind']
        if k == 'ARMY_MOVE':
            if self.started:
                return None          # the next move of the chain
            cand = opts(dec, 'army_move', **{'from': self.frm, 'to': self.to})
            if not cand:
                # the engine's destination is not open to the referee's
                # rules (a border only one map has): the nearest one
                near = opts(dec, 'army_move', **{'from': self.frm})
                if not near:
                    return None
                self.started = True
                self.near = True
                return nearest(near, self.to)
            pick = [o for o in cand if bool(o.get('all')) == self.everything]
            self.started = True
            return (pick or cand)[0]['o']
        if k == 'ADVANCE':
            cand = opts(dec, 'advance')
            if not cand:
                return None
            pick = [o for o in cand if bool(o.get('all')) == self.everything]
            self.started = True
            return (pick or cand)[0]['o']
        if k == 'ARMY_SELECTION' and self.started:
            ctx = dec.get('context') or {}
            chosen = None
            for o in opts(dec, 'count'):
                if o.get('character'):
                    want = 1 if o['character'] in self.chars else 0
                else:
                    want = self.moved.get((o.get('nation') or ctx.get('nation'),
                                           o.get('unit') or ctx.get('unit')), 0)
                if int(o.get('count', -1)) == want:
                    chosen = o['o']
            if chosen is None:
                # the nearest count offered
                best = None
                for o in opts(dec, 'count'):
                    if o.get('character'):
                        want = 1 if o['character'] in self.chars else 0
                    else:
                        want = self.moved.get((o.get('nation') or ctx.get('nation'),
                                               o.get('unit') or ctx.get('unit')), 0)
                    d = abs(int(o.get('count', 0)) - want)
                    if best is None or d < best[0]:
                        best = (d, o['o'])
                chosen = best[1] if best else None
            return chosen
        return None


class AttackPlan(Plan):
    kinds = ('CHOOSE_ATTACK', 'ARMY_SELECTION')
    name = 'attack'

    def __init__(self, frm, to):
        self.frm, self.to = frm, to

    def answer(self, dec):
        if dec['kind'] == 'CHOOSE_ATTACK':
            cand = opts(dec, 'attack', **{'from': self.frm, 'to': self.to})
            pick = [o for o in cand if o.get('all', True)]
            return first(pick or cand)
        # a rearguard question: every unit attacks
        ctx = dec.get('context') or {}
        cand = opts(dec, 'count')
        if not cand:
            return None
        avail = ctx.get('available')
        best = max(cand, key=lambda o: int(o.get('count', 0)))
        if avail is not None:
            exact = [o for o in cand if int(o.get('count', 0)) == int(avail)]
            if exact:
                return exact[0]['o']
        return best['o']


class RecruitPlan(Plan):
    kinds = ('RECRUIT',)
    name = 'recruit'

    def __init__(self, region, nation, unit):
        self.region, self.nation, self.unit = region, nation, unit
        self.done = False

    def answer(self, dec):
        if self.done:
            return None
        cand = opts(dec, 'recruit', region=self.region, nation=self.nation,
                    unit=self.unit)
        if cand:
            self.done = True
            return cand[0]['o']
        return None


class CompanionMovePlan(Plan):
    kinds = ('MOVE_COMPANIONS', 'JOIN_GROUP')
    name = 'companions'

    def __init__(self, names_, frm, to):
        self.group = set(names_)
        self.frm, self.to = frm, to
        self.started = False

    def answer(self, dec):
        if dec['kind'] == 'MOVE_COMPANIONS':
            if self.started:
                return None
            for o in opts(dec, 'move_companion'):
                if (o.get('character') in self.group and o.get('to') == self.to
                        and o.get('from', self.frm) == self.frm):
                    self.started = True
                    return o['o']
            return None
        ctx = dec.get('context') or {}
        who = ctx.get('character')
        cand_join = [o for o in opts(dec, 'join')
                     if o.get('character', who) == who]
        cand_stay = [o for o in opts(dec, 'stay')
                     if o.get('character', who) == who]
        if who in self.group:
            return first(cand_join) if cand_join else first(cand_stay)
        return first(cand_stay) if cand_stay else first(cand_join)


class SeparatePlan(Plan):
    kinds = ('SEPARATE_COMPANIONS',)
    name = 'separate'

    def __init__(self, names_, to):
        self.left = list(names_)
        self.to = to

    def answer(self, dec):
        for o in opts(dec, 'separate'):
            if o.get('character') in self.left:
                self.left.remove(o['character'])
                return o['o']
        cand = opts(dec, 'separate_to', to=self.to)
        if cand:
            return cand[0]['o']
        return None


class MinionPlan(Plan):
    """Nazgul and Minion flights: [(figure, from, to)], figure 'nazgul'
    or a Minion's key."""
    kinds = ('MOVE_MINIONS',)
    name = 'minions'

    def __init__(self, flights):
        self.flights = list(flights)

    def answer(self, dec):
        if not self.flights:
            return None
        fig, frm, to = self.flights[0]
        if fig == 'nazgul':
            moves = opts(dec, 'move_nazgul', **{'from': frm})
            select = opts(dec, 'select_nazgul', **{'from': frm})
        else:
            moves = opts(dec, 'move_minion', character=fig)
            select = opts(dec, 'select_minion', character=fig)
        if moves:
            self.flights.pop(0)
            exact = [o for o in moves if o.get('to') == to]
            if exact:
                return exact[0]['o']
            # the engine's destination is not open to the referee's
            # rules: the nearest one that is
            self.near = True
            return nearest(moves, to)
        return first(select)


def region_distance(a, b, limit=12):
    """Steps between two protocol regions on the engine's map."""
    if a == b:
        return 0
    ka, kb = NAMES[0].region(a), NAMES[0].region(b)
    if ka is None or kb is None:
        return limit + 1
    seen, frontier = {ka}, [ka]
    for d in range(1, limit + 1):
        nxt = []
        for r in frontier:
            for n in BOARD.adjacent(r):
                n = board.sanitize(n)
                if n == kb:
                    return d
                if n not in seen:
                    seen.add(n)
                    nxt.append(n)
        frontier = nxt
    return limit + 1


NAMES = [None]


def nearest(options, to, field='to'):
    """The option whose [field] region is nearest to [to]."""
    if not options:
        return None
    return min(options, key=lambda o: region_distance(o.get(field), to))['o']


class CasualtyPlan(Plan):
    """The engine allocates a round's hits at once (k Elites reduced, the
    rest Regulars removed, then Elites removed by two hits); the referee
    asks one hit at a time."""
    kinds = ('CASUALTY',)
    name = 'casualties'

    def __init__(self, key, k):
        self.key, self.k = key, int(k)

    def answer(self, dec):
        if casualty_key(dec) != self.key:
            return None
        return casualty_pick(dec, self.k > 0, self)


def casualty_key(dec):
    c = dec.get('context') or {}
    return (c.get('region'), c.get('round'), c.get('attacker'), dec.get('side'))


def casualty_pick(dec, reduce_first, plan=None):
    def most(cand, unit):
        # the Nation with the most of that unit keeps its other kinds
        return cand[0]['o'] if cand else None
    red = opts(dec, 'casualty', unit='elite', effect='reduce')
    rem_r = opts(dec, 'casualty', unit='regular')
    rem_e = opts(dec, 'casualty', unit='elite', effect='remove')
    if reduce_first and red:
        if plan is not None:
            plan.k -= 1
        return red[0]['o']
    if rem_r:
        return rem_r[0]['o']
    hits = int((dec.get('context') or {}).get('hits', 1))
    if rem_e and hits >= 2:
        return rem_e[0]['o']
    if red:
        if plan is not None:
            plan.k -= 1
        return red[0]['o']
    if rem_e:
        return rem_e[0]['o']
    return None


class RetreatPlan(Plan):
    kinds = ('RETREAT',)
    name = 'retreat'

    def __init__(self, to):
        self.to = to

    def answer(self, dec):
        if self.to is None:
            return first(opts(dec, 'stand'))
        return first(opts(dec, 'retreat', region=self.to))


class AnswerPlan(Plan):
    """A fixed answer for one later question (by option kind)."""
    name = 'answer'

    def __init__(self, kind, okind, fields=None):
        self.kinds = (kind,)
        self.okind = okind
        self.fields = fields or {}

    def answer(self, dec):
        return first(opts(dec, self.okind, **self.fields))


# ---------------------------------------------------------------------
# The bot
# ---------------------------------------------------------------------

DIE_ACTIONS = {
    # engine action -> the referee's 'action' of use_die (first match)
    'draw_card': ('draw_card',),
    'play_event': ('play_card',),
    'army_attack': ('attack', 'attack_with_leader'),
    'army_move': ('move_armies', 'move_army_with_leader'),
    'advance_political': ('diplomacy',),
    'muster_recruit': ('recruit',),
    'muster_elite': ('recruit',),
    'muster_minion': ('play_minion',),
    'transform_gandalf': ('play_companion',),
    'crown_aragorn': ('play_companion',),
    'fellowship_move': ('move_fellowship',),
    'fellowship_hide': ('hide_fellowship', 'hide_fellowship_strider'),
    'companion_move': ('move_companions',),
    'companion_separate': ('separate_companions',),
    'nazgul_move': ('move_minions',),
    'mouth_move': ('move_minions',),
    'discard_table_card': ('force_discard',),
}


class Match(object):
    """The referee's option for an engine action, the plan for the
    questions that follow (a Plan, or a function making it, called only
    for the action the search picks) and whether the engine's chain goes
    on on a copy of its state."""

    def __init__(self, o, plan=None, chain=False, resolve=None):
        self.o = o
        self._plan = plan
        self.chain = chain
        self.resolve = resolve     # the final option, given the plan

    def plan(self):
        p = self._plan
        if p is not None and not isinstance(p, Plan):
            p = p()
            self._plan = p
        return p

    def option(self):
        if self.resolve is not None:
            o = self.resolve(self.plan())
            if o is not None:
                return o
        return self.o

    def has_plan(self):
        return self._plan is not None


class AlphaWotrBot(object):

    def __init__(self, searcher, names, verbose=False, decision_log=None):
        self.searcher = searcher
        self.names = names
        self.verbose = verbose
        self.decision_log = decision_log
        self.memo = None
        self.game = None
        self.plan = None
        self.work = None           # the engine's state of an ongoing chain
        self.work_base = None      # ... and the state it started from
        self.diff = None
        self.match_stats = Stats()
        self.stats = None
        self.options = {}
        self.score = {'games': 0, 'won': 0}

    # -- protocol messages ---------------------------------------------

    def handle(self, m):
        t = m.get('type')
        if t == 'hello':
            return {'type': 'hello', 'id': m.get('id'), 'protocol': PROTOCOL,
                    'name': BOT_NAME, 'version': BOT_VERSION,
                    'author': 'AlphaWOTR (protocol adapter)', 'events': True,
                    'options': [{'name': 'sims', 'type': 'int',
                                 'default': self.searcher.sims, 'min': 1,
                                 'max': 100000}]}
        if t == 'options':
            vals = m.get('values') or {}
            if 'sims' in vals:
                try:
                    self.searcher.sims = max(1, int(vals['sims']))
                except (TypeError, ValueError):
                    pass
            if m.get('match'):
                log('match:', json.dumps(m['match']))
            return None
        if t == 'new_game':
            seed = int(m.get('seed_bot') or 0)
            random.seed(seed)
            try:
                import numpy as np
                np.random.seed(seed % (2 ** 32))
            except Exception:
                pass
            if getattr(ai, 'HAS_TORCH', False):
                ai.torch.manual_seed(seed % (2 ** 63))
            self.memo = GameMemo(side_in(m.get('side')), seed)
            self.game = m.get('game')
            self.plan = None
            self.work = None
            self.stats = Stats()
            log('game %d: %s as %s' % (self.score['games'] + 1, self.game,
                                       m.get('side')))
            return {'type': 'ready', 'id': m.get('id')}
        if t == 'event':
            if self.memo is not None:
                self.memo.event(m.get('event') or {})
            return None
        if t == 'decide':
            return self.decide(m)
        if t == 'get_view':
            return None
        if t == 'timeout':
            if self.stats is not None:
                self.stats.timeouts += 1
            self.match_stats.timeouts += 1
            log('referee: timeout', json.dumps(m))
            return None
        if t == 'error':
            if m.get('code') in ('illegal_option', 'stale_fingerprint',
                                 'not_a_choice'):
                if self.stats is not None:
                    self.stats.illegal += 1
                self.match_stats.illegal += 1
            log('referee: error', json.dumps(m))
            return None
        if t == 'game_over':
            self.score['games'] += 1
            mine = self.memo and side_out(self.memo.side)
            won = m.get('winner') == mine
            self.score['won'] += int(won)
            if self.stats is not None:
                log('game over: %s (%s) turn %s, I played %s -- %s' % (
                    m.get('winner'), m.get('how'), m.get('turn'), mine,
                    self.stats.summary()))
                self.match_stats.add(self.stats)
            self.memo = None
            return None
        if t == 'quit':
            log('match over: won %d of %d -- %s' % (
                self.score['won'], self.score['games'],
                self.match_stats.summary()))
            raise SystemExit(0)
        if 'id' in m:
            return {'type': 'error', 'id': m['id'], 'code': 'unknown_type'}
        return None

    # -- one decision --------------------------------------------------

    def decide(self, m):
        dec = m['decision']
        t0 = time.time()
        self.t0 = t0
        left = (m.get('clock') or {}).get('left_ms')
        how = 'search'
        info = {}
        o = None
        try:
            o, how, info = self._decide(dec, m.get('view') or {}, left)
        except Exception as e:
            how = 'error'
            info = {'error': '%s: %s' % (type(e).__name__, e)}
            log('! %s at %s: %s' % (self.game, dec.get('kind'), info['error']))
            log(''.join(traceback.format_exc().splitlines(True)[-6:]))
        legal = set(x['o'] for x in dec.get('options') or [])
        if o not in legal:
            if how not in ('error',):
                info['bad'] = o
            o, rule = default_option(dec)
            how = 'default' if how != 'error' else 'error-default'
            info['rule'] = rule
            self.plan = None
        secs = time.time() - t0
        if self.stats is not None:
            self.stats.count(dec['kind'], how, secs, info)
        if self.verbose:
            text = next((x.get('text') for x in dec.get('options') or []
                         if x['o'] == o), '')
            log('T%s %s %s -> %s [%s %.2fs] %s' % (
                (m.get('view') or {}).get('turn'), dec.get('side'),
                dec['kind'], text, how, secs,
                json.dumps(info.get('engine'))[:160] if info.get('engine') else
                info.get('note', '')))
        if self.decision_log is not None:
            self.decision_log.write(json.dumps({
                'game': self.game, 'turn': (m.get('view') or {}).get('turn'),
                'kind': dec['kind'], 'o': o, 'how': how, 'secs': round(secs, 3),
                'info': info}, default=str) + '\n')
            self.decision_log.flush()
        out = {'type': 'choice', 'id': m['id'], 'o': o,
               'fingerprint': dec.get('fingerprint'),
               'info': {'how': how}}
        for k in ('sims', 'value'):
            if info.get(k) is not None:
                out['info'][k] = info[k]
        if info.get('engine'):
            # the engine's action, short (the referee logs it)
            out['info']['engine'] = json.dumps(info['engine'],
                                               separators=(',', ':'))[:240]
        return out

    def _decide(self, dec, view, left):
        kind = dec['kind']
        options = dec.get('options') or []
        if len(options) == 1:
            self._continue_plan_quietly(dec)
            return options[0]['o'], 'only', {}
        # 1. the plan of the engine's last action
        if self.plan is not None and kind in self.plan.kinds:
            o = self.plan.answer(dec)
            if o is not None:
                return o, 'plan', {'note': self.plan.name}
        self.plan = None
        # 2. a chain of the engine's last action, on its copy of the state
        if self.work is not None and kind in CHAIN_KINDS:
            r = self._chain(dec, view, left)
            if r is not None:
                return r
            # the engine settled it inside its action: read the answer
            # off its board after the action
            o = diff_answer(self, dec)
            if o is not None:
                return o, 'board', {'note': 'as the engine resolved it'}
            # the engine's chain is over: so is the referee's
            if kind in ('ARMY_MOVE', 'RECRUIT', 'MOVE_COMPANIONS',
                        'MOVE_MINIONS') and opts(dec, 'end') and \
                    chain_over(self.work, kind):
                return first(opts(dec, 'end')), 'chain', {'note': 'chain over'}
        if kind not in CHAIN_KINDS:
            # a question outside the engine's action: its copy is done
            self.work = None
            self.work_base = None
            self.diff = None
        # 3. the state rebuilt from the view, the question set up
        st = build_state(view, self.memo, self.names)
        if self.stats is not None and not self.stats.checked:
            self.stats.checked = True
            notes = state_notes(st, view, self.names)
            if notes:
                log('  state notes (%s): %s' % (self.game, '; '.join(notes[:6])))
        setup = SETUPS.get(kind)
        ready = setup(self, st, dec, view) if setup is not None else None
        if ready is None:
            rule = RULES.get(kind)
            o = rule(self, dec, view) if rule is not None else None
            if o is not None:
                return o, 'rule', {'note': rule.__doc__.split('\n')[0]}
            return None, 'default', {'note': 'no engine decision for %s' % kind}
        if isinstance(ready, Match):
            return ready.o, 'rule', {'note': 'engine rule'}
        label, flag = ready
        return self._search_and_map(st, dec, label, flag, left)

    def _left(self, left):
        """The clock left for this decision once the matching is done."""
        if left is None:
            return None
        return max(0, left - 1000.0 * (time.time() - self.t0))

    def _continue_plan_quietly(self, dec):
        # a forced question still advances the plan of the engine's action
        if self.plan is not None and dec['kind'] in self.plan.kinds:
            try:
                self.plan.answer(dec)
            except Exception:
                self.plan = None

    def _search_and_map(self, st, dec, label, flag, left):
        me = self.memo.side
        with _Phase(st, flag):
            acts = ai.get_legal_actions(st, me)
        if not acts:
            return None, 'default', {'note': 'engine has no action'}
        mapped = []
        banned = set()
        for a in acts:
            try:
                mt = self.map_action(st, dec, a)
            except Exception as e:
                mt = None
                if self.verbose:
                    log('  map error %s: %s' % (act_json(a), e))
            if mt is None:
                banned.add(act_key(a))
            else:
                mapped.append((a, mt))
        info = {'menu': len(acts), 'unmatched': len(acts) - len(mapped)}
        if self.stats is not None:
            self.stats.menu(dec['kind'], len(acts), len(acts) - len(mapped))
        if not mapped:
            info['note'] = 'no engine action matches an option'
            info['engine_menu'] = [act_json(a) for a in acts[:6]]
            return None, 'default', info
        if len(set(mt.o for _, mt in mapped)) == 1 and all(
                not mt.has_plan() and not mt.chain for _, mt in mapped):
            a, mt = mapped[0]
            info['engine'] = act_json(a)
            return mt.o, 'single', info
        if len(mapped) == 1:
            a, mt = mapped[0]
        else:
            action, sinfo = self.searcher.choose(st, me, label, banned, flag,
                                                 self._left(left))
            info.update(sinfo)
            a, mt = None, None
            if action is not None:
                k = act_key(action)
                for a2, mt2 in mapped:
                    if act_key(a2) == k:
                        a, mt = a2, mt2
                        break
            if mt is None:
                info['note'] = 'search returned %s' % (
                    act_json(action) if action is not None else None)
                a, mt = mapped[0]
                info['fallback'] = 'first matched'
        info['engine'] = act_json(a)
        o = mt.option()
        self.plan = mt.plan()
        if mt.chain:
            self.work = self._apply_copy(st, a, flag)
            self.work_base = st
            self.work_action = a.action_type
            self.diff = None
        return o, 'search' if 'sims' in info else 'single', info

    def _apply_copy(self, st, a, flag=None):
        w = deep_copy_state(st)
        try:
            with _Phase(w, flag):
                ai._execute_action(w, a)
        except Exception as e:
            if self.verbose:
                log('  apply %s failed: %s' % (act_json(a), e))
            return None
        return w

    def _chain(self, dec, view, left):
        """The next link of the engine's chain, on its copy of the state."""
        w = self.work
        me = self.memo.side
        try:
            acts = ai.get_legal_actions(w, me)
        except Exception:
            return None
        if not acts or not chain_fits(w, dec, acts):
            return None
        mapped, banned = [], set()
        for a in acts:
            try:
                mt = self.map_action(w, dec, a)
            except Exception:
                mt = None
            if mt is None:
                banned.add(act_key(a))
            else:
                mapped.append((a, mt))
        if self.stats is not None:
            self.stats.menu(dec['kind'], len(acts), len(acts) - len(mapped))
        if not mapped:
            return None
        info = {'menu': len(acts), 'unmatched': len(acts) - len(mapped),
                'note': 'chain'}
        if len(mapped) == 1:
            a, mt = mapped[0]
        else:
            action, sinfo = self.searcher.choose(w, me, 'action', banned, None,
                                                 self._left(left))
            info.update(sinfo)
            a, mt = None, None
            if action is not None:
                for a2, mt2 in mapped:
                    if act_key(a2) == act_key(action):
                        a, mt = a2, mt2
                        break
            if mt is None:
                a, mt = mapped[0]
                info['fallback'] = 'first matched'
        info['engine'] = act_json(a)
        o = mt.option()
        self.plan = mt.plan()
        self.work = self._apply_copy(w, a)
        return o, 'chain', info

    # -- the engine's action -> the referee's option -------------------

    def map_action(self, st, dec, a):
        fn = MAPPERS.get(dec['kind'])
        if fn is None:
            return None
        return fn(self, st, dec, a)

    def slug(self, key):
        return self.names.slug(key)


def chain_over(w, kind):
    """The engine's state holds no further link of that kind of chain."""
    pco = getattr(w, 'pending_card_option', None)
    if kind == 'ARMY_MOVE':
        return w.pending_army_move_die is None and getattr(
            w, 'pending_card_followup', None) is None
    if kind == 'RECRUIT':
        return w.pending_muster_die is None and pco is None
    if kind == 'MOVE_COMPANIONS':
        return pco is None or pco.get('card_name') != '__companion_moves__'
    if kind == 'MOVE_MINIONS':
        return pco is None or pco.get('card_name') != '__nazgul_moves__'
    return False


CHAIN_KINDS = ('ARMY_MOVE', 'RECRUIT', 'MOVE_COMPANIONS', 'MOVE_MINIONS',
               'CARD_CHOICE', 'JOIN_GROUP', 'ARMY_SELECTION', 'CHOOSE_ATTACK',
               'SEPARATE_COMPANIONS')


def chain_fits(w, dec, acts):
    """Is the engine's open chain the question the referee asks?"""
    kind = dec['kind']
    types = set(a.action_type for a in acts)
    pco = getattr(w, 'pending_card_option', None)
    if kind == 'ARMY_MOVE':
        return w.pending_army_move_die is not None or (
            getattr(w, 'pending_card_followup', None) is not None) or (
            pco is not None and pco.get('card_name') not in (
                '__companion_moves__', '__nazgul_moves__'))
    if kind == 'RECRUIT':
        return w.pending_muster_die is not None or pco is not None
    if kind == 'MOVE_COMPANIONS':
        return pco is not None and pco.get('card_name') == '__companion_moves__'
    if kind == 'MOVE_MINIONS':
        return pco is not None and pco.get('card_name') == '__nazgul_moves__'
    if kind == 'CARD_CHOICE':
        return (pco is not None
                or getattr(w, 'pending_card_followup', None) is not None)
    return bool(types & {'card_option', 'army_move', 'army_attack'})


# ---------------------------------------------------------------------
# Setting up the engine's decision for each kind of question
# ---------------------------------------------------------------------

def setup_action(bot, st, dec, view):
    st.pending_combat = None
    st.dice_rolled = True
    return ('action', None)


def setup_fellowship(bot, st, dec, view):
    fs = st.fellowship
    # Entering Mordor: the engine's entry is positional (a Fellowship at
    # a gateway enters at the turn start, game_logic._check_mordor_entry)
    enter = opts(dec, 'enter_mordor')
    if enter and gl.fellowship_region(st) in GATEWAYS:
        return Match(enter[0]['o'])
    # The engine's own AI seat declares straight into a gateway in reach
    # (ai._auto_enter_mordor, a sanctioned hardcode of the self-play loop)
    w = deep_copy_state(st)
    try:
        if ai._auto_enter_mordor(w):
            o = first(opts(dec, 'declare', region=bot.slug(gl.fellowship_region(w))))
            if o is not None:
                return Match(o)
    except Exception:
        pass
    if not opts(dec, 'declare') or not fs.hidden:
        return Match(first(opts(dec, 'done')) if opts(dec, 'done') else None)
    try:
        if not ai._get_declare_actions(st, allow_in_place=True):
            return Match(first(opts(dec, 'done')))
    except Exception:
        return None
    return ('declare', '_declare_phase')


def setup_hunt_allocation(bot, st, dec, view):
    return ('eye', '_eye_allocation_phase')


def setup_discard(bot, st, dec, view):
    st.pending_discard = {'side': bot.memo.side}
    return ('discard', None)


def _sail_in(bot, st, b, frm, reg, atk):
    """Corsairs of Umbar: the referee's battle is fought from Umbar (the
    attackers advance by themselves on a win); the engine moves the Army
    into the region first and fights there (event_effects
    _effect_corsairs_of_umbar: attacker_region = defender_region).  Moves
    the attacking figures of [frm] into [reg] as that effect does and
    returns the frozen Shadow garrison of [reg] for a relief landing (a
    besieged Shadow Stronghold: the garrison stays inside, the landed
    force fights the besiegers), else None."""
    from army_card_cargo import inventory, freeze
    from combat import garrison_side_for
    names = bot.names
    src = army_at(st, frm)
    if src is None:
        return None
    relief = None
    if (reg in st.besieged_strongholds
            and garrison_side_for(st, reg) is atk and army_at(st, reg)):
        relief = freeze(inventory(st, reg))
    want = None
    if b.get('attackers'):
        want = {}
        for p in b['attackers']:
            k = (p.get('nation'), p.get('unit'))
            want[k] = want.get(k, 0) + int(p.get('n', 0))

    def take(k):
        # without the referee's list: every figure of the attacker
        if want is None:
            return True
        if want.get(k, 0) > 0:
            want[k] -= 1
            return True
        return False

    dst = gl.get_or_create_army(st, reg)
    keep_units = []
    for u in src.units:
        unit = 'regular' if u.kind is gs.UnitKind.REGULAR else 'elite'
        if (ai._side_of_nation(u.nation) is atk
                and take((names.nation_out[u.nation], unit))):
            dst.units.append(u)
        else:
            keep_units.append(u)
    keep_leaders = []
    for ld in src.leaders:
        tn = type(ld).__name__
        if gl._leader_side(ld) is not atk or gl._is_saruman(ld):
            go = False
        elif tn == 'Nazgul':
            go = take(('SAURON', 'nazgul'))
        elif tn == 'Leader':
            go = take((names.nation_out[ld.nation], 'leader'))
        else:
            go = True           # a Minion goes with the Army
        (dst.leaders if go else keep_leaders).append(ld)
    src.units, src.leaders = keep_units, keep_leaders
    if not (src.units or src.leaders):
        st.armies = dict((k, a) for k, a in st.armies.items() if a is not src)
    return relief


def _battle(bot, st, view, phase, rounds_fought):
    """The engine's pending_combat for the referee's battle."""
    names = bot.names
    b = view.get('battle')
    if not b:
        return None
    atk = side_in(b.get('attacker'))
    frm = names.region(b.get('from'))
    reg = names.region(b.get('region'))
    if frm is None or reg is None:
        return None
    kind = b.get('battle')
    meta = board.regions.get(reg, {})
    sailed = frm != reg and reg not in BOARD.adjacent(frm)
    relief = None
    if sailed:
        # only Corsairs of Umbar attacks a region not adjacent
        relief = _sail_in(bot, st, b, frm, reg, atk)
        pre_siege = reg in st.besieged_strongholds
        frm = reg
    pc = {'attacker_region': frm, 'defender_region': reg,
          'attacker_side': atk, 'defender_side': other(atk),
          'is_siege': kind == 'siege',
          'stronghold_region': bool(meta.get('is_stronghold')),
          'phase': phase, 'deciding_side': bot.memo.side,
          'rounds_fought': int(rounds_fought), 'shine_asked': True,
          'atk_casualty_policy': 'preserve_elites',
          'def_casualty_policy': 'preserve_elites'}
    if b.get('white_rider'):
        pc['gtw_shine'] = True
    if sailed:
        # as the engine's own landing: the Shadow may not cease the
        # battle, unless it joins a siege already under way
        pc['attacker_no_retreat'] = not pre_siege or relief is not None
        if relief is not None:
            pc['is_siege'] = False
            pc['corsairs_relief_garrison'] = relief
    # A rearguard: the units of the attacker's region not in the attack
    if frm != reg and b.get('attackers'):
        want = {}
        for p in b['attackers']:
            k = (p.get('nation'), p.get('unit'))
            want[k] = want.get(k, 0) + int(p.get('n', 0))
        army = army_at(st, frm)
        if army is not None:
            keep_units, rg_units = [], []
            for u in army.units:
                if ai._side_of_nation(u.nation) is not atk:
                    keep_units.append(u)
                    continue
                unit = 'regular' if u.kind is gs.UnitKind.REGULAR else 'elite'
                k = (names.nation_out[u.nation], unit)
                if want.get(k, 0) > 0:
                    want[k] -= 1
                    keep_units.append(u)
                else:
                    rg_units.append(u)
            if rg_units and any(ai._side_of_nation(u.nation) is atk
                                for u in keep_units):
                army.units = keep_units
                pc['rearguard_units'] = rg_units
    st.pending_combat = pc
    return pc


def setup_combat_card(bot, st, dec, view):
    b = view.get('battle') or {}
    me = bot.memo.side
    attacker = side_in(b.get('attacker')) is me
    rnd = int(b.get('round', 1))
    pc = _battle(bot, st, view, 'attacker_combat_card' if attacker
                 else 'defender_combat_card', rnd - 1)
    if pc is None:
        return None
    if not attacker:
        back = b.get('attacker_card_back')
        shown = b.get('attacker_card')
        if shown:
            pc['atk_combat_card'] = bot.names.card(shown)
        elif back:
            # committed face down: the engine's search blinds it
            # (ai._SimOverlay samples a card of that deck)
            pc['atk_combat_card'] = ai._HIDDEN_CC
            pc['atk_hidden_cc'] = {'deck': back}
        else:
            pc['atk_combat_card'] = None
    return ('action', None)


def setup_casualty(bot, st, dec, view):
    b = view.get('battle') or {}
    ctx = dec.get('context') or {}
    me = bot.memo.side
    attacker = side_in(b.get('attacker')) is me
    pc = _battle(bot, st, view, 'casualty_allocation', int(b.get('round', 1)))
    if pc is None:
        return None
    hits = int(ctx.get('hits', 1))
    pc['alloc_stage'] = 'attacker' if attacker else 'defender'
    pc['atk_hits'] = hits if attacker else 0
    pc['def_hits'] = 0 if attacker else hits
    return ('action', None)


def setup_cease(bot, st, dec, view):
    b = view.get('battle') or {}
    siege = b.get('battle') == 'siege'
    pc = _battle(bot, st, view, 'combat_round_result' if siege
                 else 'attacker_continue', int(b.get('round', 1)))
    return ('action', None) if pc is not None else None


def setup_extend_siege(bot, st, dec, view):
    b = view.get('battle') or {}
    pc = _battle(bot, st, view, 'combat_round_result', int(b.get('round', 1)))
    return ('action', None) if pc is not None else None


def setup_retreat(bot, st, dec, view):
    b = view.get('battle') or {}
    pc = _battle(bot, st, view, 'defender_retreat', int(b.get('round', 1)))
    return ('action', None) if pc is not None else None


def setup_field_or_siege(bot, st, dec, view):
    b = view.get('battle') or {}
    rnd = int(b.get('round', 1))
    if rnd <= 1:
        pc = _battle(bot, st, view, 'defender_siege_choice', 0)
    else:
        # the engine asks it with the retreat after the round before
        pc = _battle(bot, st, view, 'defender_retreat', rnd - 1)
    return ('action', None) if pc is not None else None


def setup_advance(bot, st, dec, view):
    b = view.get('battle') or {}
    pc = _battle(bot, st, view, 'victor_advance', int(b.get('round', 1)))
    if pc is None:
        return None
    pc['is_siege'] = False
    return ('action', None)


def setup_white_rider(bot, st, dec, view):
    b = view.get('battle') or {}
    pc = _battle(bot, st, view, 'fp_shine', int(b.get('round', 1)) - 1)
    if pc is None:
        return None
    pc.pop('shine_asked', None)
    return ('action', None)


def setup_hunt_casualty(bot, st, dec, view):
    ctx = dec.get('context') or {}
    st.pending_hunt = {'damage': int(ctx.get('damage', 1)),
                       'phase': 'hunt_sacrifice', 'casualty_taken': False}
    st._pending_hunt_damage = 0
    return ('action', None)


def setup_guide_ability(bot, st, dec, view):
    ctx = dec.get('context') or {}
    g = st.fellowship.guide
    name = getattr(g, 'name', '')
    if name == 'Gollum':
        st.pending_hunt = {'damage': int(ctx.get('damage', 1)),
                           'phase': 'hunt_gollum', 'casualty_taken': False}
    elif getattr(g, 'guide_ability', None) == 'sacrifice_reduce_hunt':
        st.pending_hunt = {'damage': int(ctx.get('damage', 1)),
                           'phase': 'hunt_sacrifice', 'casualty_taken': False}
    else:
        return None
    st._pending_hunt_damage = 0
    return ('action', None)


def setup_reveal(bot, st, dec, view):
    st.fellowship.hidden = False
    st._reveal_place_due = True
    st._pending_hunt_damage = 0
    st._sh_tile_drawn_regions = set()
    return ('action', None)


def setup_choose_guide(bot, st, dec, view):
    cands = []
    for o in opts(dec, 'guide'):
        n = CHAR_IN.get(o.get('character'))
        if n:
            cands.append(n)
    if not cands:
        return None
    st.pending_guide_change = {'candidates': cands}
    return ('action', None)


def setup_card_choice(bot, st, dec, view):
    """A card question outside the bot's own card play: the ones the
    engine asks as decisions of its own."""
    ctx = dec.get('context') or {}
    me = bot.memo.side
    b = view.get('battle')
    regions = [o for o in opts(dec, 'card_choice') if o.get('region')]
    if b and regions and 'retreat' in str(ctx.get('question', '')).lower():
        # a Combat card's retreat (Grimbeorn, The Red Arrow, The Spirit
        # of Mordor): the engine's mandatory card retreat
        attacker = side_in(b.get('attacker')) is me
        rnd = int(b.get('round', 1))
        pc = _battle(bot, st, view, 'attacker_card_retreat' if attacker
                     else 'defender_retreat', rnd - 1)
        if pc is not None:
            pc['_card_retreat_offered'] = True
            pc['_card_retreat_mandatory'] = True
            return ('action', None)
    counts = opts(dec, 'card_choice')
    card = bot.names.card(ctx.get('card'))
    if b and counts and all(o.get('count') is not None for o in counts) and card:
        # a Combat card's own choice (Relentless Assault, Dread and
        # Despair, Heroic Death ...): the engine's combat_effect_choice
        import combat as _cb
        cobj = cards_mod.get_card(card)
        kind = _cb.combat_effect_kind(getattr(cobj, 'combat_name', '') or '')
        attacker = side_in(b.get('attacker')) is me
        if kind:
            rnd = int(b.get('round', 1))
            pc = _battle(bot, st, view, 'combat_effect_choice', rnd - 1)
            if pc is not None:
                role = 'atk' if attacker else 'def'
                pc['effect_choice'] = {
                    'stage': 'post' if kind == 'onslaught' else 'pre',
                    'owner': role, 'effect': kind, 'card_id': card,
                    'counts': [int(o['count']) for o in counts],
                    'max_count': max(int(o['count']) for o in counts)}
                pc[role + '_combat_card'] = card
                return ('action', None)
    draws = opts(dec, 'draw')
    if draws and me is SH:
        # the Palantir of Orthanc's bonus draw (ai._open_palantir_deck_choice)
        st.pending_card_option = {
            'card_id': '__palantir_draw__', 'card_name': 'Palantir draw',
            'side': 'sh', 'decider': 'sh', 'direct': True,
            'palantir_draw': True,
            'options': [{'palantir_deck': o.get('deck')} for o in draws]}
        return ('action', None)
    if me is FP and 'damage' in ctx and opts(dec, 'card_choice', skip=True):
        # Axe and Bow / Horn of Gondor against the Hunt damage
        # (game_logic._offer_hunt_reduction)
        for o in opts(dec, 'card_choice'):
            cid = bot.names.card(o.get('target_card'))
            if cid is None:
                continue
            title = cards_mod.get_card(cid).card_name
            if plain(title) in (plain('Axe and Bow'), plain('Horn of Gondor')):
                st._pending_hunt_damage = int(ctx.get('damage', 1))
                st.pending_card_option = {
                    'card_id': cid, 'card_name': title, 'side': 'fp',
                    'decider': 'fp', 'direct': True, 'hunt_reduction': True,
                    'options': [{'use_hunt_reduction': 1, 'card_id': cid},
                                {'use_hunt_reduction': 0}]}
                return ('action', None)
    return None


SETUPS = {
    'CARD_CHOICE': setup_card_choice,
    'ACTION': setup_action,
    'FELLOWSHIP_PHASE': setup_fellowship,
    'HUNT_ALLOCATION': setup_hunt_allocation,
    'DISCARD_EXCESS_CARD': setup_discard,
    'COMBAT_CARD': setup_combat_card,
    'CASUALTY': setup_casualty,
    'CEASE_ATTACK': setup_cease,
    'EXTEND_SIEGE': setup_extend_siege,
    'RETREAT': setup_retreat,
    'FIELD_OR_SIEGE': setup_field_or_siege,
    'ADVANCE': setup_advance,
    'WHITE_RIDER': setup_white_rider,
    'HUNT_CASUALTY': setup_hunt_casualty,
    'GUIDE_ABILITY': setup_guide_ability,
    'REVEAL_FELLOWSHIP': setup_reveal,
    'CHOOSE_GUIDE': setup_choose_guide,
}


# ---------------------------------------------------------------------
# Matching an engine action to the referee's option, per question
# ---------------------------------------------------------------------

def _die_options(bot, st, dec, a, actions, **fields):
    cand = []
    for name in actions:
        cand += opts(dec, 'use_die', die=a.die_idx, action=name, **fields)
    return cand


def m_action(bot, st, dec, a):
    t = a.action_type
    p = a.params or {}
    names = bot.names
    if t == 'pass':
        o = first(opts(dec, 'pass'))
        return Match(o) if o is not None else None
    if t == 'elven_ring':
        face = next((f for f in gs.ActionDieFace if f.value == p.get('new_face')),
                    None)
        o = first(opts(dec, 'elven_ring', die=p.get('target_die_idx'),
                       to=names.face_out.get(face)))
        return Match(o) if o is not None else None
    if t not in DIE_ACTIONS:
        return None
    acts = DIE_ACTIONS[t]
    if t == 'draw_card':
        return _m(_die_options(bot, st, dec, a, acts, deck=p.get('deck')))
    if t == 'play_event':
        cand = _die_options(bot, st, dec, a, acts,
                            card=names.card_key(p.get('card_id')))
        return _m(cand, chain=True)
    if t == 'advance_political':
        nat = names.nation_out.get(p.get('nation'))
        if p.get('companion_ability'):
            cand = _die_options(bot, st, dec, a, ('companion_politics',),
                                character=CHAR_OUT.get(p['companion_ability']))
            cand = cand or _die_options(bot, st, dec, a, ('diplomacy',),
                                        nation=nat)
        else:
            cand = _die_options(bot, st, dec, a, ('diplomacy',), nation=nat)
        return _m(cand)
    if t == 'muster_minion':
        key = CHAR_OUT.get(p.get('minion'))
        return _m(_die_options(bot, st, dec, a, acts, character=key,
                               region=bot.slug(p.get('region'))))
    if t == 'transform_gandalf':
        return _m(_die_options(bot, st, dec, a, acts,
                               character='gandalf-the-white',
                               region=bot.slug(p.get('region'))))
    if t == 'crown_aragorn':
        return _m(_die_options(bot, st, dec, a, acts, character='aragorn'))
    if t in ('fellowship_move', 'fellowship_hide'):
        if t == 'fellowship_hide' and p.get('strider_ability'):
            acts = ('hide_fellowship_strider', 'hide_fellowship')
        return _m(_die_options(bot, st, dec, a, acts))
    if t == 'discard_table_card':
        cand = _die_options(bot, st, dec, a, acts,
                            card=names.card_key(p.get('card_id')))
        ring = p.get('payment') == 'ring'
        pick = [o for o in cand if bool(o.get('elven_ring')) == ring]
        return _m(pick or cand)
    me = bot.memo.side
    face = st.dice[me][a.die_idx].current if 0 <= a.die_idx < len(st.dice[me]) else None
    if t == 'army_move':
        order = acts if face is not gs.ActionDieFace.CHARACTER else acts[::-1]
        cand = _die_options(bot, st, dec, a, order[:1]) or \
            _die_options(bot, st, dec, a, order[1:])
        if not cand:
            return None
        return Match(cand[0]['o'], lambda: army_move_plan(bot, st, a),
                     chain=True)
    if t == 'army_attack':
        order = acts if face is not gs.ActionDieFace.CHARACTER else acts[::-1]
        cand = _die_options(bot, st, dec, a, order[:1]) or \
            _die_options(bot, st, dec, a, order[1:])
        if not cand:
            return None
        return Match(cand[0]['o'], AttackPlan(bot.slug(p.get('from')),
                                              bot.slug(p.get('to'))))
    if t in ('muster_recruit', 'muster_elite'):
        cand = _die_options(bot, st, dec, a, acts)
        if not cand:
            return None
        return Match(cand[0]['o'], recruit_plan(bot, st, a), chain=True)
    if t == 'companion_move':
        cand = _die_options(bot, st, dec, a, acts)
        if not cand:
            return None
        return Match(cand[0]['o'], companion_plan(bot, st, a), chain=True)
    if t == 'companion_separate':
        cand = _die_options(bot, st, dec, a, acts)
        if not cand:
            return None
        who = p.get('companions') or [p.get('companion')]
        return Match(cand[0]['o'], SeparatePlan(
            [CHAR_OUT.get(n) for n in who if CHAR_OUT.get(n)],
            bot.slug(p.get('to_region'))))
    if t in ('nazgul_move', 'mouth_move'):
        cand = _die_options(bot, st, dec, a, acts)
        if not cand:
            return None
        return Match(cand[0]['o'], minion_plan(bot, st, a), chain=True)
    return None


def _m(cand, plan=None, chain=False):
    if not cand:
        return None
    return Match(cand[0]['o'], plan, chain)


def army_move_plan(bot, st, a, advance=False):
    p = a.params or {}
    me = bot.memo.side
    frm, to = p.get('from'), p.get('to')
    w = deep_copy_state(st)
    try:
        ai._execute_action(w, a)
    except Exception:
        return None
    moved, chars, before, bchars = moved_figures(st, w, frm, me, bot.names)
    if not moved and not chars:
        return None
    # Saruman never moves: not counted for 'all'
    movable_chars = set(c for c in bchars if c != 'saruman')
    everything = (moved == before and chars == movable_chars)
    return ArmyMovePlan(bot.slug(frm), bot.slug(to), moved, chars, everything,
                        advance=advance)


def recruit_plan(bot, st, a):
    p = a.params or {}
    nat = bot.names.nation_out.get(p.get('nation'))
    if a.action_type == 'muster_elite':
        unit = 'elite'
    else:
        unit = str(p.get('type', 'regular'))
        if unit == 'leader' and nat == 'SAURON':
            unit = 'nazgul'
    return RecruitPlan(bot.slug(p.get('region')), nat, unit)


def companion_plan(bot, st, a):
    p = a.params or {}
    frm = p.get('from') or p.get('from_region')
    to = p.get('to') or p.get('to_region')
    who = p.get('companions')
    if not who:
        army = army_at(st, frm)
        who = [ld.name for ld in (army.leaders if army else [])
               if isinstance(ld, gs.Companion)]
    return CompanionMovePlan([CHAR_OUT.get(n) for n in who if CHAR_OUT.get(n)],
                             bot.slug(frm), bot.slug(to))


def minion_plan(bot, st, a):
    p = a.params or {}
    frm = p.get('from') or p.get('from_region')
    to = p.get('to') or p.get('to_region')
    f, t = bot.slug(frm), bot.slug(to)
    flights = []
    if a.action_type == 'mouth_move':
        flights.append(('mouth-of-sauron', f, t))
    else:
        army = army_at(st, frm)
        n = sum(1 for ld in (army.leaders if army else [])
                if isinstance(ld, gs.Nazgul))
        if p.get('count') is not None:
            n = min(n, int(p['count']))
        if p.get('include_wk'):
            flights.append(('witch-king', f, t))
        flights += [('nazgul', f, t)] * n
    return MinionPlan(flights)


def m_army_move(bot, st, dec, a):
    """The Army die's second move (or a card's follow-up move)."""
    t = a.action_type
    if t in ('pass', 'followup_done'):
        return _m(opts(dec, 'end'))
    if t == 'card_option':
        # a card's own Army move (The Shadow is Moving ...)
        ch = (a.params or {}).get('choices') or {}
        if ch.get('_done') or not ch:
            return _m(opts(dec, 'end'), chain=True)
        f = bot.slug(ch.get('from_region') or ch.get('from'))
        to = bot.slug(ch.get('to_region') or ch.get('to'))
        if f is None or to is None:
            return None
        cand = opts(dec, 'army_move', **{'from': f, 'to': to})
        return _m([o for o in cand if o.get('all')] or cand, chain=True)
    if t != 'army_move':
        return None
    p = a.params or {}
    cand = opts(dec, 'army_move', **{'from': bot.slug(p.get('from')),
                                     'to': bot.slug(p.get('to'))})
    if not cand:
        return None
    def make():
        plan = army_move_plan(bot, st, a)
        if plan is not None:
            plan.started = True
        return plan
    def resolve(plan):
        # 'all' or not: read off the engine's move
        if plan is None:
            return None
        pick = [o for o in cand if bool(o.get('all')) == plan.everything]
        return (pick or cand)[0]['o']
    return Match(cand[0]['o'], make, chain=True, resolve=resolve)


def m_recruit(bot, st, dec, a):
    t = a.action_type
    if t == 'pass':
        return _m(opts(dec, 'end'))
    if t not in ('muster_recruit', 'muster_elite'):
        return None
    plan = recruit_plan(bot, st, a)
    cand = opts(dec, 'recruit', region=plan.region, nation=plan.nation,
                unit=plan.unit)
    return _m(cand, chain=True)


def m_move_companions(bot, st, dec, a):
    if a.action_type == 'card_option':
        ch = (a.params or {}).get('choices') or {}
        if ch.get('_comp_done'):
            return _m(opts(dec, 'end'))
        if ch.get('from_region') and ch.get('companions'):
            plan = CompanionMovePlan(
                [CHAR_OUT.get(n) for n in ch['companions'] if CHAR_OUT.get(n)],
                bot.slug(ch['from_region']), bot.slug(ch['to_region']))
            o = plan.answer(dec)
            return Match(o, plan, chain=True) if o is not None else None
    return None


def m_move_minions(bot, st, dec, a):
    if a.action_type == 'card_option':
        ch = (a.params or {}).get('choices') or {}
        if ch.get('_naz_done'):
            return _m(opts(dec, 'end'))
        f, t = bot.slug(ch.get('from_region')), bot.slug(ch.get('to_region'))
        if f is None or t is None:
            return None
        fig = 'witch-king' if ch.get('include_wk') else 'nazgul'
        plan = MinionPlan([(fig, f, t)])
        o = plan.answer(dec)
        return Match(o, plan, chain=True) if o is not None else None
    if a.action_type in ('nazgul_move', 'mouth_move'):
        plan = minion_plan(bot, st, a)
        o = plan.answer(dec)
        return Match(o, plan, chain=True) if o is not None else None
    return None


def m_fellowship(bot, st, dec, a):
    if a.action_type == 'fellowship_declare':
        return _m(opts(dec, 'declare', region=bot.slug((a.params or {}).get('region'))))
    if a.action_type == 'pass':
        return _m(opts(dec, 'done'))
    return None


def m_hunt_allocation(bot, st, dec, a):
    if a.action_type == 'eye_allocate':
        return _m(opts(dec, 'hunt_dice', count=int((a.params or {}).get('count', 0))))
    return None


def m_discard(bot, st, dec, a):
    if a.action_type == 'discard_card':
        return _m(opts(dec, 'discard', card=bot.names.card_key((a.params or {}).get('card_id'))))
    return None


def m_combat_card(bot, st, dec, a):
    if a.action_type == 'combat_card_play':
        return _m(opts(dec, 'combat_card',
                       card=bot.names.card_key((a.params or {}).get('card_id'))))
    if a.action_type == 'combat_card_none':
        return _m(opts(dec, 'none'))
    return None


def m_casualty(bot, st, dec, a):
    if a.action_type != 'casualty_allocate':
        return None
    k = int((a.params or {}).get('count', 0))
    plan = CasualtyPlan(casualty_key(dec), k)
    o = casualty_pick(dec, k > 0, plan)
    return Match(o, plan) if o is not None else None


def m_cease(bot, st, dec, a):
    t = a.action_type
    if t in ('combat_continue', 'siege_press'):
        return _m(opts(dec, 'continue'))
    if t in ('combat_cease', 'siege_no_press'):
        return _m(opts(dec, 'cease'))
    return None


def m_extend_siege(bot, st, dec, a):
    t = a.action_type
    if t == 'siege_press':
        cand = opts(dec, 'extend')
        return _m(cand)
    if t == 'siege_no_press':
        return _m(opts(dec, 'stop'))
    return None


def m_retreat(bot, st, dec, a):
    t = a.action_type
    if t == 'combat_retreat':
        return _m(opts(dec, 'retreat', region=bot.slug((a.params or {}).get('to_region'))))
    if t == 'combat_no_retreat':
        return _m(opts(dec, 'stand'))
    return None


def m_field_or_siege(bot, st, dec, a):
    t = a.action_type
    if t == 'retreat_into_siege':
        return _m(opts(dec, 'siege'))
    if t == 'combat_no_retreat':
        return _m(opts(dec, 'field'), RetreatPlan(None))
    if t == 'combat_retreat':
        to = bot.slug((a.params or {}).get('to_region'))
        return _m(opts(dec, 'field'), RetreatPlan(to))
    return None


def m_advance(bot, st, dec, a):
    t = a.action_type
    if t == 'victor_stay':
        return _m(opts(dec, 'stay'))
    if t != 'victor_advance':
        return None
    plan = army_move_plan(bot, st, a, advance=True)
    if plan is None:
        return None
    o = plan.answer(dec)
    return Match(o, plan) if o is not None else None


def m_white_rider(bot, st, dec, a):
    if a.action_type == 'shine':
        return _m(opts(dec, 'use'))
    if a.action_type == 'shine_none':
        return _m(opts(dec, 'skip'))
    return None


def m_hunt_casualty(bot, st, dec, a):
    t = a.action_type
    if t == 'hunt_take_corruption':
        return _m(opts(dec, 'corruption'))
    if t == 'hunt_sacrifice_guide':
        return _m(opts(dec, 'guide'))
    if t == 'hunt_sacrifice_random':
        return _m(opts(dec, 'random'))
    return None


def m_guide_ability(bot, st, dec, a):
    t = a.action_type
    if t in ('hunt_gollum_reveal', 'hunt_guide_separate'):
        return _m(opts(dec, 'use'))
    if t in ('hunt_gollum_no_reveal', 'hunt_take_corruption',
             'hunt_sacrifice_guide', 'hunt_sacrifice_random'):
        plan = None
        if t != 'hunt_gollum_no_reveal':
            okind = {'hunt_take_corruption': 'corruption',
                     'hunt_sacrifice_guide': 'guide',
                     'hunt_sacrifice_random': 'random'}[t]
            plan = AnswerPlan('HUNT_CASUALTY', okind)
        return _m(opts(dec, 'skip'), plan)
    return None


def m_reveal(bot, st, dec, a):
    if a.action_type == 'fellowship_reveal_place':
        return _m(opts(dec, 'reveal', region=bot.slug((a.params or {}).get('region'))))
    return None


def m_choose_guide(bot, st, dec, a):
    if a.action_type == 'choose_guide':
        return _m(opts(dec, 'guide', character=CHAR_OUT.get((a.params or {}).get('companion'))))
    return None


# -- a card's own choices -------------------------------------------------

DECLINE_WORDS = ('none', 'skip', 'decline', 'no', 'pass', 'done', 'stay')


def choice_tokens(bot, v, out):
    """The protocol names a card option of the engine mentions."""
    names = bot.names
    if isinstance(v, dict):
        for k, x in v.items():
            if not str(k).startswith('_'):
                choice_tokens(bot, x, out)
        return
    if isinstance(v, (list, tuple, set)):
        for x in v:
            choice_tokens(bot, x, out)
        return
    if isinstance(v, bool):
        out.add(('bool', v))
        return
    if isinstance(v, int):
        out.add(('int', v))
        return
    if hasattr(v, 'name') and hasattr(v, 'value'):
        n = names.nation_out.get(v)
        if n:
            out.add(('nation', n))
        return
    s = str(v)
    slug = names.slug(s)
    if slug:
        out.add(('region', slug))
    ck = CHAR_OUT.get(s)
    if ck:
        out.add(('character', ck))
    card = names.card_key(s)
    if card:
        out.add(('card', card))
    for nk, nat in names.nation_in.items():
        if plain(s) in (plain(nk), plain(nat.value), plain(nat.name)):
            out.add(('nation', nk))
    if s.lower() in DECLINE_WORDS:
        out.add(('decline', True))
    out.add(('word', plain(s)))


def m_card_choice(bot, st, dec, a):
    t = a.action_type
    if t == 'card_option':
        ch = (a.params or {}).get('choices') or {}
        if 'palantir_deck' in ch:
            return _m(opts(dec, 'draw', deck=ch['palantir_deck']))
        if 'use_hunt_reduction' in ch:
            if ch['use_hunt_reduction']:
                return _m([o for o in opts(dec, 'card_choice')
                           if bot.names.card(o.get('target_card')) == ch.get('card_id')])
            return _m(opts(dec, 'card_choice', skip=True))
        if ch.get('_comp_done') or ch.get('_naz_done'):
            o = first(opts(dec, 'end')) or first(opts(dec, 'card_choice', skip=True))
            return Match(o, chain=True) if o is not None else None
        toks = set()
        choice_tokens(bot, ch, toks)
        best = score_options(dec, toks, not ch)
        return Match(best, chain=True) if best is not None else None
    if t == 'followup_done':
        o = first(opts(dec, 'card_choice', skip=True)) or first(opts(dec, 'end'))
        return Match(o, chain=True) if o is not None else None
    if t == 'combat_effect_choice':
        return _m(opts(dec, 'card_choice', count=int((a.params or {}).get('count', 0))))
    if t == 'combat_retreat':
        return _m(opts(dec, 'card_choice',
                       region=bot.slug((a.params or {}).get('to_region'))))
    if t in ('army_move', 'army_attack'):
        p = a.params or {}
        f, to = bot.slug(p.get('from')), bot.slug(p.get('to'))
        cand = opts(dec, 'card_choice', **{'from': f, 'to': to})
        if cand:
            return Match(cand[0]['o'], chain=True)
        word = 'move' if t == 'army_move' else 'attack'
        cand = [o for o in opts(dec, 'card_choice')
                if str(o.get('answer', '')).lower() == word]
        if cand:
            plan = army_move_plan(bot, st, a) if t == 'army_move' else \
                AttackPlan(f, to)
            return Match(cand[0]['o'], plan, chain=True)
        return None
    return None


def score_options(dec, toks, empty):
    """The referee's card answer that best fits the engine's choice."""
    best = None
    decline = ('decline', True) in toks or empty
    for o in dec.get('options') or []:
        s = 0
        for field, tk in (('region', 'region'), ('from', 'region'),
                          ('to', 'region'), ('character', 'character'),
                          ('nation', 'nation'), ('target_card', 'card')):
            if o.get(field) is not None:
                s += 3 if (tk, o[field]) in toks else -1
        if o.get('count') is not None:
            s += 3 if ('int', int(o['count'])) in toks else -1
        if 'yes' in o:
            s += 1 if bool(o['yes']) != decline else -1
        if o.get('skip'):
            s += 2 if decline else -2
        ans = str(o.get('answer', ''))
        for part in ans.replace('>', ' ').replace(':', ' ').split():
            if ('region', part) in toks or ('card', part) in toks:
                s += 1
            if ('word', plain(part)) in toks:
                s += 1
        if best is None or s > best[0]:
            best = (s, o['o'])
    if best is None or best[0] <= 0:
        return None
    return best[1]


def figure_counts(st, names):
    """{(region, nation, unit): n} and {(region, character)} of the map."""
    out = {}
    chars = set()
    for k in list(st.armies):
        slug = names.slug(k)
        for side in (FP, SH):
            f, c = figures(st, k, side, names)
            for (nat, unit), n in f.items():
                out[(slug, nat, unit)] = out.get((slug, nat, unit), 0) + n
            for ck in c:
                chars.add((slug, ck))
    return out, chars


UNIT_ANSWERS = {'REGULAR': 'regular', 'ELITE': 'elite', 'LEADER': 'leader',
                'NAZGUL': 'nazgul'}


def diff_answer(bot, dec):
    """A card question the engine settled inside its action: the answer
    whose words match its board after the action (figures added or
    removed, a region changed)."""
    if bot.work is None or bot.work_base is None or dec['kind'] not in (
            'CARD_CHOICE', 'ARMY_SELECTION', 'ARMY_MOVE'):
        return None
    if bot.diff is None:
        b, bc = figure_counts(bot.work_base, bot.names)
        w, wc = figure_counts(bot.work, bot.names)
        added, removed = {}, {}
        for k in set(b) | set(w):
            d = w.get(k, 0) - b.get(k, 0)
            if d > 0:
                added[k] = d
            elif d < 0:
                removed[k] = -d
        bot.diff = {'added': added, 'removed': removed,
                    'chars_to': wc - bc, 'chars_from': bc - wc}
    diff = bot.diff
    ctx = dec.get('context') or {}
    if dec['kind'] == 'ARMY_MOVE':
        # a card's Army moves the engine made inside the card
        if getattr(bot, 'work_action', None) != 'play_event':
            return None
        best = None
        for o in opts(dec, 'army_move'):
            f, t = o.get('from'), o.get('to')
            if (f, t) in diff.setdefault('moves_done', set()):
                continue
            s = (2 * any(k[0] == f and n > 0 for k, n in diff['removed'].items())
                 + 2 * any(k[0] == t and n > 0 for k, n in diff['added'].items()))
            if s >= 4 and (best is None or s > best[0]):
                best = (s, o['o'], (f, t))
        if best is not None:
            diff['moves_done'].add(best[2])
            return best[1]
        return first(opts(dec, 'end'))
    if dec['kind'] == 'ARMY_SELECTION':
        # a card's Army move: as many as left the region in the engine
        frm = ctx.get('from')
        best = None
        for o in opts(dec, 'count'):
            if o.get('character'):
                want = 1 if (frm, o['character']) in diff['chars_from'] else 0
            else:
                want = diff['removed'].get((frm, o.get('nation') or ctx.get(
                    'nation'), o.get('unit') or ctx.get('unit')), 0)
            d = abs(int(o.get('count', 0)) - want)
            if best is None or d < best[0]:
                best = (d, o['o'], o, want)
        if best is None:
            return None
        o = best[2]
        if not o.get('character'):
            k = (frm, o.get('nation') or ctx.get('nation'),
                 o.get('unit') or ctx.get('unit'))
            if k in diff['removed']:
                diff['removed'][k] = max(0, diff['removed'][k] - int(o.get('count', 0)))
        return best[1]
    words = str(ctx.get('question', '')).replace('?', ' ').split()
    region = next((wd for wd in words if wd in bot.names.region_in), None)
    nation = next((wd for wd in words if wd in bot.names.nation_in), None)
    unit_opts = [o for o in opts(dec, 'card_choice')
                 if str(o.get('answer', '')).upper() in UNIT_ANSWERS]
    if unit_opts:
        for o in unit_opts:
            unit = UNIT_ANSWERS[str(o['answer']).upper()]
            for k, n in diff['added'].items():
                if n > 0 and k[2] == unit and region in (None, k[0]) and \
                        nation in (None, k[1]):
                    diff['added'][k] = n - 1
                    return o['o']
        return None
    best = None
    for o in opts(dec, 'card_choice'):
        s = 0
        r = o.get('region')
        if r is not None:
            s += sum(n for k, n in diff['added'].items() if k[0] == r)
            s += sum(n for k, n in diff['removed'].items() if k[0] == r)
            s += sum(1 for k in diff['chars_to'] if k[0] == r)
        f, t = o.get('from'), o.get('to')
        path = str(o.get('answer', '')).split('>')
        if f is None and len(path) > 1:
            f, t = path[0], path[-1]
        if f is not None and t is not None:
            if any(k[0] == f for k in diff['removed']):
                s += 2
            if any(k[0] == t for k in diff['added']):
                s += 2
            # a path through regions: its steps count a little
            for mid in path[1:-1]:
                if any(k[0] == mid for k in diff['added']):
                    s += 1
        c = o.get('character')
        if c is not None:
            s += 2 * sum(1 for k in diff['chars_to'] if k[1] == c)
            s += sum(1 for k in diff['chars_from'] if k[1] == c)
        if s > 0 and (best is None or s > best[0]):
            best = (s, o['o'])
    return best[1] if best else None


MAPPERS = {
    'ACTION': m_action,
    'FELLOWSHIP_PHASE': m_fellowship,
    'HUNT_ALLOCATION': m_hunt_allocation,
    'DISCARD_EXCESS_CARD': m_discard,
    'COMBAT_CARD': m_combat_card,
    'CASUALTY': m_casualty,
    'CEASE_ATTACK': m_cease,
    'EXTEND_SIEGE': m_extend_siege,
    'RETREAT': m_retreat,
    'FIELD_OR_SIEGE': m_field_or_siege,
    'ADVANCE': m_advance,
    'WHITE_RIDER': m_white_rider,
    'HUNT_CASUALTY': m_hunt_casualty,
    'GUIDE_ABILITY': m_guide_ability,
    'REVEAL_FELLOWSHIP': m_reveal,
    'CHOOSE_GUIDE': m_choose_guide,
    'ARMY_MOVE': m_army_move,
    'RECRUIT': m_recruit,
    'MOVE_COMPANIONS': m_move_companions,
    'MOVE_MINIONS': m_move_minions,
    'CARD_CHOICE': m_card_choice,
}


# ---------------------------------------------------------------------
# Plain answers where the engine has nothing to say
# ---------------------------------------------------------------------

def rule_remove_excess(bot, dec, view):
    """a Regular of the Nation with the most units leaves first"""
    return first(opts(dec, 'remove_unit', unit='regular')) or first(
        opts(dec, 'remove_unit'))


def rule_card_choice(bot, dec, view):
    """a card that helps now is used (the engine settles it by itself)"""
    ctx = dec.get('context') or {}
    use = [o for o in opts(dec, 'card_choice') if not o.get('skip')]
    card = str(ctx.get('card') or '')
    mine = card.startswith('fp-' if bot.memo.side is FP else 'sh-')
    if mine and bot.work is None and opts(dec, 'card_choice', skip=True):
        # a table card of ours offering its effect (Worn with Sorrow and
        # Toil, Mithril Coat and Sting ...): the engine applies these by
        # itself, so the bot uses them
        return first(use)
    if 'damage' in ctx or 'tile' in ctx or 'hunt' in str(
            ctx.get('question', '')).lower():
        # a Free Peoples card against the Hunt (Mithril Coat and Sting:
        # draw a second tile instead of an Eye; the engine plays it as
        # its own Hunt rule)
        return first(use)
    q = str(ctx.get('question', '')).lower()
    answers = dict((str(o.get('answer', '')).lower(), o['o'])
                   for o in opts(dec, 'card_choice'))
    f = view.get('fellowship') or {}
    if 'corruption' in answers and 'eliminate' in answers:
        # Lure of the Ring (the Shadow's card, the Free Peoples choose):
        # the Corruption, unless it brings the Ring-bearers near the end
        n = next((int(w) for w in q.split() if w.isdigit()), 1)
        if int(f.get('corruption', 0)) + n >= 10:
            return answers['eliminate']
        return answers['corruption']
    if 'hand' in answers and 'skip' in answers:
        # Worn with Sorrow and Toil: a card from a hand of three or more,
        # else the Corruption
        hand = len((view.get('cards') or {}).get('hand') or [])
        return answers['hand'] if hand >= 3 else answers['skip']
    chars = [o for o in opts(dec, 'card_choice') if o.get('character')]
    if chars and 'separate' in q:
        # The Breaking of the Fellowship: the Companion of the lowest
        # level leaves (the Guide last)
        level = {'gandalf-the-grey': 3, 'strider': 3, 'boromir': 2,
                 'legolas': 2, 'gimli': 2, 'meriadoc': 1, 'peregrin': 1}
        guide = f.get('guide')
        return min(chars, key=lambda o: (o['character'] == guide,
                                         level.get(o['character'], 2)))['o']
    draws = opts(dec, 'draw')
    if draws:
        # a bonus draw of the Free Peoples (Gandalf the Grey as Guide):
        # the deck with more cards left
        decks = ((view.get('cards') or {}).get('decks') or {}).get(
            side_out(bot.memo.side)) or {}
        return max(draws, key=lambda o: decks.get(o.get('deck'), 0))['o']
    return None


def rule_end(bot, dec, view):
    """the engine has no further move of this kind: end"""
    return first(opts(dec, 'end'))


def rule_join(bot, dec, view):
    """a Companion asked outside the engine's move stays"""
    return first(opts(dec, 'stay'))


def rule_separate_to(bot, dec, view):
    """a Companion leaving the Fellowship by the rules (not the engine's
    choice) goes where a Free Peoples Army stands, else nearest the
    Fellowship"""
    regions = view.get('regions') or {}
    fp = set(n for n in bot.names.nation_in if n in (
        'DWARVES', 'ELVES', 'GONDOR', 'NORTH', 'ROHAN'))
    cand = opts(dec, 'separate_to')
    for o in cand:
        r = regions.get(o.get('to')) or {}
        if any(u.get('nation') in fp and u.get('unit') in ('regular', 'elite')
               for u in r.get('units') or []):
            return o['o']
    here = (view.get('fellowship') or {}).get('region')
    return nearest(cand, here) if cand else None


def rule_choose_attack(bot, dec, view):
    """a card's attack with one target: the whole Army attacks (the
    engine has no rearguard)"""
    cand = opts(dec, 'attack')
    pairs = set((o.get('from'), o.get('to')) for o in cand)
    if len(pairs) != 1:
        return None
    return first([o for o in cand if o.get('all', True)] or cand)


def rule_casualty(bot, dec, view):
    """a hit outside a battle (a card's): a Regular first, an Elite
    reduced before one is removed"""
    return casualty_pick(dec, False)


RULES = {
    'CASUALTY': rule_casualty,
    'CHOOSE_ATTACK': rule_choose_attack,
    'SEPARATE_COMPANIONS': rule_separate_to,
    'REMOVE_EXCESS_UNIT': rule_remove_excess,
    'CARD_CHOICE': rule_card_choice,
    'MOVE_MINIONS': rule_end,
    'MOVE_COMPANIONS': rule_end,
    'JOIN_GROUP': rule_join,
}


def default_option(dec):
    """A sensible legal option without the engine: (o, rule)."""
    kind = dec['kind']
    options = dec.get('options') or []
    if not options:
        return 0, 'none'

    def pick(*kinds):
        for k in kinds:
            c = opts(dec, k)
            if c:
                return c[0]['o']
        return None

    if kind == 'ACTION':
        o = pick('use_die')
        return (o, 'first die action') if o is not None else (options[0]['o'], 'first')
    if kind == 'REMOVE_EXCESS_UNIT':
        c = opts(dec, 'remove_unit', unit='regular')
        if c:
            return c[0]['o'], 'a Regular first'
    if kind in ('ARMY_MOVE', 'MOVE_COMPANIONS', 'MOVE_MINIONS', 'RECRUIT'):
        o = pick('end')
        if o is not None:
            return o, 'end'
    if kind == 'ARMY_SELECTION':
        c = opts(dec, 'count')
        if c:
            return max(c, key=lambda x: int(x.get('count', 0)))['o'], 'all of them'
    if kind == 'JOIN_GROUP':
        o = pick('stay')
        if o is not None:
            return o, 'stay'
    if kind == 'CASUALTY':
        o = casualty_pick(dec, False)
        if o is not None:
            return o, 'a Regular first'
    if kind == 'COMBAT_CARD':
        o = pick('none')
        if o is not None:
            return o, 'no card'
    if kind == 'CARD_CHOICE':
        c = [x for x in opts(dec, 'card_choice') if not x.get('skip')]
        if c:
            return c[0]['o'], 'first answer'
    if kind == 'GUIDE_ABILITY':
        o = pick('skip')
        if o is not None:
            return o, 'skip'
    return options[0]['o'], 'option 0'


# ---------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------

class Stats(object):
    def __init__(self):
        self.decisions = 0
        self.by_how = {}
        self.defaults = {}
        self.secs = 0.0
        self.max_secs = 0.0
        self.menus = 0
        self.unmatched = 0
        self.timeouts = 0
        self.illegal = 0
        self.checked = False

    def count(self, kind, how, secs, info):
        self.decisions += 1
        self.by_how[how] = self.by_how.get(how, 0) + 1
        if how in ('default', 'error-default') or info.get('fallback'):
            key = '%s: %s' % (kind, info.get('note') or info.get('fallback')
                              or info.get('error') or '?')
            self.defaults[key] = self.defaults.get(key, 0) + 1
        self.secs += secs
        self.max_secs = max(self.max_secs, secs)

    def menu(self, kind, n, unmatched):
        self.menus += n
        self.unmatched += unmatched

    def add(self, o):
        self.decisions += o.decisions
        for k, v in o.by_how.items():
            self.by_how[k] = self.by_how.get(k, 0) + v
        for k, v in o.defaults.items():
            self.defaults[k] = self.defaults.get(k, 0) + v
        self.secs += o.secs
        self.max_secs = max(self.max_secs, o.max_secs)
        self.menus += o.menus
        self.unmatched += o.unmatched

    def summary(self):
        avg = self.secs / self.decisions if self.decisions else 0.0
        parts = ['%d decisions (%s)' % (self.decisions, ', '.join(
            '%s %d' % kv for kv in sorted(self.by_how.items())))]
        parts.append('%.2f s average, %.1f s max' % (avg, self.max_secs))
        if self.menus:
            parts.append('engine actions without a counterpart: %d of %d'
                         % (self.unmatched, self.menus))
        parts.append('timeouts %d, illegal %d' % (self.timeouts, self.illegal))
        if self.defaults:
            parts.append('fallbacks: ' + '; '.join(
                '%s x%d' % kv for kv in sorted(self.defaults.items(),
                                               key=lambda kv: -kv[1])[:8]))
        return ', '.join(parts)


# ---------------------------------------------------------------------
# Transport: stdin/stdout, or a WebSocket the bot dials
# ---------------------------------------------------------------------

def run_stdio(bot, out):
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            m = json.loads(line)
        except ValueError:
            log('not JSON:', line[:200])
            continue
        answer = bot.handle(m)
        if answer is not None:
            out.write(json.dumps(answer, separators=(',', ':')) + '\n')
            out.flush()


class WebSocket(object):
    """A small WebSocket client (RFC 6455) on the standard library."""

    def __init__(self, url):
        parts = urlparse(url)
        secure = parts.scheme == 'wss'
        port = parts.port or (443 if secure else 80)
        raw = socket.create_connection((parts.hostname, port))
        self.sock = (ssl.create_default_context().wrap_socket(
            raw, server_hostname=parts.hostname) if secure else raw)
        key = base64.b64encode(os.urandom(16)).decode()
        path = (parts.path or '/') + ('?' + parts.query if parts.query else '')
        host = parts.hostname + ('' if parts.port is None else ':%d' % parts.port)
        request = ('GET %s HTTP/1.1\r\nHost: %s\r\nUpgrade: websocket\r\n'
                   'Connection: Upgrade\r\nSec-WebSocket-Key: %s\r\n'
                   'Sec-WebSocket-Version: 13\r\n\r\n') % (path, host, key)
        self.sock.sendall(request.encode())
        self.buffer = b''
        header = self._read_until(b'\r\n\r\n').decode('latin-1')
        if ' 101 ' not in header.split('\r\n')[0]:
            raise RuntimeError('the referee refused the connection: '
                               + header.split('\r\n')[0])

    def _read(self, n):
        while len(self.buffer) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise EOFError('connection closed')
            self.buffer += chunk
        data, self.buffer = self.buffer[:n], self.buffer[n:]
        return data

    def _read_until(self, marker):
        while marker not in self.buffer:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise EOFError('connection closed')
            self.buffer += chunk
        i = self.buffer.index(marker) + len(marker)
        data, self.buffer = self.buffer[:i], self.buffer[i:]
        return data

    def _frame(self, opcode, payload):
        header = bytes([0x80 | opcode])
        n = len(payload)
        if n < 126:
            header += bytes([0x80 | n])
        elif n < 65536:
            header += bytes([0x80 | 126]) + struct.pack('>H', n)
        else:
            header += bytes([0x80 | 127]) + struct.pack('>Q', n)
        mask = os.urandom(4)
        self.sock.sendall(header + mask + bytes(
            b ^ mask[i % 4] for i, b in enumerate(payload)))

    def send(self, text):
        self._frame(0x1, text.encode('utf-8'))

    def receive(self):
        message = b''
        while True:
            b0, b1 = self._read(2)
            opcode, n = b0 & 0x0F, b1 & 0x7F
            if n == 126:
                n = struct.unpack('>H', self._read(2))[0]
            elif n == 127:
                n = struct.unpack('>Q', self._read(8))[0]
            mask = self._read(4) if b1 & 0x80 else None
            payload = self._read(n)
            if mask:
                payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
            if opcode == 0x8:
                return None
            if opcode == 0x9:
                self._frame(0xA, payload)
                continue
            if opcode in (0x0, 0x1):
                message += payload
                if b0 & 0x80:
                    return message.decode('utf-8')

    def close(self):
        try:
            self._frame(0x8, struct.pack('>H', 1000))
            self.sock.close()
        except OSError:
            pass


def run_websocket(bot, url, token):
    ws = WebSocket(url)
    try:
        while True:
            text = ws.receive()
            if text is None:
                log('the referee closed the connection')
                return
            for line in text.splitlines():
                if not line.strip():
                    continue
                m = json.loads(line)
                answer = bot.handle(m)
                if answer is not None:
                    if answer.get('type') == 'hello' and token:
                        answer['token'] = token
                    ws.send(json.dumps(answer, separators=(',', ':')))
    except SystemExit:
        pass
    finally:
        ws.close()


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='AlphaWOTR as a WotR Bot Protocol client (draft 0.1).')
    ap.add_argument('--connect', help="a referee's WebSocket URL (ws:// or "
                    'wss://); default: the protocol on stdin and stdout')
    ap.add_argument('--token', help='the seat token (or ?token= in the URL)')
    ap.add_argument('--sims', type=int, default=300,
                    help='search simulations per decision (default 300, '
                    "the engine's own default; scaled per kind as "
                    'engine_play does)')
    ap.add_argument('--max-seconds', type=float, default=None,
                    help='cap on the seconds per decision (the referee '
                    'clock is respected as well)')
    ap.add_argument('--threads', type=int, default=1,
                    help='torch threads (default 1)')
    ap.add_argument('--src', default=DEFAULT_SRC,
                    help="the engine's src folder (default: ../src)")
    ap.add_argument('--verbose', action='store_true',
                    help='each decision on stderr')
    ap.add_argument('--log', help='one JSON line per decision to this file')
    args = ap.parse_args(argv)

    # The protocol keeps the original stdout; everything the engine
    # prints goes to stderr.
    proto = os.fdopen(os.dup(1), 'w', buffering=1, encoding='utf-8')
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    import_engine(args.src)
    names = Names()
    searcher = Searcher(args.sims, args.max_seconds, args.threads)
    dlog = open(args.log, 'a', encoding='utf-8') if args.log else None
    bot = AlphaWotrBot(searcher, names, verbose=args.verbose, decision_log=dlog)
    try:
        if args.connect:
            run_websocket(bot, args.connect, args.token)
        else:
            try:
                run_stdio(bot, proto)
            except SystemExit:
                pass
    finally:
        if dlog is not None:
            dlog.close()


if __name__ == '__main__':
    main()
