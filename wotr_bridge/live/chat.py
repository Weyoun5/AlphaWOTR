"""Sided chat protocol vocabulary - THE ONE EXIT for chat (design rule 7).

THE GOVERNING USER RULE (REBUILD_PLAN + LIVE_0818_BUGLIST, verbatim
class): chat is ONLY for what cannot be seen from the physical pieces.
  ALLOWED:   'field battle', 'nc', '0e', attack-region declarations,
             listing PLAYABLE cards, hit counts PHRASED ('1 hit for
             you, 0 hits for me'), combat-card decision magnitudes (a
             decision of 0 must still be stated), excess-dice notes.
  FORBIDDEN AS INTERFACES: discards (F1), casualty allocation (F4),
             retreat region (F2/F5), naming/reading the human's cards
             (F3), 'pick N' menus for anything the board shows.

WHAT THIS MODULE HOLDS: the outbound vocabulary (ALLOWED_KINDS, one
anchored form per kind; FORBIDDEN_KINDS naming the interfaces the USER
ruled out), the single exit ChatOut.say, the inbound grammar
(normalize_body / classify_command / match_choice), and the few text
builders with a live caller: the driver's refusal explanations, the
server's roll request, and declined_leg (USER 2026-08-18: a declined
OPTIONAL card leg is said in chat; the live caller is the AI's
card_option commit).  Human decision points are asked through the
PANEL, never chat, so there is no prompt builder here.

POLICY BY CONSTRUCTION, not post-hoc censoring:
  * ChatOut.say(kind, text) is the single exit; a kind outside the
    vocabulary RAISES ChatPolicyError (a forbidden interface cannot be
    emitted at all), and text is validated against the kind's own
    anchored form so a menu can never ride an allowed kind (deny
    outranks keep).
  * BLINDING BY CONSTRUCTION: no builder takes an action-params object
    or a hand card.  The card names that reach chat are the AI's OWN
    plays (the engine's card_id through nice_card) or a card ALREADY
    face-up on the table; the process never holds human-hand
    identities (design rule 8).
  * Free prose reaches the exit only under a kind whose text a builder
    REGISTERED (_MANDATED); unregistered prose under 'prompt' or
    'refusal' is refused.
"""
import re

__all__ = ['pending_reminder', 'attack_from_ask', 
    'ChatPolicyError', 'ChatOut', 'ALLOWED_KINDS', 'FORBIDDEN_KINDS',
    'normalize_body', 'classify_command', 'match_choice', 'nice_card',
    'declined_leg', 'declined_leg_phrase', 'roll_request',
    'refusal_illegal', 'refusal_closest', 'refusal_cannot_pass',
]


class ChatPolicyError(RuntimeError):
    """An emission the sided chat vocabulary forbids."""


# ---------------------------------------------------------------------
# registry of builder-produced prose (the _chat_ok lift, :14248-14259):
# free-prose kinds ('prompt', 'refusal') accept ONLY texts a builder in
# this module produced - a hand-rolled prompt cannot pass the exit.
_MANDATED = set()


def _register(text):
    _MANDATED.add(str(text).strip().lower())
    return text


# ---------------------------------------------------------------------
# DENY (lifted :14123-14162): by FUNCTION, not wording - enumerations,
# menu instructions/headers, hand-disclosure phrasings, dumped params.
# Deny outranks every keep (RT3-2b) and applies to EVERY kind.
_DENY_RE = [re.compile(p, re.I) for p in (
    # enumeration, any alphabet: TWO or more 'x) ' markers (one ')' is
    # ordinary punctuation - the G3 announce ends '(0, 1, 2)')
    r'(?<![\w(])[a-z0-9]{1,3}\)\s+\S.*(?<![\w(])[a-z0-9]{1,3}\)\s+\S',
    r':\s*-\s+\S.*\s-\s+\S',          # dash bullets after a header
    r'\bpick\s+n\b',                  # the answer-in-chat instruction
    r'\banswer in chat\b',
    r'\bsay which\b', r'\bwhich\b.{0,30}\bdo you\b',
    r'^your .{2,40}\b(decision|allocation|choice|selection|'
    r'option)s?\s*:',                 # the menu header, any wording
    # the AI telling him what he holds (backstop; the guard is that
    # builders never receive hand identities)
    r'\byou hold\b',
    r'\byour hand\b\s*(is|holds|contains|:)',
    r'\byou have\b.{0,30}\bin (your )?hand\b',
    r"\{'",                           # a params dict dumped into chat
    r'\bcard_id\b',
)]

_PLAYABLE_OK_RE = re.compile(r'^playable: ', re.I)

# sentinel: this kind's text must come from a builder (_MANDATED)
_REGISTERED = object()

# ---------------------------------------------------------------------
# THE VOCABULARY (validators lifted from _CHAT_KEEP_RE :13995-14092,
# one anchored form per kind).  A kind absent here cannot be said.
ALLOWED_KINDS = {
    'field_battle':   re.compile(r'^field battle$', re.I),
    'nc':             re.compile(r'^nc$', re.I),
    'zero_eyes':      re.compile(r'^0e$', re.I),
    # WAVE-K R2: the AI-as-FP closes its own Fellowship phase
    'fp_ok':          re.compile(r'^fp ok$', re.I),
    # corpus A8: bare region, or 'X from Y' (region validity is the
    # caller's - it computed the attack; this is the FORM)
    'attack_declare': re.compile(
        r"^[a-z0-9][a-z0-9 '\-]*( from [a-z0-9][a-z0-9 '\-]*)?$", re.I),
    # the AI's 'from?' ask when a bare target is ambiguous (USER
    # 2026-08-22): lists the legal source regions
    'attack_from_ask': re.compile(r"^from\? [a-z0-9][a-z0-9 ,'\-]*$", re.I),
    'playable':       re.compile(
        r"^playable: [a-z0-9][a-z0-9 ,'\-]*$", re.I),
    'hit_count':      re.compile(
        r'^\d+ hits? for you, \d+ hits? for me$', re.I),
    'magnitude':      re.compile(r'^.{2,40}: i choose \d+$', re.I),
    'excess_dice':    re.compile(
        r'^using the first \d+: [\d ]+ \(ignoring [\d ]+\)$', re.I),
    'roll_request':   re.compile(r'^roll \d+ (die|dice)$', re.I),
    # USER-MANDATED (2026-08-15): the Shadow asks out loud before the
    # client's built-in random-companion draw (parser-inert, ends '?')
    'random_ask':     re.compile(r'^random companion draw\?$', re.I),
    'leg_decline':    re.compile(r'^.{2,40}: pass$', re.I),
    'card_announce':  re.compile(
        r'^plays [a-z0-9][^{}\[\]<>]*$', re.I),
    'echo':           re.compile(r'^you chose: [a-z0-9][^{}]*$', re.I),
    'pass':           re.compile(r'^pass$', re.I),
    # the Dread-and-Despair leadership-forfeit statement (USER ff<N>
    # protocol, LIVE_0818 G3 family: a decision of 0 must be stated)
    'ff':             re.compile(r'^ff\s?\d$', re.I),
    'concede':        re.compile(r'^(gg|concede|resign)\b', re.I),
    # USER list (restored 2026-08-15) - corpus-attested one-liners
    # combat protocol words (donor lift :8607-8641 - the same short
    # words classify_command/match_choice already accept INBOUND; the
    # WAVE-H combat leg needed the OUTBOUND kinds: emission's
    # combat_continue/'press', combat_cease/'stop' and shine rode the
    # 'nc' kind whose ^nc$ validator refused every one of them)
    'press':          re.compile(r'^press$', re.I),
    'stop':           re.compile(r'^stop$', re.I),
    'shine':          re.compile(r'^gandalf shines$', re.I),
    'hunt_roll':      re.compile(r'^rolling for hunt$', re.I),
    'discarding':     re.compile(r'^discarding card$', re.I),
    'reveal':         re.compile(r'^(reveal|revealed)$', re.I),
    'wwsat':          re.compile(r'^wwsat$', re.I),
    'balrog':         re.compile(r'^balrog$', re.I),
    'corsairs':       re.compile(r'^corsairs to .+$', re.I),
    'tile_face':      re.compile(r'^(-?\d{1,2}r|eye)$', re.I),
    'mordor_entry':   re.compile(r'^fsp enters mordor$', re.I),
    'card_region':    re.compile(
        r"^(the )?(dead men of dunharrow|spirit of mordor|"
        r"dreadful spells|faramir's rangers|help unlooked for): .+$",
        re.I),
    'hunt_tile':      re.compile(
        r"^hunt tile: [a-z0-9\-][a-z0-9 ,'!:\-]*$", re.I),
    'cc':             re.compile(
        r"^cc: [a-z0-9][a-z0-9 ,'!:\-]*$", re.I),
    # free prose, builder-registered ONLY
    'prompt':         _REGISTERED,
    'refusal':        _REGISTERED,
}

# Named forbidden interfaces: the refusal names the USER rule broken.
FORBIDDEN_KINDS = {
    'discard_menu':   "F1: discards are not handled in chat - wait "
                      "for the human to physically remove a card",
    'casualty_menu':  "F4: casualty allocation is read from the "
                      "board, never asked in chat",
    'retreat_prompt': "F2/F5: retreat is shown by moving units; "
                      "outbound retreat prompts are forbidden",
    'hand_list':      "the AI must never name or read the human's "
                      "cards (identity is blinded by construction)",
    'pick_menu':      "no 'pick N' menus for anything the board shows",
    'status':         "status chatter is PANEL-ONLY (2026-08-16), "
                      "never game chat",
}


# ---------------------------------------------------------------------
# card vocabulary backstop (lifted :14182-14246): a message naming TWO
# or more distinct cards is enumerating a hand, however phrased.  ONE
# card is the ordinary announce grammar; the 'playable:' form the USER
# explicitly allowed is exempt.
_CARD_VOCAB = None


def _deaccent(s):
    import unicodedata as _ud
    try:
        s = _ud.normalize('NFKD', str(s))
    except Exception:
        return str(s)
    return ''.join(c for c in s if not _ud.combining(c)).encode(
        'ascii', 'ignore').decode('ascii')


def _card_vocab():
    global _CARD_VOCAB
    if _CARD_VOCAB is None:
        v = {}
        try:
            import os as _os
            import sys as _sys
            _src = _os.path.abspath(_os.path.join(
                _os.path.dirname(_os.path.abspath(__file__)),
                '..', '..', 'src'))
            if _src not in _sys.path:
                _sys.path.insert(0, _src)
            import cards as _cd
            for cid in _cd.get_cards():
                raw = str(cid or '')
                for pre in ('FREE_PEOPLES_', 'SHADOW_'):
                    if raw.startswith(pre):
                        raw = raw[len(pre):]
                        break
                forms = [raw]
                for deck in ('Character_', 'Strategy_'):
                    if raw.startswith(deck):
                        forms.append(raw[len(deck):])
                for n in forms:
                    n = n.replace('_', ' ').strip().lower()
                    n = _deaccent(n)
                    if len(n) >= 6:
                        v[n] = cid
        except Exception:
            pass
        _CARD_VOCAB = v
    return _CARD_VOCAB


def _cards_named(msg):
    """How many DISTINCT CARDS (not names - two spellings per card)
    this text names."""
    t = _deaccent(str(msg or '').lower())
    return len({cid for n, cid in _card_vocab().items()
                if n and n in t})


# ---------------------------------------------------------------------
# THE ONE EXIT
class ChatOut(object):
    """Every chat emission passes through say(kind, text); anything
    outside the vocabulary raises ChatPolicyError.  `sink(kind, text)`
    is called for accepted texts (the emission module renders them);
    accepted texts also accumulate on .sent."""

    def __init__(self, sink=None):
        self.sink = sink
        self.sent = []

    def say(self, kind, text):
        self.validate(kind, text)
        self.sent.append((kind, text))
        if self.sink is not None:
            self.sink(kind, text)
        return text

    @staticmethod
    def validate(kind, text):
        """Raise ChatPolicyError unless (kind, text) is sanctioned."""
        if kind in FORBIDDEN_KINDS:
            raise ChatPolicyError('forbidden chat interface %r: %s'
                                  % (kind, FORBIDDEN_KINDS[kind]))
        rule = ALLOWED_KINDS.get(kind)
        if rule is None:
            raise ChatPolicyError(
                'chat kind %r is not in the sided vocabulary '
                '(chat is only for what the pieces cannot show)'
                % (kind,))
        if not isinstance(text, str) or not text.strip() \
                or '\n' in text:
            raise ChatPolicyError('chat text must be one non-empty '
                                  'line, got %r' % (text,))
        # DENY FIRST - it outranks every keep, registration included
        for rx in _DENY_RE:
            if rx.search(text):
                raise ChatPolicyError(
                    'denied by content rule %r: %r'
                    % (rx.pattern[:40], text[:90]))
        # two or more distinct cards = enumerating a hand (exempt: the
        # one 'playable:' form the USER allowed)
        if not _PLAYABLE_OK_RE.match(text) and _cards_named(text) >= 2:
            raise ChatPolicyError(
                'names %d cards - a card list is a hand disclosure: %r'
                % (_cards_named(text), text[:90]))
        if rule is _REGISTERED:
            if text.strip().lower() not in _MANDATED:
                raise ChatPolicyError(
                    '%r text was not produced by a sanctioned builder:'
                    ' %r' % (kind, text[:90]))
        elif not rule.search(text):
            raise ChatPolicyError(
                'text does not fit the %r form (%r): %r'
                % (kind, rule.pattern[:50], text[:90]))


# ---------------------------------------------------------------------
# inbound: body normalization (lifted :2148-2167 / :1553-1557)
_HASH_RE = re.compile(r'[0-9a-fA-F]{6}$')


def normalize_body(body):
    """Chat body -> matching form: strip the trailing checksum token
    ('a1b2c3') or offline 'null', lowercase, collapse edges."""
    s = str(body or '').strip()
    toks = s.split()
    if len(toks) > 1 and (_HASH_RE.fullmatch(toks[-1])
                          or toks[-1] == 'null'):
        s = ' '.join(toks[:-1])
    return s.strip().lower()


def classify_command(body):
    """Protocol-level inbound recognizer (grammar lifted :1538-1650 +
    the buglist forms).  Returns (kind, arg) or None.  Menu answers go
    through match_choice(); this classifies the commands that exist
    OUTSIDE any menu ('fp ok', 'resync', ...)."""
    txt = normalize_body(body)
    toks = txt.split()
    if not toks:
        return None
    if ' '.join(toks[:2]) == 'fp ok':          # B7 handshake, :1558
        return ('fp_ok', None)
    if txt == 'resync':
        return ('resync', None)
    if txt.rstrip('.!') == 'pass':             # :1584
        return ('pass', None)
    m = re.fullmatch(r'(\d)e', txt)            # D3: '0e' / 'Ne'
    if m:
        return ('eyes', int(m.group(1)))
    if txt == 'nc':
        return ('nc', None)
    if txt == 'field battle':
        return ('field_battle', None)
    if toks[0] == 'cc' and len(toks) > 1:
        return ('cc', ' '.join(toks[1:]))
    if toks[0] == 'retreat':
        return ('retreat', ' '.join(toks[1:]) or None)
    if toks[0] in ('stand', 'stay') or txt == 'no retreat':
        return ('stand', None)
    if toks[0] == 'pick' and len(toks) > 1 and toks[1].isdigit():
        return ('pick', int(toks[1]))
    if txt == 'random':
        return ('random', None)
    if toks[0] in ('press', 'continue', 'fight'):
        return ('press', None)
    if toks[0] in ('stop', 'cease') or txt in ('no press', 'nopress'):
        return ('stop', None)
    m = re.fullmatch(r'ff\s*(\d)', txt)
    if m:
        return ('ff', int(m.group(1)))
    if len(toks) == 1 and toks[0].isdigit() and len(toks[0]) <= 2:
        return ('magnitude', int(toks[0]))
    if toks[0] in ('gg', 'concede', 'resign'):
        return ('concede', None)
    # attack declaration (USER 2026-08-22): '<target> from <source>'
    # (the AI's own form), 'from <source>' answering the AI's 'from?'
    # ask, or a bare '<target>' - the driver checks it is a region
    # and resolves the source when only one army can attack it
    m = re.fullmatch(r"(?:attack )?([a-z0-9][a-z0-9 '\-]*?) from "
                     r"([a-z0-9][a-z0-9 '\-]*)", txt)
    if m:
        return ('attack', (m.group(1).strip(), m.group(2).strip()))
    m = re.fullmatch(r"from ([a-z0-9][a-z0-9 '\-]*)", txt)
    if m:
        return ('attack_from', m.group(1).strip())
    m = re.fullmatch(r"(?:attack )?([a-z0-9][a-z0-9 '\-]*)", txt)
    if m:
        return ('region', m.group(1).strip())
    return None


def match_choice(body, menu, nice=None):
    """Map one human chat answer to a menu Action, or None (lifted
    :2169-2271).  Menu items carry .action_type and .params.

    Vocabulary = the corpus chat grammar; 'pick N' kept as the LEGACY
    fallback for decision points the corpus has no chat gesture for.
    DEVIATIONS from the old matcher (each modelled per the task/buglist):
      * 'Ne' eye statements -> eye_allocate count=N (D3 grammar,
        two-sided);
      * a BARE NUMBER binds to combat_effect_choice count=N when such
        a menu is open (G3: magnitude statements - the old index path
        could not even express a stated 0), and stays the legacy
        1-based menu index otherwise;
      * 'field battle' -> combat_no_retreat (F2: chat's only sanctioned
        word in the retreat-into-siege window);
      * 'siege' / 'retreat into siege' -> retreat_into_siege (LEGACY
        inbound tolerance - the F2 specimen shows humans typing them);
      * combat_retreat region matching reads params 'to_region' OR
        'to' (both spellings exist: monolith :6432 vs engine_play:270).
    """
    if nice is None:
        nice = nice_card
    txt = normalize_body(body)
    toks = txt.split()
    if not toks:
        return None

    def _first(tt):
        return next((a for a in menu if a.action_type == tt), None)

    # D3: '0e'/'Ne' -> eye allocation count
    m = re.fullmatch(r'(\d)e', txt)
    if m:
        n = int(m.group(1))
        return next((a for a in menu
                     if a.action_type == 'eye_allocate'
                     and int((a.params or {}).get('count', -1)) == n),
                    None)
    # 'pick N' / bare number
    mnum = None
    picked_explicitly = False
    if toks[0] == 'pick' and len(toks) > 1 and toks[1].isdigit():
        mnum = int(toks[1])
        picked_explicitly = True
    elif len(toks) == 1 and toks[0].isdigit():
        mnum = int(toks[0])
    # G3 magnitude statement: a bare number is the COUNT when a
    # combat_effect_choice menu is open ('state it even if it is 0')
    if mnum is not None and not picked_explicitly:
        mags = [a for a in menu
                if a.action_type == 'combat_effect_choice']
        if mags:
            return next(
                (a for a in mags
                 if int((a.params or {}).get('count', -1) or 0)
                 == mnum), None)
    if mnum is not None and 1 <= mnum <= len(menu):
        return menu[mnum - 1]
    # HUNT-DECISION GRAMMAR (USER 2026-08-17: "literally just
    # 'random'") - physical gestures carry the rest
    if toks[0] == 'random':
        return _first('hunt_sacrifice_random')
    if txt in ('corruption', 'take corruption', 'take the corruption'):
        return _first('hunt_take_corruption')
    if txt in ('guide', 'kill guide', 'kill the guide',
               'sacrifice guide'):
        return _first('hunt_sacrifice_guide')
    if txt == 'reveal' and _first('hunt_gollum_reveal'):
        return _first('hunt_gollum_reveal')
    if txt in ('no reveal', 'no') and _first('hunt_gollum_no_reveal'):
        return _first('hunt_gollum_no_reveal')
    # corpus A4: 'nc' matched on the LAST token too ('ok nc')
    if toks[0] == 'nc' or toks[-1] == 'nc':
        return _first('combat_card_none')
    # corpus A5: 'ff3' / 'ff 3'
    m = re.fullmatch(r'ff\s*(\d)', txt)
    if m:
        n = int(m.group(1))
        return next(
            (a for a in menu
             if a.action_type == 'combat_effect_choice'
             and int((a.params or {}).get('count', -1) or 0) == n),
            None)
    if toks[0] == 'cc' and len(toks) > 1:
        want = ' '.join(toks[1:])
        return next(
            (a for a in menu
             if a.action_type == 'combat_card_play'
             and want in nice((a.params or {}).get(
                 'card_id')).lower()), None)
    if toks[0] == 'pass':
        return _first('pass')
    if toks[0] in ('cease', 'stop') or txt in ('no press', 'nopress'):
        return _first('combat_cease') or _first('siege_no_press')
    if toks[0] in ('continue', 'press', 'fight'):
        return _first('combat_continue') or _first('siege_press')
    if toks[0] == 'assault':
        return _first('siege_assault')
    if txt in ('shine', 'gandalf shines', 'shines'):
        return _first('shine')
    if txt in ('no shine', 'noshine', 'dont shine', "don't shine"):
        return _first('shine_none')
    # F2: 'field battle' = stand and fight in the field
    if txt == 'field battle':
        return _first('combat_no_retreat')
    if txt in ('siege', 'retreat into siege', 'into siege'):
        return _first('retreat_into_siege')
    if toks[0] in ('stay', 'stand', 'hold') or txt in (
            'no retreat', 'no advance', 'stay put'):
        # the same words answer the VICTOR's advance decision (no piece
        # can show "I stay" - chat is the only gesture for it); the
        # menus never overlap, so the first present wins
        return (_first('combat_no_retreat') or _first('siege_no_press')
                or _first('victor_stay'))
    if toks[0] == 'retreat':
        rets = [a for a in menu if a.action_type == 'combat_retreat']
        if len(toks) > 1:
            want = ' '.join(toks[1:])
            for a in rets:
                dest = str((a.params or {}).get(
                    'to_region', (a.params or {}).get('to', '')))
                if want in dest.lower():
                    return a
            return None
        if len(rets) == 1:
            return rets[0]
    # bare combat-card NAME, matched LAST so grammar words never
    # collide with card names
    cands = [a for a in menu
             if a.action_type == 'combat_card_play'
             and (txt in nice((a.params or {}).get('card_id')).lower()
                  or nice((a.params or {}).get(
                      'card_id')).lower() in txt)]
    if len(cands) == 1:
        return cands[0]
    return None


# ---------------------------------------------------------------------
# builders (pure functions; free-prose ones self-register)
def nice_card(cid):
    """The card's TITLE, as a human says it (lifted :1856-1889 with the
    FP two-token-side fix: deck word stripped on BOTH sides)."""
    s = str(cid or '')
    for _side in ('FREE_PEOPLES_', 'SHADOW_'):
        if s.startswith(_side):
            s = s[len(_side):]
            break
    for _deck in ('Character_', 'Strategy_'):
        if s.startswith(_deck):
            s = s[len(_deck):]
            break
    return s.replace('_', ' ').strip()


_LEG_TERMINATORS = ('_tsl_done', '_tsim_done', '_naz_done',
                    '_ents_chain_done', '_ffm_done', '_comp_done')


def declined_leg_phrase(card_title):
    """The corpus decline word inside the '<card>: <detail>' frame
    (lifted :5914-5919)."""
    return '%s: pass' % str(card_title).strip().lower()


def declined_leg(chosen, acts, nice=None):
    """USER 2026-08-18: a declined OPTIONAL leg must be indicated in
    chat - announced only when a real alternative existed (lifted
    :5879-5913).  Returns the phrase or None."""
    if nice is None:
        nice = nice_card
    if chosen is None or getattr(chosen, 'action_type', None) \
            != 'card_option':
        return None
    ch = (getattr(chosen, 'params', None) or {}).get('choices') or {}
    if not isinstance(ch, dict):
        return None
    if not any(ch.get(k) for k in _LEG_TERMINATORS):
        return None
    alt = 0
    for a in (acts or []):
        if a is chosen or a.action_type != 'card_option':
            continue
        c2 = (getattr(a, 'params', None) or {}).get('choices') or {}
        if isinstance(c2, dict) and not any(
                c2.get(k) for k in _LEG_TERMINATORS):
            alt += 1
    if not alt:
        return None
    nm = (nice((chosen.params or {}).get('card_id'))
          or (chosen.params or {}).get('card_name') or 'card').lower()
    return declined_leg_phrase(nm)


def roll_request(n):
    """G1 ask: the human rolls their own combat dice."""
    n = int(n)
    return 'roll %d %s' % (n, 'die' if n == 1 else 'dice')


# ------------------------------------------------- refusal explanations
# (texts lifted from _refusal_explain :5754-5862; each names the ACTUAL
# failed constraint - USER P1 2026-08-17: refusals must EXPLAIN)
_SCRUB_RE = re.compile(r'[{}\[\]<>]')


def _prose(s):
    return _SCRUB_RE.sub('', str(s or '')).strip()


def _fold_region(text):
    """sanitize + possessive fold: "Helm's Deep" -> 'helm s deep' ->
    'helms deep', so 'helms deep' / "helm's deep" / 'helm s deep' all
    meet the same key."""
    from board import sanitize
    t = sanitize(str(text)).strip().lower()
    return re.sub(r'\b(\w+) s\b', r'\1s', t)


def resolve_region(text, region_keys):
    """THE chat region resolver (USER 2026-08-23: 'region list +
    accepted shortforms').  Returns (key, candidates):
      key        the ONE region the text names, else None
      candidates the regions it could mean (2+ = ambiguous, the
                 refusal lists them; 0 = not a region)
    Accepted forms, in order: the exact name (case / accents /
    punctuation ignored: 'barad dur'); the possessive without its
    apostrophe ('helms deep', 'goblins gate'); any UNIQUE prefix
    ('osg' -> osgiliath, 'fords of i' -> fords of isen); any unique
    WORD-prefix sequence ('tirith' -> minas tirith, 'n ithilien' ->
    north ithilien).  A short form that fits several regions is
    ambiguous ('fords' -> fords of bruinen / fords of isen), never a
    guess."""
    from board import sanitize
    keys = sorted(set(sanitize(str(k)).strip().lower()
                      for k in region_keys))
    if not keys:
        return None, []
    t = sanitize(str(text)).strip().lower()
    if not t:
        return None, []
    if t in keys:
        return t, [t]
    ft = _fold_region(t)
    folded = {k: _fold_region(k) for k in keys}
    exact = [k for k in keys if folded[k] == ft]
    if len(exact) == 1:
        return exact[0], exact
    toks = ft.split()
    pre = [k for k in keys if folded[k].startswith(ft)]
    if len(pre) == 1:
        return pre[0], pre

    def _word_match(k):
        words = folded[k].split()
        i = 0
        for w in words:
            if i < len(toks) and w.startswith(toks[i]):
                i += 1
        return i == len(toks)
    word = [k for k in keys if _word_match(k)]
    cands = sorted(set(exact or pre or word))
    if len(cands) == 1:
        return cands[0], cands
    return None, cands


def region_ambiguous(text, candidates):
    return _register('%s could be: %s - type more of the name'
                     % (str(text).strip().lower(),
                        ' / '.join(str(c) for c in candidates)))


def attack_from_ask(sources):
    """'from? a, b, c' - the bare target can be attacked from several
    regions; the human answers 'from <region>' (or '<target> from
    <region>')."""
    return 'from? ' + ', '.join(str(x).strip().lower()
                                 for x in sources)


def pending_reminder(what):
    """One-time reminder that a HUMAN decision is still owed (R5: the
    machine holds, it never decides for the human)."""
    return _register('your %s decision is still open - answer it '
                     'first (board or chat)' % _prose(what))


def undo_rebased():
    """The human retracted a window the AI had already answered: the
    engine followed the client (window popped, the AI's reply re-based
    onto the restored state) - said once, so the human knows the reply
    stands and the action is theirs again."""
    return _register("your action was taken back; the AI's reply "
                     'stands - your turn')


def refusal_illegal(detail=None):
    t = 'that move is not legal'
    if detail:
        t += ' - %s' % _prose(detail)
    return _register(t)


def refusal_closest(description, differing_keys):
    return _register('closest legal move: %s - your gesture differs '
                     'on %s' % (_prose(description),
                                ', '.join(_prose(k) for k in
                                          list(differing_keys)[:3])))


def refusal_cannot_pass(detail=None):
    t = 'you cannot pass'
    if detail:
        t += ' - %s' % _prose(detail)
    return _register(t)


def refusal_eyes_by_board(n):
    """USER doctrine: chat only for what pieces cannot show.  A Shadow
    eye allocation IS a piece move (SA Dice -> The Hunt For The Ring,
    or the client's allocate-eye button), so an 'Ne' chat is refused
    with the drag it stands for; '0e' stays chat (no piece shows
    zero)."""
    n = int(n)
    return _register('eyes go on the board, not in chat - drag %d '
                     'Shadow action %s from SA Dice to The Hunt For '
                     'The Ring (or press allocate eye %d time%s), then '
                     'roll' % (n, 'die' if n == 1 else 'dice', n,
                               '' if n == 1 else 's'))
