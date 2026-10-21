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

__all__ = [
    'pending_reminder', 'attack_from_ask',
    'ChatPolicyError', 'ChatOut', 'ALLOWED_KINDS', 'FORBIDDEN_KINDS',
    'normalize_body', 'classify_command', 'match_choice', 'nice_card',
    'declined_leg', 'declined_leg_phrase', 'roll_request',
    'refusal_illegal', 'refusal_closest', 'refusal_cannot_pass',
]


class ChatPolicyError(RuntimeError):
    """An emission the sided chat vocabulary forbids."""


# ---------------------------------------------------------------------
# registry of builder-produced prose: free-prose kinds ('prompt',
# 'refusal') accept ONLY texts a builder in this module produced - a
# hand-rolled prompt cannot pass the exit.
_MANDATED = set()


def _register(text):
    _MANDATED.add(str(text).strip().lower())
    return text


# ---------------------------------------------------------------------
# DENY by FUNCTION, not wording - enumerations, menu instructions/
# headers, hand-disclosure phrasings, dumped params.  Deny outranks
# every keep and applies to EVERY kind.
_DENY_RE = [re.compile(p, re.I) for p in (
    # enumeration, any alphabet: TWO or more 'x) ' markers (one ')' is
    # ordinary punctuation - the G3 announce ends '(0, 1, 2)')
    r'(?<![\w(])[a-z0-9]{1,3}\)\s+\S.*(?<![\w(])[a-z0-9]{1,3}\)\s+\S',
    r':\s*-\s+\S.*\s-\s+\S',          # dash bullets after a header
    r'\bpick\s+n\b',                  # the answer-in-chat instruction
    r'\bpick\s+(?:a\s+|the\s+)?(?:<\s*)?(?:number\b|n\b|#?\d+)',
    r'\b(?:choose|select|enter|type)\s+(?:a\s+|the\s+)?(?:option\s+|choice\s+)?number\b',
    r'\bnumbered\s+(?:choices|options|menu)\b',
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
# THE VOCABULARY (one anchored form per kind).  A kind absent here
# cannot be said.
ALLOWED_KINDS = {
    'field_battle':   re.compile(r'^field battle$', re.I),
    'nc':             re.compile(r'^nc$', re.I),
    'zero_eyes':      re.compile(r'^0e$', re.I),
    # R2: the AI-as-FP closes its own Fellowship phase
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
        r'^(?:using the first \d+: [\d ]+ \(ignoring [\d ]+\)'
        r'|roll already complete; ignoring extra dice: [\d ]+)$', re.I),
    # BL-122 (wave 26): the optional ' #N' tail is the ask NONCE -
    # the seat answers each nonce exactly once (same-text asks recur
    # legitimately round after round; text is not identity)
    'roll_request':   re.compile(r'^roll \d+ (die|dice)( #\d+)?$',
                                 re.I),
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
    # corpus-attested one-liners (USER list 2026-08-15): the combat
    # protocol words, as OUTBOUND kinds (the same short words
    # classify_command/match_choice accept INBOUND; emission's press /
    # stop / shine ride their own kinds)
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
# card vocabulary backstop: a message naming TWO or more distinct cards
# is enumerating a hand, however phrased.  ONE card is the ordinary
# announce grammar; the 'playable:' form the USER explicitly allowed is
# exempt.
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


_REGION_VOCAB = None


def _region_vocab():
    """Normalized board-region keys (lazy; empty set on any import
    failure - the region-value lane then simply never fires)."""
    global _REGION_VOCAB
    if _REGION_VOCAB is None:
        v = set()
        try:
            import os as _os
            import sys as _sys
            _src = _os.path.abspath(_os.path.join(
                _os.path.dirname(_os.path.abspath(__file__)),
                '..', '..', 'src'))
            if _src not in _sys.path:
                _sys.path.insert(0, _src)
            import board as _bd
            for k in _bd.regions:
                v.add(re.sub(r'[^a-z0-9]+', ' ',
                             _deaccent(str(k)).lower()).strip())
        except Exception:
            pass
        _REGION_VOCAB = v
    return _REGION_VOCAB


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
# inbound: body normalization
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
    if txt.rstrip('.!') == 'resync':
        return ('resync', None)
    if txt.rstrip('.!') in ('reset', 'reset resync', 'reset + resync', 'reset+resync', 'retry'):
        return ('reset', None)
    if txt.rstrip('.!') in ('rewind', 'rewind turn', 'rewind to start of turn'):
        return ('rewind', None)
    if txt.rstrip('.!') == 'resume':
        return ('resume', None)
    if txt.rstrip('.!') in ('skip', 'skip decision'):
        return ('skip', None)
    if txt == 'adopt':
        # SQF-11: the standing census acceptance (the panel's adopt
        # route, reachable from the CLI seat)
        return ('adopt', None)
    if txt == 'yes':
        return ('yes', None)        # answers an outstanding self-repair offer
    if txt == 'no':
        return ('no', None)
    if txt in ('help', '?', 'commands'):       # USER 2026-08-24
        return ('help', None)
    if txt.rstrip('.!') == 'pass':             # :1584
        return ('pass', None)
    m = re.fullmatch(r'(\d+)\s*e', txt.rstrip('.!'))
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


def matches_battle_target(body, combat, region_keys=None):
    """Whether this declaration names the already-open battle exactly."""
    txt = normalize_body(body)
    declaration = classify_command(
        'attack ' + txt[len('assault '):] if txt.startswith('assault ') else txt)
    if declaration is None or declaration[0] not in ('region', 'attack'):
        return False
    target, source = (declaration[1] if declaration[0] == 'attack'
                      else (declaration[1], None))
    keys = region_keys if region_keys is not None else _region_vocab()
    named_target, _ = resolve_region(target, keys)
    battle_target, _ = resolve_region(str((combat or {}).get('defender_region', '')), keys)
    if named_target is None or named_target != battle_target:
        return False
    if source is None:
        return True
    named_source, _ = resolve_region(source, keys)
    battle_source, _ = resolve_region(str((combat or {}).get('attacker_region', '')), keys)
    return named_source is not None and named_source == battle_source


def match_choice(body, menu, nice=None, *, combat=None, region_keys=None):
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
      * 'field', 'fight', or a phrase containing either word ->
        combat_no_retreat while the defender's choice is open;
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

    # Outgoing named combat choices must also identify the same legal action
    # when typed by a human. Payment rows remain physical board gestures.
    from combat_decision_text import parse_choice
    stated = parse_choice(body)
    if stated is not None:
        from cards import get_card
        matches = []
        for action in menu:
            params = action.params or {}
            if action.action_type != 'combat_effect_choice' or all(
                    key in params for key in ('nation', 'kind', 'operation')):
                continue
            card = get_card(params.get('card_id'))
            if card is None or card.combat_name.strip().casefold().replace(' ', '_') != stated['effect']:
                continue
            if 'count' in stated and params.get('count') != stated['count']:
                continue
            if 'target' in stated:
                if normalize_body(str(params.get('target', ''))) != normalize_body(stated['target']):
                    continue
            elif params.get('target') is not None:
                continue
            matches.append(action)
        return matches[0] if len(matches) == 1 else None

    def _first(tt):
        return next((a for a in menu if a.action_type == tt), None)

    # Field defense has no piece gesture. Accept natural phrasing only
    # while this legal defender action is available; otherwise 'fight'
    # retains its existing attacker-continuation meaning below.
    field_defense = _first('combat_no_retreat')
    if field_defense is not None and re.search(r'\b(?:field|fight)\b', txt):
        if re.search(r"\b(?:no|not|never|without|cannot|(?:do|does|did|wo|ca|is|are|should|would|could)n['\u2019]?t)\b", txt):
            return None
        return field_defense

    if txt in ('keep', 'keep flocks', 'keep flocks of crebain',
               'discard flocks', 'discard flocks of crebain'):
        discard = txt.startswith('discard ')
        flocks = [a for a in menu if a.action_type == 'card_option'
                  and (a.params or {}).get('card_id') == 'SHADOW_Character_Flocks_of_Crebain'
                  and ((a.params or {}).get('choices') or {}).get('crebain_discard') is discard]
        if flocks:
            return flocks[0] if len(flocks) == 1 else None

    if txt in ('draw', 'decline', 'no draw'):
        take = txt == 'draw'
        gandalf = next((a for a in menu
                        if a.action_type == 'card_option'
                        and (a.params or {}).get('card_id') == '__gandalf_draw__'
                        and ((a.params or {}).get('choices') or {}).get(
                            'gandalf_draw') is take), None)
        if gandalf is not None:
            return gandalf

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
    named_onslaught = re.fullmatch(r'onslaught\s*(\d+)', txt)
    if named_onslaught:
        from cards import get_card
        matches = []
        for action in menu:
            if (action.action_type != 'combat_effect_choice'
                    or (action.params or {}).get('count') != int(named_onslaught[1])):
                continue
            card = get_card((action.params or {}).get('card_id'))
            if card is not None and str(card.combat_name).lower() == 'onslaught':
                matches.append(action)
        return matches[0] if len(matches) == 1 else None
    if mnum is not None and not picked_explicitly:
        mags = [a for a in menu
                if a.action_type == 'combat_effect_choice']
        if mags:
            matches = [a for a in mags
                       if type((a.params or {}).get('count')) is int
                       and a.params['count'] == mnum]
            if len(matches) > 1:
                # A magnitude cannot identify which figure pays. The published
                # explicit pick number still identifies the full legal action.
                return None
            _mhit = matches[0] if matches else None
            # FLA-15 (par5 g1#10): the magnitude grammar keeps
            # precedence when a COUNT matches; otherwise the number
            # is the advertised say_pick INDEX - returning None here
            # refused the published row's own advertised phrase
            if _mhit is not None:
                return _mhit
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
        return _first('hunt_sacrifice_guide') or _first('hunt_guide_separate')
    if txt == 'reveal' and _first('hunt_gollum_reveal'):
        return _first('hunt_gollum_reveal')
    if txt in ('no reveal', 'no') and _first('hunt_gollum_no_reveal'):
        return _first('hunt_gollum_no_reveal')
    # NAZGUL-FAMILY CHAINS (USER 2026-08-24: "the chat it should be
    # waiting for is 'done' / 'rolling for hunt' / 'hunt' /
    # 'discarding card'").  These sit ABOVE the region-value lane so
    # the words never fall through to region resolution.
    def _copt_choice(pred):
        hits = [a for a in menu
                if a.action_type == 'card_option'
                and isinstance((a.params or {}).get('choices'), dict)
                and pred((a.params or {}).get('choices'))]
        return hits[0] if len(hits) == 1 else None

    if re.fullmatch(r"(all |i'?m |i am )?(done|finished)"
                    r"( moving( nazgul)?)?", txt):
        # the chain terminator (_naz_done and the whole family)
        act = _copt_choice(lambda ch: any(ch.get(k)
                                          for k in _LEG_TERMINATORS))
        if act is not None:
            return act
        act = _first('followup_done')
        if act is not None:
            return act
    if re.fullmatch(r'(rolling for (the )?hunt|roll (for )?(the )?hunt'
                    r'|hunt( roll)?)', txt):
        act = _copt_choice(lambda ch: ch.get('action') == 'hunt')
        if act is not None:
            return act
    m9 = re.fullmatch(r'discard(ing)?( (the )?card)?(:)?\s*'
                      r'(?P<name>.*)', txt)
    if m9 is not None:
        _dts = [a for a in menu
                if a.action_type == 'card_option'
                and isinstance((a.params or {}).get('choices'), dict)
                and (a.params or {}).get('choices').get('action')
                == 'discard_table_card']
        _nm9 = _deaccent(m9.group('name') or '').strip()
        if _nm9:
            _named = [a for a in _dts
                      if _nm9 in _deaccent(nice(
                          (a.params or {}).get('choices')
                          .get('target_card'))).lower()]
            if len(_named) == 1:
                return _named[0]
        elif len(_dts) == 1:
            return _dts[0]
        # FLD-10c (par5b_g5#9, exact g5 repeat): the AI's OWN help
        # advertises 'discarding card <name>' - the hand-limit
        # discard_card rows must answer to it too (identity by the
        # card's rendered name, unique match only)
        if _nm9:
            _hnamed = [a for a in menu
                       if a.action_type == 'discard_card'
                       and _nm9 in _deaccent(nice(
                           (a.params or {}).get('card_id'))).lower()]
            if len(_hnamed) == 1:
                return _hnamed[0]
        if _dts:
            return None      # ambiguous: never guess (driver refuses)
    if txt in ('none', 'neither', 'no card', 'decline'):
        act = _copt_choice(lambda ch: ch.get('action') == 'none')
        if act is not None:
            return act
    # corpus A4: 'nc' matched on the LAST token too ('ok nc')
    if toks[0] == 'nc' or toks[-1] == 'nc':
        return _first('combat_card_none')
    # Keep corpus shorthand, while making new logs readable without decoding it.
    from combat_forfeit_text import dread_forfeit_count
    n = dread_forfeit_count(txt)
    if n is not None:
        from cards import get_card
        def is_dread(action):
            card = get_card((action.params or {}).get('card_id', ''))
            return card is not None and card.combat_name.strip().lower() == 'dread and despair'
        matches = [a for a in menu
                   if a.action_type == 'combat_effect_choice'
                   and is_dread(a)
                   and type((a.params or {}).get('count')) is int
                   and a.params['count'] == n]
        return matches[0] if len(matches) == 1 else None
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
    if txt == 'assault':
        return _first('siege_assault')
    # A region names the assault already opened against that stronghold.
    # Its action has no region params, so consult the actual pending battle;
    # never turn a different named target into an unqualified assault.
    assault = _first('siege_assault')
    if (assault is not None and (combat or {}).get('phase') == 'attacker_siege_decision'
            and matches_battle_target(body, combat, region_keys)):
        return assault
    if txt in ('shine', 'gandalf shines', 'shines'):
        return _first('shine')
    if txt in ('no shine', 'noshine', 'dont shine', "don't shine"):
        return _first('shine_none')
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
    # Figure choices use only identities present in the actual legal menu.
    # Heroic Death may offer several cancellation magnitudes for one figure;
    # its name alone must not silently choose a magnitude.
    figure_text = re.sub(r'^(?:cancel|sacrifice|kill|choose|guide)\s+', '', txt)
    figure_count = None
    count_match = re.fullmatch(r'(.+?)\s+(?:for\s+)?(\d+)(?:\s+hits?)?', figure_text)
    if count_match:
        figure_text, figure_count = count_match[1], int(count_match[2])
    def _figure_fold(value):
        return re.sub(r'[^a-z0-9]+', ' ', _deaccent(str(value)).lower()).strip()
    wanted = _figure_fold(figure_text)
    figures = []
    for action in menu:
        if action.action_type not in ('combat_effect_choice', 'choose_guide'):
            continue
        params = action.params or {}
        name = params.get('target', params.get('companion', params.get('name')))
        if not isinstance(name, str) or not name:
            continue
        if figure_count is not None and params.get('count') != figure_count:
            continue
        figures.append((action, _figure_fold(name)))
    matches = [action for action, name in figures if name == wanted]
    if not matches and len(wanted) >= 3:
        matches = [action for action, name in figures if name.startswith(wanted + ' ')]
    if len(matches) == 1:
        return matches[0]
    # a REGION NAME answering the open pending (tester 2026-08-24:
    # multi-siege Fighting Uruk-hai targets, and the Grond / Black
    # Captain Commands / Ringwraiths-are-Abroad follow-up attack
    # grants - 'pick N' indices are invisible to the human).  The ONE
    # resolver runs over each action's own region-valued params
    # (card_option choices + to/target fields); a UNIQUE hit answers.
    def _region_vals(a):
        # ONLY values that really are board regions (an 'action'
        # value like 'hunt' must never be region-matched - it made
        # bare 'hunt' an accidental, fragile hit before the explicit
        # lanes above landed)
        p = a.params or {}
        vals = [str(v) for v in
                (p.get('choices') or {}).values()
                if isinstance(v, str)]
        for k in ('to', 'to_region', 'region', 'attack_target',
                  'target_region'):
            v = p.get(k)
            if isinstance(v, str):
                vals.append(v)
        _rv = _region_vocab()
        return [v for v in vals
                if re.sub(r'[^a-z0-9]+', ' ',
                          _deaccent(v).lower()).strip() in _rv]
    _hits = []
    for a in menu:
        _vals = _region_vals(a)
        if _vals:
            _k9, _c9 = resolve_region(txt, _vals)
            if _k9 is not None:
                _hits.append(a)
    if len(_hits) == 1:
        return _hits[0]
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
    if (chosen.params or {}).get('card_id') == '__gandalf_draw__' \
            and ch.get('gandalf_draw') is False:
        if any(a.action_type == 'card_option'
               and (a.params or {}).get('card_id') == '__gandalf_draw__'
               and ((a.params or {}).get('choices') or {}).get(
                   'gandalf_draw') is True for a in (acts or ())):
            return declined_leg_phrase('Gandalf guide draw')
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
# each names the ACTUAL failed constraint (USER P1 2026-08-17: refusals
# must EXPLAIN)
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


REGION_SHORTFORMS = {
    'dg': 'dol guldur', 'hd': "helm's deep", 'mt': 'minas tirith',
    'da': 'dol amroth', 'mg': 'mount gundabad', 'mm': 'minas morgul', 'lor': 'lorien',
    'wlr': 'woodland realm', 'ofr': 'old forest road',
    'fords': 'fords of isen', 'pel': 'pelargir', 'shire': 'the shire',
    'riv': 'rivendell', 'ereb': 'erebor', 'gh': 'grey havens',
    'gray havens': 'grey havens',
}


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
    ambiguous unless it is an explicitly defined short form. Reserved
    short forms never fall back to a different region when their target
    is absent from the supplied legal candidates."""
    from board import sanitize
    keys = sorted(set(sanitize(str(k)).strip().lower()
                      for k in region_keys))
    if not keys:
        return None, []
    t = sanitize(str(text)).strip().lower()
    if not t:
        return None, []
    if t in REGION_SHORTFORMS:
        wanted = _fold_region(REGION_SHORTFORMS[t])
        aliases = [k for k in keys if _fold_region(k) == wanted]
        return (aliases[0], aliases) if len(aliases) == 1 else (None, aliases)
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
    from .decision_prompts import decision_label, pending_instructions
    return _register('your %s is still open: %s'
                     % (decision_label(what), pending_instructions({'kind':what})))


def roll_context(call, human_side):
    from .roll_prompts import description
    return _register(description(call, human_side))


def waiting_on(what, words):
    """USER 2026-08-24: when the AI waits on a CHAT decision it says
    so and names the words it understands.  `words` is a short
    slash-separated vocabulary string (no card names, no menus)."""
    from .decision_prompts import decision_label
    return _register('waiting on your %s: %s'
                     % (decision_label(what), str(words)))


def nazgul_chain_open_note():
    return _register('move nazgul on the board, then say done')


def companion_chain_open_note():
    """FLD-10a (par5b_g1#4): the FP companion chain got the nazgul
    wording."""
    return _register('move the companions on the board, then say '
                     'done')


def die_removed_note(side_name, n):
    """FLD-10e (par5b_g5#20 / g3#14a): an engine-side die removal
    (a die-granting character died) renders AND says so - the silent
    shrink left a phantom client die that wedged the next roll."""
    n = int(n)
    return _register('returning %d %s action %s to Reinforcements '
                     '(the granting character died)'
                     % (n, side_name.replace('_', ' ').lower(),
                        'die' if n == 1 else 'dice'))


def nazgul_strike_choice_note():
    return _register('nazgul with the ring-bearers: say rolling for '
                     'hunt or discarding card <name> (or none)')


def discard_target_ambiguous():
    return _register('several table cards can be discarded - say '
                     'discard <card name>')


def skip_status(message):
    """Report the explicitly requested operator override in the chat pane."""
    return _register(message)


HELP_LINES = (
    'help 1/5: attack = <region> or <region> from <region>; from '
    '<region> answers from?; pass; resync re-reads the board; '
    'skip abandons the current stuck interaction; reset or retry retries stuck work; '
    'rewind restores the saved turn start; help',
    'help 2/5: combat = nc, cc <card>, press, stop, siege, field '
    '(or fight, field battle), retreat <region>, stand',
    'help 3/5: hunt = random, corruption, guide, reveal, no reveal; '
    'fellowship = fp ok, fsp enters mordor',
    'help 4/5: place eyes manually; 0e, 1e, 2e, etc. confirm the total; '
    'done accepts the current allocation; card chains = done, rolling for '
    'hunt, discarding card <name>, or none',
    'help 5/5: turn-start draws, action-dice rolls and dice recovery are automatic; '
    'event cards play from your own hand window',
)
for _hl in HELP_LINES:
    _register(_hl)


def stronghold_tile_note():
    """BL-34 (2026-08-25): the rulebook's extra stronghold hunt tile
    on a reveal into a Shadow-stronghold region fired unannounced and
    read as a double-draw bug."""
    return _register('revealed at a shadow stronghold - drawing a '
                     'stronghold hunt tile')


def lenient_accept_note():
    """BL-32b (2026-08-25): a lenient degrade of a refused gesture
    must be VISIBLE - the fpgame3 tester's published fellowship row
    was silently adopted with no engine execution and no hunt."""
    return _register('your gesture did not match my rules read - '
                     'accepted as given; check the board and say '
                     'resync if it looks wrong')


def undo_took_ai_line():
    """BL-26 (2026-08-25): the client's undo stack is global - a
    human undo can take back the AI's own line while the engine
    keeps it.  FLD-6f (par5b_g5#2): never name a redo button the
    seat lacks - the routes are ones every seat has."""
    return _register('your undo took back MY line and i cannot '
                     're-inject it - say resync and i will restore '
                     'the board (the engine keeps its move)')


def resync_ack(recovered):
    """FLD-6e (par5b_g5#2): the chat 'resync' verb acks in chat -
    it completed silently while the diagnostics showed
    resync_recovery completed."""
    if recovered:
        return _register('resync done - i adopted your client board '
                         'and cleared my stuck machinery; play on')
    return _register('resync checked - nothing was stuck and the '
                     'boards agree; play on')


def draw_ask(deck_label):
    """BL-30 (2026-08-25): human-owed card draws starved silently
    (rolls are asked out loud, draws were not) - fpgame3 froze 60s
    on an unannounced draw."""
    return _register('draw a %s card' % str(deck_label).lower())


def held_choice_note():
    """BL-21(1) (2026-08-25): a human-owed casualty choice whose
    client-drawn options the driver must not guess - said out loud
    instead of riding a silent findings row while the game rolls
    past it."""
    return _register('your casualty choice is open - kill the piece '
                     'on the board, or say random or guide')


def owed_die_kill():
    """BL-9a (2026-08-25): the battle closed but the human's staged
    attack die was never killed - the die->play->resolve->kill
    invariant owes it, and recovery misses a die left in Current
    Action (shgame2: pid 103 stranded, pool census 8 vs 6)."""
    return _register('battle over - kill your staged action die '
                     '(drag it out of Current Action)')


def resolved_event_die_kill():
    return _register('card effect complete - kill your staged action die '
                     '(drag it out of Current Action)')


def bonus_die_drag(side_label, n):
    """At rollover every face is reset; the human adds their new piece
    before the normal draws, declaration, Hunt allocation and pool roll."""
    n = int(n)
    return _register('you gained %d action %s - drag %s from %s '
                     'Reinforcements to %s Dice; wait for the normal '
                     'pool roll after Hunt allocation'
                     % (n, 'die' if n == 1 else 'dice',
                        'it' if n == 1 else 'them',
                        side_label, side_label))


def bonus_die_already_taken(side_label, n_total):
    """USER manual game 2026-08-28: the human had already pulled the
    new Aragorn die out of Reinforcements (and killed it with the
    spent die) before the rollover - the 'drag it from Reinforcements'
    prompt would then send them after the WRONG spare piece.  Say what
    the board shows instead."""
    return _register('your new action die is already on the table - '
                     'Recover prepares your %d %s dice; wait for the '
                     'normal pool roll after Hunt allocation'
                     % (int(n_total), side_label))


def leadership_forfeit_note(count):
    if type(count) is not int or count < 0:
        raise ValueError('Leadership forfeiture must be a nonnegative integer')
    return _register('Dread and Despair: forfeit %d Nazgul leadership' % count)


def siege_press_note(armed=False, paid=False, blocked_stop=False):
    if blocked_stop:
        return _register('your Elite reduction is not yet committed: say press to '
                         'confirm it, or undo that reduction before saying stop')
    if paid:
        return _register('press recorded; your shown Elite reduction has paid for the next siege round')
    if armed:
        return _register('to continue the siege, physically reduce one participating '
                         'Elite of your choice and place its same-nation Regular '
                         'replacement if available; or say stop before paying')
    return _register('say press to continue the siege, then reduce one participating '
                     'Elite on the board; or say stop. Casualties alone do not continue the siege')


def echo_casualties_owed_note(rows):
    """Describe only exact, newly banked army losses and replacements.

    The choice is already committed. These are physical obligations, not a
    fresh casualty menu; mixed/unknown/non-casualty rows retain legacy wording.
    """
    from collections import Counter
    import game_state as gs
    nations = {n.value.casefold(): n for n in gs.Nation}
    shadow = {gs.Nation.SAURON, gs.Nation.ISENGARD, gs.Nation.SOUTHRON_EASTERLINGS}
    regions, sides, losses, arrivals = set(), set(), Counter(), Counter()
    for row in rows:
        kind, n = row.get('kind'), row.get('n')
        region = row.get('region')
        if (kind not in ('unit', 'unit_in') or type(n) is not int or n <= 0
                or not isinstance(region, str) or not region.strip()):
            return None
        parts = str(row.get('tkey', '')).rsplit(':', 1)
        if len(parts) != 2 or parts[0].casefold() not in nations or parts[1] not in ('regular', 'elite'):
            return None
        nation = nations[parts[0].casefold()]
        side = 'shadow' if nation in shadow else 'fp'
        if row.get('side') != side or (kind == 'unit_in' and parts[1] != 'regular'):
            return None
        regions.add(region.strip()); sides.add(side)
        (losses if kind == 'unit' else arrivals)[nation, parts[1]] += n
    if len(regions) != 1 or len(sides) != 1 or not losses:
        return None
    if any(n > losses[nation, 'elite'] for (nation, _kind), n in arrivals.items()):
        return None
    steps = []
    for nation in sorted({n for n, _k in losses}, key=lambda n: n.value):
        regular = losses[nation, 'regular']
        replacement = arrivals[nation, 'regular']
        elite = losses[nation, 'elite'] - replacement
        if regular:
            steps.append('remove %d %s Regular%s (%d hit%s)' % (
                regular, nation.value, '' if regular == 1 else 's', regular, '' if regular == 1 else 's'))
        if replacement:
            steps.append('reduce %d %s Elite%s and place %d %s Regular replacement%s (%d hit%s)' % (
                replacement, nation.value, '' if replacement == 1 else 's', replacement, nation.value,
                '' if replacement == 1 else 's', replacement, '' if replacement == 1 else 's'))
        if elite:
            # Without an arriving Regular the delta cannot distinguish a
            # two-hit removal from a one-hit reduction with no replacement.
            steps.append('remove %d %s Elite%s' % (
                elite, nation.value, '' if elite == 1 else 's'))
    return _register('Complete the committed casualty payment on the board in %s: %s. '
                     'This payment is already recorded; do not take additional losses.' % (
                         next(iter(regions)).title(), '; '.join(steps)))


def echo_owed_note():
    """BL-43/D6 (2026-08-25 shgame5): a chat-committed choice whose
    board legs the human must perform themselves used to bank in
    total silence ('pick 15' looked ignored while the engine waited
    for the drag)."""
    return _register('your choice is committed - now show it on the '
                     'board (move or remove the pieces you chose)')


def casualty_choice_note(region, hits, choices):
    """Explain legal physical losses without soliciting menu indices."""
    options = '; or '.join(label for _i, label in choices)
    return _register('take %d hit%s in %s by removing or reducing your pieces '
                     'on the board: %s. When reducing an Elite, place its Regular replacement. '
                     'Whole-Elite losses count as 2 hits without a chat confirmation.'
                     % (hits, '' if hits == 1 else 's', region, options))


def action_identity(a):
    """BL-96 (fpgame10 G3): the ONE identity key for a menu action,
    shared by the publication (say_pick -> key) and the chat-time
    pick resolver - a pick must commit the ROW THE HUMAN READ, never
    a fresh enumeration's index (3556 card_option rows drifted and a
    separation pick committed an invisible-effect row instead)."""
    return (str(getattr(a, 'action_type', '')),
            getattr(a, 'die_idx', None),
            repr(sorted((getattr(a, 'params', None) or {}).items(),
                        key=lambda kv: kv[0])))


def pick_committed_note(desc):
    """BL-96: every pick commit VOICES what it did - a committed
    answer with no announce is the no-silent-ignore breach in a new
    costume (the Meriadoc separation vanished without a trace)."""
    detail=_prose(desc)
    if '_' in detail or '=' in detail or 'combat:' in detail:
        detail='the action you selected'
    return _register('recorded: %s; complete any required piece changes on the board' % detail)


def search_no_find_note():
    """N7 (shgame14): the Nazgul Search chain closed with no reveal
    and no line at all - the player could not tell found-nothing from
    effect-skipped."""
    return _register('search complete - no nazgul with the '
                     'fellowship, nothing found')


def roll_restore_note(n):
    """BL-120 (shgame14 N1): the refused oversize roll's rolled-Eye
    faces walked into the client hunt box and stood there - the
    counter-drag is demanded, never silently adopted (dice boxes are
    un-adoptable by design)."""
    return _register('that roll was refused - drag %d eye '
                     'di%s back from the hunt box to your dice pool, '
                     'then roll only your unset dice'
                     % (int(n), 'e' if int(n) == 1 else 'ce'))


def step_wedge_note():
    """BL-105 (shgame12 K1's class, generalized): a card-option step
    wedged in an unrecoverable halt loop is resolved by its
    terminator (or first option), loudly - never a frozen session."""
    return _register('that effect could not be completed and was '
                     'resolved by the safest legal choice - play on '
                     '(a bug report was filed)')


def combat_recovery_note():
    text = ('combat recovery: finish the current choice or outstanding piece '
            'changes; the battle and its rolls are retained')
    _MANDATED.add(text)
    return text


def battle_quarantined_note():
    """BL-87 (fpgame9 T8): an unresolvable battle crash-looped 238+
    cycle errors with every recovery lever dead - the breaker drops
    the poisoned pending loudly instead of freezing the session."""
    return _register('that battle could not be resolved and was '
                     'abandoned - the boards are unchanged, your die '
                     'is spent; play on (a bug report was filed)')


def echo_move_note(frm, to):
    """BL-77 (2026-08-25 shgame9): a rules-mandated relocation the
    engine performed inside a commit (the siege move-in above all)
    banked with only the generic owed note - the human never knew
    WHICH army to drag WHERE, and the unperformed echo froze
    reconcile for the rest of the game."""
    return _register('advance your army: drag it from %s into %s '
                     '(the engine already moved it there)'
                     % (frm, to))


def table_discard_cancel_note():
    """BL-40 (2026-08-25 shgame5): the AI discarding its own table
    card to cancel an effect ran in total silence - a legitimate
    Wizard's Staff hunt-cancel read as a broken hunt."""
    return _register('discarding my table card to cancel the effect')


def siege_seat_note():
    """BL-76 (2026-08-25 fpgame8): a formed siege's defenders stayed
    in the field zone - the siege dissolved six times because nobody
    demanded the Stronghold-box drags."""
    return _register('move your garrison into the stronghold box - '
                     'the siege holds only inside it')


def table_seat_note():
    """BL-67 (2026-08-25 fpgame7): a human-played persistent card
    parked in the transient Current-Card zone forever - the Board
    slot is where it lives and blocks."""
    return _register('seat your played table card on a free board '
                     'slot')


def discard_board_note():
    """BL-62 (2026-08-25 fpgame6): a pick/chat-committed discard left
    the physical card in the client hand with no demand - the hand
    census split and the limit looked unenforced."""
    return _register('your discard is committed - remove that card '
                     'from your hand on the board')


def echo_kills_owed_note():
    """fpgame6 blocker (2026-08-25): the AI's own resolution removed
    HUMAN pieces engine-side - the physical removals are the human's
    to perform and were never demanded."""
    return _register('your destroyed pieces are still on the board - '
                     'remove them (my last action resolved their '
                     'loss)')


def echo_arrivals_note():
    """fpgame13 J2: a card recruit banked ARRIVAL rows and the demand
    chat said 'your destroyed pieces...' with nothing destroyed all
    game - the wording is picked by ROW TYPE now."""
    return _register('your recruited pieces are owed - drag them in '
                     'from your reinforcements')


def cc_reveal_drag():
    """BL-41 (2026-08-25 shgame5): a chat-committed combat card has
    no physical counterpart on the table - at reveal the human drags
    it, and the table mopup disposes it (otherwise the client hand
    keeps the card forever)."""
    return _register('reveal your committed combat card - drag it '
                     'from your hand to the table')


def turn_over_recover():
    """Announce the automatic native bookkeeping once at turn end."""
    return _register('turn over - recovering both dice pools and drawing '
                     'the next turn\'s cards automatically')


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


def stall_repair_ask(side, kind, n_options, n_kills, to_act, last_killed,
                     log_name):
    """USER 2026-08-28 (self-repair): when the lane stalls the AI says
    EXACTLY what it is stuck on and EXACTLY what it will do, and does
    nothing until the human says yes.  No card names, no menu."""
    from .decision_prompts import decision_label
    return _register(
        'i am stuck on the %s %s decision: your last board line matches '
        'none of its %d legal options. i can self-repair: my log is saved '
        'as %s; i would drop my stuck state, kill the %d staged die and '
        'card piece(s), resync my board to your client with the political '
        'track from my engine, and resume with %s to act since the last '
        'die killed was %s. say yes to do this, or no and use the panel '
        'to reset and resync, skip the decision, or go manual'
        % (_prose(side), decision_label(kind), int(n_options), str(log_name),
           int(n_kills), _prose(to_act), _prose(last_killed)))


def stall_repair_declined():
    return _register('ok, nothing changed; use the panel to reset and '
                     'resync, skip the decision, or undo your last line')


def stall_repair_done(to_act, n_kills, log_name):
    return _register(
        'self-repair done: log %s saved, %d staged piece(s) killed, board '
        'resynced to your client, political track from my engine; %s to '
        'act' % (str(log_name), int(n_kills), _prose(to_act)))


def refusal_closest(description, differing_keys):
    return _register('closest legal move: %s - your gesture differs '
                     'on %s' % (_prose(description),
                                ', '.join(_prose(k) for k in
                                          list(differing_keys)[:3])))


def divergence_gate_hold():
    """FLA-9 (par5 g5#9): a fully legal, fully rendered move was
    force-undone by the gate with an impossible closest-legal
    ('Attack Lorien from Lorien') because the engine judged it
    against a board a CONFIRMED divergence had already said was
    wrong.  Held, never undone, retried when the boards agree."""
    return _register('your move is HELD, not undone - my board and '
                     'your client disagree right now, so I cannot '
                     'judge it; say adopt to take your client board '
                     'or resync, and I will re-check your move')


def refusal_cannot_pass(detail=None):
    t = 'you cannot pass'
    if detail:
        t += ' - %s' % _prose(detail)
    return _register(t)


# ---------------------------------------------------------------------
# BRLANE fix wave (2026-08-29): the lanes that used to answer with
# silence or a panel note now answer in chat, through registered prose.
def refusal_cannot_pass_counts(mine, theirs, hunt):
    """BRLANE-10 / M11: a rule-correct pass refusal names the numbers it
    was judged on (amendment 1.223: unused ACTION dice only; Hunt Box
    dice never count) - four USER games could not tell the rule from a
    desync because the refusal printed neither count."""
    return _register(
        'you cannot pass - you have %d unused dice, i have %d (%d in the '
        'hunt box do not count); a pass needs fewer unused dice than the '
        'opponent' % (int(mine), int(theirs), int(hunt)))


def chat_not_a_command(context):
    """BRLANE-4 / M7: free text that matches no grammar is never
    silent - the reply names what IS accepted right now."""
    return _register('that is not a command here - %s; say help for the '
                     'words i understand' % _prose(context))


def card_window_region_held():
    """BRLANE-4 / BRLANE-12: a region chatted while the human's card-play
    window is open is the card's TARGET, not an attack declaration."""
    return _register('a card-play window is open - kill the die to '
                     'complete the play; i will use that region as its '
                     'target if the card asks for one')


def undo_ai_line_reasserted():
    """BRLANE-7 / BL-26: the AI re-plays its own undone line itself, so
    a redo would double it."""
    return _register('your undo took back my line - i re-played it; do '
                     'not press redo')


def held_window_expired(what):
    """BRLANE-2: a held gesture whose blocking condition never cleared
    is voiced, never aged out."""
    return _register('your held gesture (%s) never became playable - undo '
                     'it and redo it once the open decision is answered'
                     % _prose(what))


def emission_unacknowledged_note(what):
    """A missing or invalid acknowledgement does not prove a missing move."""
    return _register('I could not confirm my last move (%s). Play is paused; '
                     'check the board before using resync. Do not repeat '
                     'casualties already shown on the board' % _prose(what))


def draw_ask_ambiguous():
    """BRLANE-14: a starving human-deck draw whose deck depends on which
    candidate the gesture was."""
    return _register('a card draw is owed but which deck depends on your '
                     'gesture - draw only from the deck the card you '
                     'played grants')


def battle_by_hand_ask(what, words):
    """Item 16 / M7: the human resolved the AI's battle by dragging its
    pieces while their own combat decision is still open."""
    from .decision_prompts import decision_label
    return _register('the battle still needs your %s: %s; or undo the '
                     'uncommitted drags' % (decision_label(what), _prose(words)))


def card_play_extra_rows(card, rows):
    """Item 14 / M10: the card leaving the hand IS the play; rows the
    engine did not produce are explained, never refused."""
    return _register('played %s; your gesture also changed %s, which is '
                     'not part of that card - drag it back or say resync'
                     % (_prose(card), _prose(rows)))


def census_adopt_ask(n_rows, first_row, log_name):
    """BRCMP design 3 (handoff): the CONFIRMED board divergence ASK in
    chat - the human answers yes / no."""
    return _register(
        'my board and your client disagree on %d piece row(s) (first: %s), '
        'unchanged for 10 seconds with nothing in flight; my log is saved '
        'as %s. say yes and i adopt your client board into my engine '
        '(pieces, sieges, captures, casualties as shown), or no and i keep '
        'my engine and you move the pieces to match it'
        % (int(n_rows), _prose(first_row)[:120], str(log_name)))


def census_adopt_done(n_rows):
    return _register('done: adopted your client board into my engine '
                     '(%d row(s))' % int(n_rows))


def census_adopt_declined_reply():
    """SQF-11: the census 'no' (and any later yes/no while the
    divergence stands) names the CLI-reachable standing routes - the
    one-shot ask must never be a dead end from the seat."""
    return _register(
        'ok - i keep my board and the divergence stands. say adopt at '
        'any time and i adopt your client board, say resync to reset '
        'and resync, or move the client pieces to match me and i '
        'retract; the panel can also skip or undo')


def die_parked_note(pid):
    """SQF-6: the client recover parked a live die out of its pool."""
    return _register(
        'your client parked action die %s in a reinforcements box but '
        'that die is still in your unspent pool here - drag it back '
        'to your dice pool or your next roll will come up short'
        % str(pid))


def partial_hunt_corruption_note(damage, current, target, remaining):
    return _register(
        'this Hunt has %d damage; %d more corruption reaches the full '
        'track total. '
        'to take it all as corruption, move the marker from %d to %d '
        'or say corruption. to choose a companion instead, undo the '
        'partial marker move first, then say guide or random'
        % (int(damage), int(remaining), int(current), int(target)))


def elven_ring_choice_note(face_changed=False):
    if face_changed:
        return _register(
            'Elven Ring: waiting for the ring payment and an unambiguous '
            'selected die (the client piece ID is needed when faces repeat).')
    return _register(
        'Elven Ring ready: use your chosen die and carry out the action on '
        'the board. Undo the ring counter to cancel.')


def gandalf_draw_choice_note(deck):
    deck = str(deck).lower()
    if deck not in ('character', 'strategy'):
        raise ValueError('Gandalf draw requires a Character or Strategy deck')
    return _register(
        'Gandalf guide draw is optional: draw one Free %s card '
        'on the board, or say draw; say decline for no bonus draw.' % deck.title())


def elven_ring_finish_note():
    return _register(
        'Use your chosen die to finish the Elven Ring action, or undo '
        'its counter before passing.')


def combat_target_choice_note(title, target, count=None):
    """Announce the selected public figure, preserving a supplied magnitude."""
    detail = _prose(str(target))
    if count is not None:
        detail += ' (count %d)' % int(count)
    return _register('%s: i choose %s' % (_prose(title), detail))


def anduril_conversion_note(text):
    """Sanction only an exact resolved Anduril cost/conversion fact."""
    from .anduril_reports import parse_report
    if parse_report(text) is None:
        raise ValueError('invalid Anduril conversion report')
    return _register(text)


def fixed_leadership_forfeit_note(card_id, count):
    """Name an observed fixed card cost, without manufacturing a choice."""
    from cards import get_card
    from combat_forfeit_text import fixed_forfeit_report
    card = get_card(str(card_id))
    if card is None:
        raise ValueError('fixed leadership report has no recognized card')
    text = '%s: forfeit %d Nazgul leadership' % (card.combat_name, int(count))
    if fixed_forfeit_report(text) is None:
        raise ValueError('fixed leadership report does not match the printed cost')
    return _register(text)


def combat_effect_choice_note(params):
    """Public decision text names the combat face, never its unrelated event."""
    from cards import get_card
    from combat import combat_effect_kind
    card = get_card(str(params.get('card_id', '')))
    if card is None:
        raise ValueError('combat decision has no recognized card')
    title = str(card.combat_name).strip()
    effect = combat_effect_kind(title) or title.casefold().replace(' ', '_')
    target = params.get('target')
    count = params.get('count')
    if effect == 'dread_despair':
        return leadership_forfeit_note(int(count or 0))
    if count == 0 and target is None:
        return _register('%s: decline' % title)
    if effect == 'heroic_death' and target is not None and count is not None:
        return _register('Heroic Death: sacrifice %s; cancel %d hits'
                         % (_prose(str(target)), int(count)))
    if effect in ('black_breath', 'words_of_power') and target is not None:
        return _register('%s: target %s' % (title, _prose(str(target))))
    if effect == 'relentless_assault' and count is not None:
        return _register('Relentless Assault: take %d hits' % int(count))
    if effect == 'onslaught' and count is not None:
        return _register('Onslaught: roll %d extra dice' % int(count))
    if target is not None:
        return combat_target_choice_note(title, target, count)
    return _register('%s: i choose %d' % (title, int(count or 0)))

def heroic_death_incoming_note(hits, human_owner):
    text = 'Heroic Death: %d incoming hit%s' % (int(hits), '' if int(hits) == 1 else 's')
    if human_owner:
        text += '; name the figure and how many hits to cancel, or say 0 to decline'
    else:
        text += ' before my prevention choice'
    return _register(text)


def selected_card_intent_note():
    return _register('Your card is selected, but no attack is registered. If attacking, '
            'name the attacked region (or <target> from <source>); otherwise '
            'play the event card or deselect it. Your die has not been spent.')
