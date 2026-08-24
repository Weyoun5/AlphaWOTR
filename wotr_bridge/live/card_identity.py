"""AI-draw card identity from the raw request stream.

SANCTIONED RAW IMPORT #1 (rendered-baseline directive; REBUILD_PLAN
Phase 3 item 2): exactly two client statements live only on the raw
channel - this one is the COMMITMENT DECODE of the AI's OWN drawn
cards.  The rendered feed says only 'draws Shadow Character Card.';
which physical card that is exists nowhere but in the client's
commitment tables (fact 10: at hand-claim Game.I()/J() reassign every
deck piece's identity from the aY commitment shuffle, so the static
alphabetical pid map is WRONG in a claimed game - PROTOCOL.md, the
root cause of the old card desync).

LIFTED (donor read-only, nothing imported from it):
  ai_move_server._build_pid_card_map / _client_name_to_cid
                                :539-603, :12254-12319 -> name_to_cid
  ai_move_server._live_card_maps decode use  :12321-12375 ->
                                CardIdentity.harvest (per-draw, not
                                whole-roster: the live driver needs
                                identity per draw EVENT, in feed order)
  raw draw shape: the strict draw-TO-HAND pattern of
  src/rendered_extract.py:111-113 (pattern copied, module NOT imported
  - a live consumer imports nothing from rendered_extract).

BLINDING (design rule 8, fail-closed BY CONSTRUCTION):
  * only the AI side's password is ever set; the human side's decoder
    password is a poison string no client accepts, so the human's
    commitments can NEVER crack in this process even if asked;
  * harvest() decodes ONLY draws whose destination is the AI's OWN
    hand zone (Controls.45=FP hand, 46=SH hand - fact 3: attribution
    by ZONE, never nick) AND whose deck is the AI side's;
  * a human draw line yields nothing - not even a decode attempt.

The CardDecoder import comes from log_replay's re-export - the ONE
sanctioned log_replay surface for live consumers (REBUILD_PLAN keep
table).
"""
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, '..', '..', 'src'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# the strict draw-to-hand pattern.  EVERY deck menu has its OWN
# Controls key family (donor _DRAW_LINE_RE, ai_move_server.py:96-98:
# 'Controls.491/509/0/549/557 vary by menu, the zone does not');
# the deck ZONE token identifies side+deck and the hand tail
# distinguishes a real draw from other deck events.  LIVE SPECIMENS
# (boot2 leg, 2026-08-19 -- the 509-only shape matched NONE of the
# AI's own draws and wedged the ai_draws step):
#   $Wey<~Controls.491~>117<~Controls.492~><~Game.2004~><~Controls.493~><~Controls.45~>
#   $Wey<~Controls.509~>119<~Controls.510~><~Game.2005~><~Controls.511~><~Controls.45~>
#   $Wey<~Controls.0~>200<~Controls.549~><~Game.2006~><~Controls.550~><~Controls.46~>
RAW_DRAW_RE = re.compile(r'<~Controls\.(?:491|509|0|549|557)~>(\d+)'
                         r'<~Controls\.\d+~><~Game\.(200[4-7])~>'
                         r'<~Controls\.\d+~><~Controls\.(45|46)~>')

# deck zone -> (side_name, deck key, owning hand zone, decoder type)
DECK_INFO = {
    '2004': ('FREE_PEOPLES', 'character', '45', 'FP Character'),
    '2005': ('FREE_PEOPLES', 'strategy', '45', 'FP Strategy'),
    '2006': ('SHADOW', 'character', '46', 'SA Character'),
    '2007': ('SHADOW', 'strategy', '46', 'SA Strategy'),
}

# a password no client ever accepts: guarantees the human side's
# commitment tables never crack in this process (rule 8)
_POISON_PW = '\x00__never_a_client_password__\x00'


def _norm(s):
    import unicodedata
    s = unicodedata.normalize('NFKD', str(s)).encode(
        'ascii', 'ignore').decode()
    return re.sub(r'[^a-z0-9]', '', s.lower())


_NAME_IDX = None
_NAME_BUCKETS = None


def _build_name_index():
    """(side, deck, normalized-title) -> cid index over cards.py
    (donor :12254-12286, logic kept; accent/subtitle tolerant)."""
    global _NAME_IDX, _NAME_BUCKETS
    if _NAME_IDX is not None:
        return
    from cards import get_cards
    idx = {}
    buckets = {}
    for cid, c in get_cards().items():
        sd = 'FP' if cid.startswith('FREE_PEOPLES') else 'SA'
        dk = 'Character' if '_Character_' in cid else 'Strategy'
        names = set()
        for attr in ('card_name', 'combat_name', 'event_name', 'name'):
            v = getattr(c, attr, None)
            if v:
                names.add(_norm(v))
        names.add(_norm(cid.split('_', 2)[-1]))
        names = {n for n in names if n}
        buckets.setdefault((sd, dk), []).append((cid, names))
        for nm in names:
            idx.setdefault((sd, dk, nm), cid)
            pk = (sd, dk, nm[:12])
            # a 12-char prefix shared by two DIFFERENT cards is ambiguous:
            # mark it so only a unique prefix ever resolves
            if pk in idx and idx[pk] != cid:
                idx[pk] = None
            else:
                idx.setdefault(pk, cid)
    _NAME_IDX = idx
    _NAME_BUCKETS = buckets


# client titles that differ from the engine name beyond a leading
# article ('A Balrog is Come!' = Balrog_of_Moria) - donor :12301-12303
_KEYWORD_ALIASES = (('balrog',), ('muster', 'long'), ('corsair',),
                    ('nazgul', 'strike'), ('prove', 'swifter'),
                    ('wisdom', 'elrond'))


def name_to_cid(name, deck_type):
    """Engine card id for a client card title, within the title's deck
    bucket (donor _client_name_to_cid :12254-12319, logic kept).
    None = unresolved (the caller decides how loud to be)."""
    _build_name_index()
    sd = 'FP' if str(deck_type).startswith('FP') else 'SA'
    dk = 'Character' if 'Character' in str(deck_type) else 'Strategy'
    n0 = _norm(name)
    n1 = _norm(re.sub(r'^(the|a)\s+', '', str(name), flags=re.I))
    hit = (_NAME_IDX.get((sd, dk, n0)) or _NAME_IDX.get((sd, dk, n1))
           or _NAME_IDX.get((sd, dk, n0[:12]))
           or _NAME_IDX.get((sd, dk, n1[:12])))
    if hit:
        return hit
    for kws in _KEYWORD_ALIASES:
        if not all(k in n0 for k in kws):
            continue
        for cid, names in _NAME_BUCKETS.get((sd, dk), []):
            blob = ''.join(sorted(names)) + _norm(cid)
            if all(k in blob for k in kws):
                return cid
    # last resort: unique containment within the deck bucket
    cands = set()
    for cid, names in _NAME_BUCKETS.get((sd, dk), []):
        for nm in names:
            if n0 and (n0 in nm or nm in n0 or n1 in nm or nm in n1):
                cands.add(cid)
                break
    if len(cands) == 1:
        return cands.pop()
    return None


class CardIdentity(object):
    """Incremental decoder over the raw request stream (the client log
    AiOpp sends whole every tick).

    harvest(raw_lines) consumes only lines not yet seen (the log is
    append-only within a session; a SHRUNK log resets the cursor and is
    surfaced in the result so the driver can treat it as a restart) and
    returns one dict per NEW AI-side draw:
        {'side', 'deck', 'cid', 'pid', 'name'}
    cid is None when the commitment decode or the name->cid mapping
    failed - the caller must HALT loudly (donor rule: undecodable AI
    cards are a LIVE HALT, never a guess)."""

    def __init__(self, ai_side_name, password='ai'):
        if ai_side_name not in ('SHADOW', 'FREE_PEOPLES'):
            raise ValueError('ai_side_name must be SHADOW or '
                             'FREE_PEOPLES, got %r' % (ai_side_name,))
        from log_replay import CardDecoder      # sanctioned import
        self.ai_side = ai_side_name
        self.decoder = CardDecoder()
        if ai_side_name == 'SHADOW':
            self.decoder.sa_password = password
            self.decoder.fp_password = _POISON_PW
        else:
            self.decoder.fp_password = password
            self.decoder.sa_password = _POISON_PW
        self._seen = 0              # raw lines already processed
        self._claims = set()        # claim lines already brute-forced
        self.restarted = False      # a shrunk log was observed

    def harvest(self, raw_lines):
        lines = list(raw_lines)
        if len(lines) < self._seen:
            # the client log can only shrink across a game restart -
            # reset and surface (the driver's feed reader raises its
            # own FeedRestart for the rendered channel)
            self._seen = 0
            self.restarted = True
        new = lines[self._seen:]
        self._seen = len(lines)
        out = []
        for ln in new:
            if 'Controls.464' in ln or 'Controls.479' in ln:
                if ln not in self._claims:
                    self._claims.add(ln)
                    # brute-forces ONLY sides whose password is set
                    # correctly; the poison side never cracks
                    self.decoder.preprocess_log([ln])
            m = RAW_DRAW_RE.search(ln)
            if m is None:
                continue
            pid, deck_code, hand = m.group(1), m.group(2), m.group(3)
            side, deck, hand_zone, deck_type = DECK_INFO[deck_code]
            if side != self.ai_side or hand != hand_zone:
                # the HUMAN's draw (or a cross-zone oddity): identity
                # is never decoded here - blinding by construction
                continue
            num = self.decoder.decode_piece(int(pid), deck_type)
            name = (self.decoder.get_card_name(num, deck_type)
                    if num is not None else None)
            cid = name_to_cid(name, deck_type) if name else None
            out.append({'side': side, 'deck': deck, 'cid': cid,
                        'pid': pid, 'name': name})
        return out
