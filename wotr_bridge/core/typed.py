"""What a typed attack names: the regions' abbreviations, and a word given with the attack.

Owner: "if there is only 1 army next to osgiliath, i can say osg nc. or osg from si nc."

A typed attack is  [attack] <target> [from <source>] [nc | cc <card> | press | stop | continue].
  * A region is typed whole or by one of the players' own abbreviations, and by nothing else (owner: "remove
    the auto generated abbreviations you added, only do the ones i explicitly listed").  The abbreviations are
    ONE list, the one the parser of the players' game logs reads (log_replay.LogReplayer._REGION_ABBREVS:
    "osg" - osgiliath, "si" - south ithilien, "wlr" - woodland realm ...): community().  A new one is added there.
  * What is typed is read among the attacks that are legal NOW - the regions that can be attacked, and the armies
    that can attack those: the whole name first, then the abbreviation.
  * The last word may be his answer to a question of the battle the attack opens (bundled): it is cut off here,
    and kept by the caller for that question.

Every line and name comes in as a chat key (lower case, no punctuation), and the caller says how an option's
regions are named; of the game this module knows the players' list alone.
"""
import re

_LIST = None


def _key(name):
    return ' '.join(re.sub(r'[^a-z0-9 ]', ' ', str(name or '').lower().replace("'", '')).split())


def community():
    """{the players' abbreviation: its region}, both as plain lower-case words."""
    global _LIST
    if _LIST is None:
        import log_replay
        _LIST = {_key(word): _key(region) for word, region in log_replay.LogReplayer._REGION_ABBREVS.items()}
    return _LIST


def meant(typed, places):
    """The names among `places` ({the name as it is typed: its region key}) that `typed` means: the one it spells
    out, else the one the players' list gives that abbreviation."""
    region = community().get(typed)
    return [n for n in sorted(places) if n == typed] or [n for n in sorted(places) if region and _key(places[n]) == region]


def attacks(text, menu, name):
    """A typed attack against the attacks of the menu -> (the options it names, does it name a place of any
    attack at all).  name(region key) -> the region as it is typed."""
    rest = text[7:] if text.startswith('attack ') else text
    to, _, frm = rest.partition(' from ')
    rows = [(a, name(a.params.get('to')), name(a.params.get('from'))) for a in menu if a.action_type == 'army_attack']
    targets = meant(to, {t: a.params.get('to') for a, t, _ in rows})
    hits = [row for row in rows if row[1] in targets]
    if frm:
        sources = meant(frm, {f: a.params.get('from') for a, _, f in hits})
        hits = [row for row in hits if row[2] in sources]
    half = bool(frm) and bool(targets or meant(frm, {f: a.params.get('from') for a, _, f in rows}))
    return [a for a, _, _ in hits], half


def bundled(text, early, menu, name):
    """A line that gives an attack of the menu AND, with it, his answer to a question of the battle that attack
    opens ("osg nc", "dale from erebor cc <card>") -> (the attack's words, the word kept for that question, the
    question), else (text, None, None).  early: word -> its question, for the words that may be given so."""
    parts = text.split()
    head, with_it, asks = text, None, None
    if len(parts) > 1 and parts[-1] in early:
        head, with_it, asks = ' '.join(parts[:-1]), parts[-1], early[parts[-1]]
    elif ' cc ' in text:
        head, _, card = text.partition(' cc ')
        with_it, asks = 'cc ' + card, 'card'
    if with_it and head and attacks(head, menu, name)[0]:
        return head, with_it, asks
    return text, None, None
