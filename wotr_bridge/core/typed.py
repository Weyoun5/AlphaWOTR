"""What a typed attack names: the regions' abbreviations and short forms, and a word given with the attack.

Owner: "if there is only 1 army next to osgiliath, i can say osg nc. or osg from si nc."

A typed attack is  [attack] <target> [from <source>] [nc | cc <card> | press | stop | continue].
  * A region is typed, and read in this order:
      - whole;
      - by one of the players' own abbreviations: the list the parser of their game logs reads
        (log_replay.LogReplayer._REGION_ABBREVS: "osg", "si", "wlr" ...) and, beside it, CHAT_ONLY - the ones the
        published README names that the parser's list lacks, and the owner's "ni".  community().  A new one is
        added to one of the two.  An abbreviation means its own region and never another: when that region cannot
        be attacked now, it names nothing;
      - SHORT, the published README's rule (owner: "implement that readme rule. any unique 4 letters ... so w
        harondor works. or if it's say, dol g"): four letters or more that are the start of the name ("osgi",
        "dol g"), else whose words are, in their order, the starts of words of the name ("w harondor", "n ithilien",
        "tirith") - short_for.
  * What is typed is read among the attacks that are legal NOW - the regions that can be attacked, and the armies
    that can attack those.  A short form that fits several of them is not guessed: the caller's verdict is
    "several" and the person is told what he can type.
  * The last word may be his answer to a question of the battle the attack opens (bundled): it is cut off here,
    and kept by the caller for that question.

Every line and name comes in as a chat key (lower case, no punctuation), and the caller says how an option's
regions are named; of the game this module knows the players' list alone.
"""
import re

SHORT_MIN = 4       # the fewest letters of a short form
_LIST = None
# the abbreviations of the published README ("Common aliases include dg, hd, mt, da, mg, mm, lor, wlr, ofr, pel,
# shire, riv, ereb and gh") that the parser's list does not hold, the other spelling of one name, and the owner's
CHAT_ONLY = {'shire': 'the shire', 'ereb': 'erebor', 'gh': 'grey havens', 'gray havens': 'grey havens',
             'ni': 'north ithilien'}


def _key(name):
    return ' '.join(re.sub(r'[^a-z0-9 ]', ' ', str(name or '').lower().replace("'", '')).split())


def community():
    """{the players' abbreviation: its region}, both as plain lower-case words."""
    global _LIST
    if _LIST is None:
        import log_replay
        both = dict(CHAT_ONLY, **log_replay.LogReplayer._REGION_ABBREVS)
        _LIST = {_key(word): _key(region) for word, region in both.items()}
    return _LIST


def short_for(typed, name):
    """Is `typed` a short form of the region `name`?  2: the start of the name; 1: its words are, in their order,
    the starts of words of the name; 0: no, or fewer than SHORT_MIN letters."""
    said = typed.split()
    if sum(len(w) for w in said) < SHORT_MIN:
        return 0
    if name.startswith(typed):
        return 2
    i = 0
    for word in name.split():
        if i < len(said) and word.startswith(said[i]):
            i += 1
    return 1 if i == len(said) else 0


def meant(typed, places):
    """The names among `places` ({the name as it is typed: its region key}) that `typed` means."""
    names, region = sorted(places), community().get(typed)
    whole = [n for n in names if n == typed]
    if whole or region:
        return whole or [n for n in names if _key(places[n]) == region]
    return [n for n in names if short_for(typed, n) == 2] or [n for n in names if short_for(typed, n)]


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
