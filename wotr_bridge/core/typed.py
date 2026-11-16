"""What a typed attack names: the regions' short forms, and a word given with the attack.

Owner: "if there is only 1 army next to osgiliath, i can say osg nc. or osg from si nc."

A typed attack is  [attack] <target> [from <source>] [nc | cc <card> | press | stop | continue].
  * A region may be typed whole, by one of the players' own short words, or SHORT:
      - the players' words are ONE list, the one the parser of their game logs reads
        (log_replay.LogReplayer._REGION_ABBREVS: "wlr" - woodland realm, "west" - westemnet, "fords" - fords of
        isen, "ofr", "mt" ...): community();
      - short (short_for): its first letters, three or more ("osg"); the first letter of each of its words
        ("si" - south ithilien); the first letters of each word in turn ("s ith", "min tir"); or the first
        letters, three or more, of any ONE of its words ("tirith", "morgul").
  * What is typed is read among the attacks that are legal NOW - the regions that can be attacked, and the armies
    that can attack those.  In this order: the whole name; a word of the players' list whose region is among
    them; a short form.  One that fits several is not guessed: the caller's verdict is "several" and the person
    is told what he can type.
  * The last word may be his answer to a question of the battle the attack opens (bundled): it is cut off here,
    and kept by the caller for that question.

Every line and name comes in as a chat key (lower case, no punctuation), and the caller says how an option's
regions are named; of the game this module knows the players' list alone.
"""
import re

SHORT_MIN = 3       # the fewest first letters of a name, or of one of its words, that are a short form of it
_LIST = None


def _key(name):
    return ' '.join(re.sub(r'[^a-z0-9 ]', ' ', str(name or '').lower().replace("'", '')).split())


def community():
    """{the players' short word: its region}, both as plain lower-case words."""
    global _LIST
    if _LIST is None:
        import log_replay
        _LIST = {_key(word): _key(region) for word, region in log_replay.LogReplayer._REGION_ABBREVS.items()}
    return _LIST


def short_for(typed, name):
    """Is `typed` a short form of the region `name`?"""
    parts, said = name.split(), typed.split()
    if len(typed) >= SHORT_MIN and name.startswith(typed):
        return True
    if len(parts) > 1 and typed == ''.join(p[0] for p in parts):
        return True
    if len(parts) > 1 and len(said) == len(parts) and all(p.startswith(s) for p, s in zip(parts, said)):
        return True
    return len(parts) > 1 and len(said) == 1 and len(typed) >= SHORT_MIN and any(p.startswith(typed) for p in parts)


def meant(typed, places):
    """The names among `places` ({the name as it is typed: its region key}) that `typed` means: the one it spells
    out; else the one the players' list gives that word; else every one it is a short form of."""
    names = sorted(places)
    region = community().get(typed)
    return [n for n in names if n == typed] or [n for n in names if region and _key(places[n]) == region] \
        or [n for n in names if short_for(typed, n)]


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
