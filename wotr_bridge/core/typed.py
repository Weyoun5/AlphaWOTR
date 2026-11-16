"""What a typed attack names: the regions' short forms, and a word given with the attack.

Owner: "if there is only 1 army next to osgiliath, i can say osg nc. or osg from si nc."

A typed attack is  [attack] <target> [from <source>] [nc | cc <card> | press | stop | continue].
  * A region may be typed whole or SHORT (short_for): its first letters, three or more ("osg"); the first letter
    of each of its words ("si" - south ithilien); or the first letters of each word in turn ("s ith", "min tir").
  * What a short form means is read among the attacks that are legal NOW - the regions that can be attacked, and
    the armies that can attack those - never from a table of the whole map.  One that fits several is not guessed:
    the caller's verdict is "several" and the person is told what he can type.  A whole name is never taken for a
    longer one that begins with it.
  * The last word may be his answer to a question of the battle the attack opens (bundled): it is cut off here,
    and kept by the caller for that question.

Nothing here knows the game: every line and name comes in as a chat key (lower case, no punctuation), and the
caller says how an option's regions are named.
"""

SHORT_MIN = 3       # the fewest first letters of a name that are a short form of it


def short_for(typed, name):
    """Is `typed` a short form of the region `name`?"""
    parts, said = name.split(), typed.split()
    if len(typed) >= SHORT_MIN and name.startswith(typed):
        return True
    if len(parts) > 1 and typed == ''.join(p[0] for p in parts):
        return True
    return len(parts) > 1 and len(said) == len(parts) and all(p.startswith(s) for p, s in zip(parts, said))


def meant(typed, names):
    """The names among `names` that `typed` means: the one it spells out, else every one it is a short form of."""
    return [n for n in names if n == typed] or [n for n in names if short_for(typed, n)]


def attacks(text, menu, name):
    """A typed attack against the attacks of the menu -> (the options it names, does it name a place of any
    attack at all).  name(region key) -> the region as it is typed."""
    rest = text[7:] if text.startswith('attack ') else text
    to, _, frm = rest.partition(' from ')
    rows = [(a, name(a.params.get('to')), name(a.params.get('from'))) for a in menu if a.action_type == 'army_attack']
    targets = meant(to, sorted({t for _, t, _ in rows}))
    hits = [row for row in rows if row[1] in targets]
    if frm:
        sources = meant(frm, sorted({f for _, _, f in hits}))
        hits = [row for row in hits if row[2] in sources]
    half = bool(frm) and bool(targets or meant(frm, sorted({f for _, _, f in rows})))
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
