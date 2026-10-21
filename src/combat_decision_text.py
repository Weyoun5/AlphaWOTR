"""Exact public combat-choice grammar. Context and legality belong to callers."""
import re
import unicodedata

_NAMES = {'heroic death': 'heroic_death', 'black breath': 'black_breath',
          'words of power': 'words_of_power', 'relentless assault': 'relentless_assault',
          'onslaught': 'onslaught'}

def _clean(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', str(text or ''))
                   if not unicodedata.combining(c)).strip()


def _legacy_choice(title, body, card):
    """Former emitter grammar, trusted only with the actual revealed card."""
    from cards import get_card
    card = get_card(getattr(card, 'card_id', card))
    if card is None:
        return None
    effect = _NAMES.get(_clean(card.combat_name).casefold())
    if effect is None:
        return None
    # These are the same explicit Java/parser event-name aliases used by the
    # old emitter. The remaining event titles differ only by normalization.
    aliases = {'wisdomofelrond': 'The Wisdom of Elrond',
               'celebornsgaladhrim': "Celeborn's Galadrim",
               'grimbeorntheoldsonofbeorn': 'Grimbeorn the Old',
               'nazgulstrike': 'The Nazgul Strike!',
               'balrogofmoria': 'A Balrog is Come!',
               'musteringsoflongplannedwar': 'Mustering of a Long-planned War'}
    event = _clean(card.card_name)
    key = re.sub(r'[^a-z0-9]', '', event.casefold())
    titles = {event.casefold(), _clean(aliases.get(key, event)).casefold()}
    count = re.fullmatch(r'i choose (\d+)', body, re.I)
    allowed = titles | {name[:40] for name in titles} if count else titles
    if title.strip().casefold() not in allowed:
        return None
    if count:
        value = int(count[1])
        if value == 0:
            return {'effect': effect, 'decline': True, 'count': 0}
        return {'effect': effect, 'count': value} if effect in ('relentless_assault', 'onslaught') else None
    if effect == 'heroic_death':
        target = re.fullmatch(r'i choose (.+?) \(count (\d+)\)', body, re.I)
        if target:
            return {'effect': effect, 'target': target[1], 'count': int(target[2])}
    elif effect in ('black_breath', 'words_of_power'):
        target = re.fullmatch(r'i choose (.+)', body, re.I)
        if target:
            return {'effect': effect, 'target': target[1]}
    return None


def parse_choice(text, card=None):
    text = ''.join(c for c in unicodedata.normalize('NFKD', str(text or ''))
                   if not unicodedata.combining(c)).strip().rstrip('.!')
    match = re.fullmatch(r'([^:]+):\s*(.+)', text)
    if not match:
        return None
    effect = _NAMES.get(match[1].strip().casefold())
    body = match[2].strip()
    if effect is None:
        return _legacy_choice(match[1], body, card) if card is not None else None
    if body.casefold() == 'decline':
        return {'effect': effect, 'decline': True, 'count': 0}
    if effect == 'heroic_death':
        chosen = re.fullmatch(r'sacrifice\s+(.+?);\s*cancel\s+(\d+)\s+hits?', body, re.I)
        if chosen:
            return {'effect': effect, 'target': chosen[1].strip(), 'count': int(chosen[2])}
    elif effect in ('black_breath', 'words_of_power'):
        chosen = re.fullmatch(r'target\s+(.+)', body, re.I)
        if chosen:
            return {'effect': effect, 'target': chosen[1].strip()}
    else:
        pattern = r'take\s+(\d+)\s+hits?' if effect == 'relentless_assault' else r'roll\s+(\d+)\s+extra\s+(?:die|dice)'
        chosen = re.fullmatch(pattern, body, re.I)
        if chosen:
            return {'effect': effect, 'count': int(chosen[1])}
    return None
