"""Native temporary seats do not change an action die's owner or payment."""

TEMPORARY_ACTION_ZONES = frozenset((
    'current action', 'current fp card', 'current sa card',
))
HUNT_ZONE = 'the hunt for the ring'


def zone(value):
    text = str(value).strip().lower()
    return {'fp dead': 'fp casualties', 'sa dead': 'sa casualties',
            'fp card play': 'current fp card', 'sa card play': 'current sa card',
            'hunt box': HUNT_ZONE}.get(text, text)


def temporary(value):
    return zone(value) in TEMPORARY_ACTION_ZONES


def pool(side):
    return 'fp dice' if side == 'FREE_PEOPLES' else 'sa dice'


def casualties(side):
    return 'fp casualties' if side == 'FREE_PEOPLES' else 'sa casualties'
