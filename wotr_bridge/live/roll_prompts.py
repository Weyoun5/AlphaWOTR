"""Public dice context; never reads a hand or changes the random tape."""

_KINDS = {
    'combat roll': 'normal combat roll',
    'leader reroll': 'leader re-roll',
    'hunt roll': 'Hunt roll',
    'hunt reroll': 'Hunt re-roll',
    'dwarven_ring': 'Dwarven Ring re-roll',
    'sacrifice hits': 'Onslaught attack',
    'shelob d6 damage': 'Shelob damage roll',
    'event card': 'event card roll',
    'pre-combat Charge': 'Charge pre-combat attack',
    'pre-combat Sudden Strike': 'Sudden Strike pre-combat attack',
    "pre-combat Durin's Bane": "Durin's Bane pre-combat attack",
    'post-casualty We Come to Kill': 'We Come to Kill attack',
    'post-casualty Onslaught': 'Onslaught attack',
}


def _side(value):
    return getattr(value, 'name', str(value)) if value is not None else None


def description(call, human_side=None):
    """Describe an existing d6 request without changing its protocol identity."""
    if call.kind != 'd6':
        raise ValueError('A roll description requires a d6 request')
    side = _side(call.kw.get('side'))
    who = {'FREE_PEOPLES': 'Free Peoples', 'SHADOW': 'Shadow'}.get(side, 'Client')
    if side in ('FREE_PEOPLES', 'SHADOW'):
        who += ' (you)' if side == _side(human_side) else ' (AI)'
    role = call.kw.get('role')
    if role in ('attacker', 'defender'):
        who += ', ' + role
    kind = _KINDS.get(call.kw.get('ctx'), 'dice roll')
    n = int(call.kw.get('need', 0))
    if n < 1:
        raise ValueError('A pending roll must request at least one die')
    return '%s - %s: %d %s' % (who, kind, n, 'die' if n == 1 else 'dice')
