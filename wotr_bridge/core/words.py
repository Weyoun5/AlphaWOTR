"""The sentences the bridge says.  Plain words, no internal names.

Everything a person reads comes from here or from the engine's own action
describer (engine_play.describe_action): status sentences, decision titles,
option labels, the chat aliases a person may type, the blinding refusal.
"""
import functools
import re
import types
import unicodedata
from . import typed

_SCRUB = re.compile(r'[{}\[\]<>]')
_NAME_BAD = re.compile(r"[^a-z0-9 ,'!\-:]")
_DIE_PREFIX = re.compile(r'^\[die \d+:.\] ')


def _prose(text):
    return _SCRUB.sub('', str(text or '')).strip()


def chat_card_name(name):
    """A card name as chat may say it: ASCII, lower case."""
    s = unicodedata.normalize('NFKD', str(name or ''))
    s = s.encode('ascii', 'ignore').decode('ascii').lower()
    return re.sub(r'\s+', ' ', _NAME_BAD.sub('', s)).strip()


def combat_effect_choice_note(params):
    """The magnitude or target of a combat card decision the board cannot show."""
    from cards import get_card
    from combat import combat_effect_kind
    card = get_card(str(params.get('card_id', '')))
    if card is None:
        raise ValueError('combat decision has no recognized card')
    title = str(card.combat_name).strip()
    effect = combat_effect_kind(title) or title.casefold().replace(' ', '_')
    target, count = params.get('target'), params.get('count')
    if effect == 'dread_despair':
        return 'Dread and Despair: forfeit %d Nazgul leadership' % int(count or 0)
    if count == 0 and target is None:
        return '%s: decline' % title
    if effect == 'heroic_death' and target is not None and count is not None:
        return 'Heroic Death: sacrifice %s; cancel %d hits' % (_prose(target), int(count))
    if effect in ('black_breath', 'words_of_power') and target is not None:
        return '%s: target %s' % (title, _prose(target))
    if effect == 'relentless_assault' and count is not None:
        return 'Relentless Assault: take %d hits' % int(count)
    if effect == 'onslaught' and count is not None:
        return 'Onslaught: roll %d extra dice' % int(count)
    if target is not None:
        detail = _prose(target) + ('' if count is None else ' (count %d)' % int(count))
        return '%s: i choose %s' % (_prose(title), detail)
    return '%s: i choose %d' % (title, int(count or 0))


def table_discard_cancel_note():
    return 'I am discarding my table card to cancel the effect.'


def side_name(side):
    name = getattr(side, 'name', str(side))
    return 'Free Peoples' if name == 'FREE_PEOPLES' else 'Shadow'


_PLACES = None


def _places():
    """Engine region key -> the name the map itself shows ("helm s deep" ->
    "Helm's Deep"): the client's own location names, for the board's regions."""
    global _PLACES
    if _PLACES is None:
        _PLACES = {}
        try:
            import board
            import log_replay as lr
            regions = {r.key for r in board.load_board().regions}
            for table in (lr._AUTH_CODES, lr.LOCATION_CODES):
                for name in table.values():
                    if board.sanitize(str(name)) in regions:
                        _PLACES[board.sanitize(str(name))] = str(name)
        except Exception:
            pass
    return _PLACES


def place(key):
    """A region key as the map names it ('helm s deep' -> "Helm's Deep"); any
    other word with capitals ('elves' -> 'Elves')."""
    small = ('of', 'the', 'and')
    text = ' '.join(str(key or '').replace('_', ' ').split())
    known = _places().get(text.lower())
    if known is None and text and _PLACES:
        import board                    # the engine's own spelling ("Barad-Dûr") too
        known = _PLACES.get(board.sanitize(text))
    if known:
        return known
    words = text.split()
    return ' '.join(w if (i and w in small) else w[:1].upper() + w[1:]
                    for i, w in enumerate(words))


def card_title(cid):
    """The printed title of a card id (or of a card step the engine names)."""
    try:
        from cards import get_card
        return str(get_card(str(cid)).card_name)
    except Exception:
        text = str(cid or '')
        if text.startswith(('FREE_PEOPLES_', 'SHADOW_')):
            text = text.split('_', 3 if text.startswith('FREE') else 2)[-1]
        text = text.strip('_').replace('_', ' ').strip()
        return text[:1].upper() + text[1:]


_SPLIT = re.compile(r'(\d+)r/(\d+)e')
_LEADERS = re.compile(r'\+(\d+) leader\(s\)')
_INTERNAL = re.compile(r'[_=\[\]{}<>]|->')


def _split_words(m):
    parts = ['%s %s' % (n, word) for n, word in ((m.group(1), 'Regular'), (m.group(2), 'Elite'))
             if n != '0']
    return ', '.join(parts) or 'no units'


def plain(text):
    """A sentence without internal notation: no identifiers, brackets or
    key=value pairs.  Everything a person reads passes through here."""
    text = _SPLIT.sub(_split_words, str(text or '')).replace('+leaders', 'with the Leaders')
    text = _LEADERS.sub(r'with \1 Leader(s)', text).replace(' -> ', ' to ')
    text = text.replace('->', ' to ').replace('=', ' ').replace('_', ' ')
    return ' '.join(_SCRUB.sub('', text).split())


def is_plain(text):
    return _INTERNAL.search(str(text or '')) is None


def _names(value):
    value = [value] if isinstance(value, str) else list(value or ())
    return ' and '.join(str(v) for v in value)


_KEYS = {'include_wk': 'with the Witch-king', 'unit_type': 'unit', 'dice_count': 'dice',
         'dice_faces': 'faces', 'casualty_operation': 'casualty', 'target_region': 'in'}
_A_UNIT = {'regular': 'a Regular', 'elite': 'an Elite', 'leader': 'a Leader', 'nazgul': 'a Nazgul'}
_CASUALTY = {'remove_regular': 'Remove a %s Regular', 'reduce_elite': 'Reduce a %s Elite to a Regular',
             'remove_elite': 'Remove a %s Elite'}


def _nation(value):
    return place(str(getattr(value, 'value', value)))


def _choice(value):
    """The usual shapes of a card's choice, as a person would say them."""
    keys = set(value)
    unit = value.get('kind') or value.get('unit_type')
    where = value.get('region') or value.get('target_region')
    if 'casualty_operation' in value and 'nation' in value:
        text = _CASUALTY.get(str(value['casualty_operation']), 'Hit a %s unit') % _nation(value['nation'])
        return text[:1].lower() + text[1:] + (' in %s' % place(where) if where else '')
    if unit and 'nation' in value and where and len(keys) == 3:
        return 'the %s %s in %s' % (_nation(value['nation']), place(str(unit)), place(where))
    if keys == {'unit_type'}:
        return _A_UNIT.get(str(unit).lower(), place(str(unit)))
    if keys == {'nation'}:
        return _nation(value['nation'])
    if keys == {'region'}:
        return place(value['region'])
    if keys == {'regions'} or keys == {'settlements'}:
        return ' and '.join(place(r) for r in list(value.values())[0]) or 'none'
    if 'dice_count' in keys and keys <= {'dice_count', 'dice_faces'}:
        n = int(value['dice_count'] or 0)
        faces = ', '.join(place(str(f).lower()) for f in value.get('dice_faces') or ())
        return 'no dice' if not n else '%d %s: %s' % (n, 'die' if n == 1 else 'dice', faces)
    return None


def _value(value):
    """A parameter of an option in words."""
    if isinstance(value, dict):
        value = {k: v for k, v in value.items() if not str(k).startswith('_') and v is not None}
        if len(value) == 1 and isinstance(list(value.values())[0], dict):
            key, inner = list(value.items())[0]
            text = _value(inner)                        # one named group: its content
            return 'remove ' + text if key == 'eliminate' else text
        known = _choice(value)
        if known is not None:
            return known
        frm = value.pop('from_region', None) or value.pop('from', None)
        to = value.pop('to_region', None) or value.pop('to', None)
        head = ''
        if frm and to:
            who = _names(value.pop('companions', None)) or (
                '%s Nazgul' % value.pop('count') if 'count' in value else
                'the Witch-king' if value.pop('include_wk', None) else 'the pieces')
            head = 'move %s from %s to %s' % (who, place(frm), place(to))
        elif frm or to:
            value['from' if frm else 'to'] = frm or to
        value = {_KEYS.get(k, str(k).replace('_region', '').replace('_', ' ')): v
                 for k, v in value.items() if v is not False}
        rest = ', '.join(k if v is True else '%s %s' % (k, _value(v))
                         for k, v in sorted(value.items(), key=str))
        return ', '.join(t for t in (head, rest) if t) or 'nothing'
    if isinstance(value, (list, tuple, set, frozenset)):
        return ' and '.join(_value(v) for v in value) or 'none'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    value = getattr(value, 'value', value)
    text = str(value)
    if text.startswith(('FREE_PEOPLES_', 'SHADOW_')) and text.count('_') >= 2:
        return card_title(text)
    return place(text.lower() if text.isupper() else text) if isinstance(value, str) else text


_FIXED = {
    'pass': 'Pass', 'combat_card_none': 'No combat card', 'combat_continue': 'Continue the battle',
    'combat_cease': 'Stop the attack', 'combat_no_retreat': 'Stand and fight',
    'retreat_into_siege': 'Withdraw into the stronghold', 'siege_press': 'Press the siege',
    'siege_no_press': 'Do not press the siege', 'siege_assault': 'Assault the stronghold',
    'victor_stay': 'Stay where you are', 'followup_done': 'Done',
    'fellowship_move': 'Move the Fellowship (the Shadow hunts for it)',
    'fellowship_hide': 'Hide the Fellowship',
    'hunt_take_corruption': 'Take the Hunt damage as Corruption',
    'hunt_sacrifice_guide': 'The Guide takes the Hunt damage and leaves the game',
    'hunt_sacrifice_random': 'A random Companion takes the Hunt damage and leaves the game',
    'shine': 'Gandalf the White gives up his Leadership to cancel the Nazgul Leadership',
    'shine_none': 'Gandalf the White keeps his Leadership', 'rearguard_none': 'No rearguard',
    'crown_aragorn': 'Crown Aragorn',
}


def inner_step(record):
    """Is this card record (an option's parameters, an open card choice) no card at all?  The
    engine asks a move in several steps like a card's choice, under an inner name."""
    return any(str(record.get(key) or '').startswith('__') for key in ('card_id', 'card_name'))


def _own_label(state, side, action):
    """The options the engine's describer words in its own notation."""
    kind, p = action.action_type, action.params or {}
    to = place(p.get('to_region') or p.get('to') or '')
    frm = place(p.get('from_region') or p.get('from') or '')
    if kind == 'card_option':
        import engine_play
        choices = _value(dict(p.get('choices') or {}))
        if inner_step(p):
            return 'Move no more' if choices == 'nothing' else choices[:1].upper() + choices[1:]
        return '%s: %s' % (card_title(engine_play._card_name(p)),
                           'resolve it' if choices == 'nothing' else choices)
    if kind == 'combat_effect_choice':
        if p.get('operation') in ('remove', 'downgrade'):
            # the unit that pays for the combat card: worded like a casualty
            return _CASUALTY['reduce_elite' if p['operation'] == 'downgrade' else
                             'remove_%s' % p.get('kind')] % place(str(p.get('nation', '')).lower())
        try:
            return combat_effect_choice_note(p)
        except Exception:
            return None
    if kind == 'elven_ring':
        try:
            old = state.dice[side][int(p.get('target_die_idx'))].current.name
        except Exception:
            old = 'chosen'
        return 'Use an Elven Ring: change the %s die to %s' % (
            place(old.lower()), place(str(getattr(p.get('new_face'), 'name',
                                                  p.get('new_face'))).lower()))
    if kind in ('dwarven_ring', 'siege_trim'):
        return '%s: %s' % ('Use a Dwarven Ring' if kind == 'dwarven_ring'
                           else 'Keep five units in the stronghold', _value(dict(p)))
    if kind == 'eye_allocate':
        n = int(p.get('count') or 0)
        return 'Put %s in the Hunt Box' % ('no dice' if not n else '1 die' if n == 1
                                          else '%d dice' % n)
    if kind == 'nazgul_move' and frm:
        n = p.get('count')              # (no count: every Nazgul of the region)
        who = ['%s Nazgul' % n] if n else [] if n == 0 else ['all the Nazgul']
        who += ['the Witch-king'] if p.get('include_wk') else []
        return 'Move %s from %s to %s' % (' and '.join(who) or 'no Nazgul', frm, to)
    if kind in ('companion_move', 'companion_separate'):
        who = _names(p.get('companions') or p.get('companion')) or 'Companions'
        if kind == 'companion_separate':
            return 'Separate %s from the Fellowship to %s' % (who, to)
        return 'Move %s%s to %s' % (who, ' from %s' % frm if frm else '', to)
    if kind == 'combat_retreat' and to:
        return 'Retreat to %s' % to
    region = place(p.get('region') or '')
    nation = _nation(p['nation']) if p.get('nation') is not None else ''
    if kind in ('play_event', 'discard_card', 'discard_table_card', 'combat_card_play'):
        import engine_play
        return {'play_event': 'Play the card %s', 'discard_card': 'Discard the card %s',
                'discard_table_card': 'Discard the table card %s',
                'combat_card_play': 'Play the combat card %s'}[kind] % engine_play._card_name(p)
    if kind == 'muster_recruit':
        unit = str(p.get('type') or 'unit').lower()
        if p.get('placements'):         # the Voice of Saruman: one in every Isengard settlement
            return 'Recruit %s in each of %s (Saruman)' % (
                _A_UNIT.get(unit, 'a ' + place(unit)), ', '.join(place(r) for r in p['placements']))
        return 'Recruit %s in %s' % (_A_UNIT.get(unit, 'a ' + place(unit)), region)
    if kind == 'muster_elite':
        if p.get('saruman'):            # ... or two Regulars of Isengard become Elites
            return 'Upgrade two Regulars to Elites in %s (Saruman)' % region
        return 'Upgrade a Regular to an Elite in %s' % region
    if kind == 'muster_minion':
        return 'Bring %s into play at %s' % (p.get('minion'), region)
    if kind == 'draw_card':
        return 'Draw a %s card' % place(str(p.get('deck') or 'Event'))
    if kind == 'advance_political':
        return 'Move %s one step on the political track%s' % (
            nation, ' (the ability of %s)' % p['companion_ability'] if p.get('companion_ability') else '')
    if kind == 'choose_guide':
        return 'Make %s the Guide' % p.get('companion')
    if kind == 'casualty_allocate' and str(p.get('operation')) in _CASUALTY:
        return _CASUALTY[str(p['operation'])] % nation
    if kind == 'retreat_into_siege':
        import ai
        import game_logic as gl
        battle = getattr(state, 'pending_combat', None) or {}
        army = ai._armies_lookup(state, battle['defender_region']) if battle else None
        if len(getattr(army, 'units', None) or ()) > gl.MAX_SIEGE_UNITS:
            # (all of them go into the box; who leaves is asked only once the siege stands)
            return ('Withdraw into the stronghold with all your units (if the siege begins I '
                    'will ask which of them go back to the reinforcements)')
    return _FIXED.get(kind)


def option_label(state, side, action, viewer):
    """One legal option in plain words: the engine's own describer (which
    hides what `viewer` may not see), without the die (the panel shows the
    dice), with places named and no internal notation."""
    import engine_play
    p = action.params or {}
    text = None
    if engine_play._masked_private_body(state, side, action, viewer) is None:
        text = _own_label(state, side, action)
    if text is None:
        text = _DIE_PREFIX.sub('', engine_play.describe_action(state, side, action, viewer=viewer))
        for key in ('from', 'to', 'region', 'to_region', 'from_region'):
            value = p.get(key)
            if isinstance(value, str) and value and value in text:
                text = text.replace(value, place(value))
    text = plain(text)
    # ONE CHOICE, ONE NAME (ruling 9): what tells two rows of a menu apart is in the label
    text = text.replace(' nazgul', ' Nazgul').replace(' (Mouth: muster die as army)', '')
    if p.get('n_nazgul') == 0 and 'n_nazgul' in p:      # the Leaders go, the Nazgul stay
        text = text.replace('with the Leaders', 'with the Leaders but no Nazgul') \
            if 'with the Leaders' in text else text + ' (no Nazgul)'
    if p.get('mouth_muster'):                           # a Muster die spent as an Army die
        text += ' (a Muster die, by the Mouth of Sauron)'
    return text


def units_words(params):
    """The units of an army move row ('2 Regular, 1 Elite, with the Leaders')."""
    parts = []
    for key, word in (('n_regular', 'Regular'), ('n_elite', 'Elite'), ('n_nazgul', 'Nazgul')):
        if params.get(key):
            parts.append('%s %s' % (params[key], word))
    if params.get('take_leaders'):
        figures = ', '.join(str(r[1]) for r in params.get('leader_set') or ())
        parts.append('with %s' % (figures or 'the Leaders'))
    return ', '.join(parts) or 'the army'


# kind of an open decision -> (title, what is asked, how to answer it on the board, what to type)
DECISIONS = {
    'action': ('Your action', 'use one of your action dice',
               'spend the die and move your pieces',
               'an attack ("dale from erebor") or "pass"'),
    'declare': ('The Fellowship phase', 'declare where the Fellowship is, or go on',
                'move the Fellowship figure to its region', '"fp ok" to go on without declaring'),
    'eye': ('Hunt allocation', 'put dice into the Hunt Box before the roll',
            'drag the dice into the Hunt Box', '"0e" for no dice'),
    'discard': ('Hand limit', 'you hold more than six cards, discard one',
                'discard a card from your hand', ''),
    'guide_change': ('The Guide', 'choose the Guide of the Fellowship',
                     'put the Companion into the Guide box', '"fp ok" to keep the one who sits there'),
    'hunt': ('The Hunt', 'the Hunt found the Fellowship, choose how to take the damage',
             'move the Corruption marker or remove the Companion', ''),
    'hunt:hunt_gollum': ('The Hunt', 'Gollum is the Guide: he may reveal the Fellowship to '
                         'take one damage less', '', ''),
    'hunt:hobbit_place': ('The Hunt', 'a hobbit leaves the Fellowship, choose where he goes',
                          'move him to the region', ''),
    'fellowship_tail': ('The Fellowship phase', 'answer the effect that is open', '', ''),
    'reveal_place': ('The Fellowship is revealed', 'choose where the Fellowship is found',
                     'move the Fellowship figure there', ''),
    'card_option': ('Card effect', 'choose how the card takes effect',
                    'do with the pieces what the card says', ''),
    'card_followup': ('Card effect', 'take the next step of the card, or end it',
                      'do the next step with the pieces', '"done" to end it'),
    'combat:attacker_combat_card': ('Combat card', 'play a combat card face down, or none',
                                    'pick the card with the client\'s card chooser',
                                    '"nc" for no card, or "cc" and the card\'s name'),
    'combat:defender_combat_card': ('Combat card', 'play a combat card, or none',
                                    'pick the card with the client\'s card chooser',
                                    '"nc" for no card, or "cc" and the card\'s name'),
    'combat:combat_effect_choice': ('Combat card effect', 'choose how your combat card takes effect',
                                    '', ''),
    'combat:casualty_allocation': ('Casualties', 'choose which of your pieces take the hits',
                                   'remove or reduce your units', ''),
    'combat:attacker_casualty_policy': ('Casualties', 'choose which units you would rather keep', '', ''),
    'combat:defender_casualty_policy': ('Casualties', 'choose which units you would rather keep', '', ''),
    'combat:attacker_continue': ('After the round', 'continue the battle or stop the attack',
                                 '', '"continue" or "stop"'),
    'combat:combat_round_result': ('The siege', 'press the siege (an Elite is reduced) or stop',
                                   '', '"press" or "stop"'),
    'combat:attacker_siege_decision': ('The siege', 'assault the stronghold', '', '"press"'),
    'combat:defender_siege_choice': ('Field or siege', 'fight in the field or withdraw into the stronghold',
                                     'move your units into the stronghold box',
                                     '"field" to fight in the field'),
    'combat:defender_retreat': ('Retreat', 'retreat to a free region or stand',
                                'move the army to the region', '"field" to stand'),
    'combat:attacker_card_retreat': ('Retreat', 'your combat card lets you retreat: retreat or stand',
                                     'move the army to the region', '"field" to stand'),
    'combat:victor_advance': ('Advance', 'advance into the conquered region or stay',
                              'move the units in', '"stand" to stay'),
    'combat:siege_trim': ('Too many units', 'choose the five units that stay in the stronghold',
                          'remove the others', ''),
    'combat:fp_shine': ('Gandalf the White', 'decide whether Gandalf the White cancels the Nazgul '
                        'Leadership in this battle', '', ''),
    'combat:attacker_rearguard': ('The attack', 'confirm the attack', '', ''),
}


def _row(kind, state):
    """The DECISIONS row of an open decision: for the Hunt the row of the question it asks."""
    asked = '%s:%s' % (kind, (getattr(state, 'pending_hunt', None) or {}).get('phase'))
    return asked if asked in DECISIONS else kind


def decision_words(kind, state):
    """(title, what is asked) of the decision of `kind` that is open in `state`."""
    kind = _row(kind, state)
    if kind in DECISIONS:
        title, what = DECISIONS[kind][:2]
    else:
        title = what = kind.split(':')[-1].replace('_', ' ')
        what = 'answer the %s%s' % ('battle step: ' if kind.startswith('combat') else 'step: ', what)
    return title[:1].upper() + title[1:], what[:1].upper() + what[1:] + '.'


def family(kind):
    """The panel tab a decision belongs to."""
    if kind.startswith('combat'):
        return 'combat'
    if kind in ('hunt', 'reveal_place', 'guide_change'):
        return 'hunt'
    return 'card' if kind.startswith('card_') else 'action'


# chat words a person may type for the answers the board cannot show: word -> the option
# types it names (said_type)
ALIASES = {
    'nc': ('combat_card_none',),
    'pass': ('pass',),
    'fp ok': ('pass',),
    'done': ('followup_done',),
    'field': ('combat_no_retreat',),
    'siege': ('retreat_into_siege',),
    'continue': ('combat_continue', 'siege_press', 'siege_assault'),
    'press': ('siege_press', 'siege_assault', 'combat_continue'),
    'stop': ('combat_cease', 'siege_no_press'),
    'stand': ('victor_stay', 'combat_no_retreat'),
}
# a word that belongs to ONE part of the turn is a word only in its decisions: "fp ok" closes the
# Fellowship phase - it goes on without declaring, it keeps the Guide who sits in the Guide box
# (ends) - and typed anywhere else it answers nothing (it never passes an action)
ONLY_IN = {'fp ok': ('declare', 'guide_change')}
# The answers that leave NO trace on the board: only these may be committed by a chat
# word (owner ruling: the chat is only for what cannot be seen from the pieces).  Every
# other option is answered on the board (M3) or by a click; the panel is free for all.
UNSEEN = frozenset((
    'pass', 'declare_decline', 'combat_card_none', 'combat_card_play', 'army_attack',
    'combat_continue', 'combat_cease', 'siege_press', 'siege_no_press', 'siege_assault',
    'combat_no_retreat', 'victor_stay', 'followup_done', 'shine', 'shine_none',
    'rearguard_none', 'combat_effect_choice', 'casualty_preserve_units',
    'casualty_preserve_elites', 'hunt_gollum_no_reveal', 'end_turn'))
EARLY = {'nc': 'card', 'press': 'round', 'stop': 'round', 'continue': 'round'}
_ROUND_KINDS = ('combat:attacker_continue', 'combat:combat_round_result',
                'combat:attacker_siege_decision')


def said_type(action):
    """The option type the word tables (ALIASES, UNSEEN) know an option by.  The engine ends a
    follow-up with an action of its own, a chain of steps with a card option that holds only
    a done mark: both are the same answer, "done", and neither moves a piece."""
    choices = (action.params or {}).get('choices') or {}
    ends = action.action_type == 'card_option' and any(str(k).endswith('_done') for k in choices) \
        and all(str(k).startswith('_') for k in choices)
    return 'followup_done' if ends else action.action_type


def by_word(action):
    """May a chat word commit this option?  (Its answer cannot be seen on the board.)"""
    if action.action_type == 'eye_allocate':
        return not int((action.params or {}).get('count') or 0)
    return said_type(action) in UNSEEN


def is_word(text):
    """Is this chat line one of the words (a fixed word, an attack, a card, the Eyes), not talk?"""
    return text in ALIASES or text.startswith(('attack ', 'cc ')) or bool(EYES.match(text))


def word_fits(word, kind):
    """Is this fixed word a word of a decision of `kind`?"""
    return word in ALIASES and kind in ONLY_IN.get(word, (kind,))


def ends(text, kind):
    """Is this the closing word of the part of the turn a decision of `kind` belongs to?  Where
    it names no option of that decision, it closes it as the board stands."""
    return kind in ONLY_IN.get(chat_key(text), ())


def chat_key(text):
    """A chat line as it is compared: ASCII, lower case, no punctuation."""
    text = chat_card_name(text).replace("'", '').replace('-', ' ')
    return ' '.join(re.sub(r'[^a-z0-9 ]', ' ', text).split())


def early_kind(text, battle, side):
    """The question of the open `battle` that a word of `side` may be given for in advance
    ('card', 'round'), or None.  A word waits only for the question it was typed for: none
    with no battle open, and no "nc" / "cc" once his card of this round is in."""
    early = 'card' if text.startswith('cc ') else EARLY.get(text)
    if early is None or not isinstance(battle, dict):
        return None
    role = 'atk' if battle.get('attacker_side') is side else 'def'
    return None if early == 'card' and role + '_combat_card' in battle else early


def early_fits(early, kind):
    return kind.endswith('combat_card') if early == 'card' else kind in _ROUND_KINDS


def _card_names(action):
    names = {chat_key(card_title((action.params or {}).get('card_id')))}
    try:
        from cards import get_card
        names.add(chat_key(get_card(str(action.params['card_id'])).combat_name))
    except Exception:
        pass
    return names - {''}


def spoken(state, side, action):
    """What a person may type in chat to name this option: its label."""
    return {chat_key(option_label(state, side, action, side))}


def word_for(state, side, action):
    """The shortest thing to type for an option a word may commit."""
    kind, p = action.action_type, action.params or {}
    if kind == 'army_attack':
        return '%s from %s' % (region_key(p.get('to')), region_key(p.get('from')))
    if kind == 'combat_card_play':
        return 'cc %s' % chat_key(card_title(p.get('card_id')))
    return next((w for w in ('nc', 'pass', 'field', 'siege', 'press', 'stop', 'stand', 'done')
                 if said_type(action) in ALIASES[w]), None)


EYES = re.compile(r'^(\d+)e$')          # "0e", "2e": that many dice into the Hunt Box


def _face(state, side, action):
    """The face letter of the die an option spends (None: it spends none)."""
    from .emitter import _FACE_LETTER
    try:
        die = state.dice[side][action.die_idx] if action.die_idx >= 0 else None
        return _FACE_LETTER.get(die.current.name)
    except Exception:
        return None


def region_key(key):        # a region as it is typed in chat
    return chat_key(place(key))


def heard(text, state, side, menu, staged, kind):
    """A chat line of the person against the menu of his open decision (of
    `kind`) -> (verdict, action): 'pick' it names exactly one option; 'several' / 'none'
    it is one of the words (a fixed word, an attack, a combat card, a label)
    and fits more than one option / none; (None, None) ordinary talk.
    A fixed word is heard only in the decisions it belongs to (word_fits).
    A typed move names no die: when dice of several faces can make it, the one
    he staged decides (staged: the face letters of his dice in the play area);
    none or several of them staged -> ('dice', the face letters that can).
    A combat card's name that several of his cards carry -> ('cards', None)."""
    t = chat_key(text)
    hits, word = [], is_word(t)
    if word_fits(t, kind):
        hits = [a for a in menu if said_type(a) in ALIASES[t]]
    if EYES.match(t):
        hits = [a for a in menu if a.action_type == 'eye_allocate'
                and int((a.params or {}).get('count') or 0) == int(t[:-1])]
    if not hits and t not in ALIASES:       # (a fixed word is never read as a region's short form)
        hits, half = typed.attacks(t, menu, region_key)
        word = word or half
    if not hits:
        name = t[3:] if t.startswith('cc ') else t
        hits = [a for a in menu if a.action_type == 'combat_card_play' and name in _card_names(a)]
        if len({str(a.params.get('card_id')) for a in hits}) > 1:
            return 'cards', None
    if not hits:
        hits = [a for a in menu if t in spoken(state, side, a)]
    faces = sorted({_face(state, side, a) for a in hits} - {None})
    if len(faces) > 1 and by_word(hits[0]):     # (which die matters only where a word commits)
        on = [f for f in faces if f in staged]
        if len(on) != 1:
            return 'dice', faces
        hits = [a for a in hits if _face(state, side, a) == on[0]]
    keys = {(a.action_type, repr(sorted((a.params or {}).items(), key=str))) for a in hits}
    if len(keys) == 1:
        return 'pick', hits[0]
    return ('several' if keys else 'none' if word else None), None


def which_die(faces):
    """ONE sentence for a typed move that dice of several faces can make while the
    die he staged does not say which."""
    from .menu import FACE_WORDS
    names = ['the %s die' % FACE_WORDS.get(f, f) for f in faces]
    return ('You can make that move with %s or %s: put the die you mean into the Current Action '
            'box and type it again, or click the move in the panel.' % (', '.join(names[:-1]), names[-1]))


_GONE = ('That is no longer a word here: type "back" and I put the board back as the game has it, '
         'or "undo" to take back your last decision.')
_OLD = {
    'help': 'You can type "back" (I put the board back), "undo" (take back your last decision), "finish" '
            '(I do what I left to your own hand), "pass", "fp ok", "done", "nc" or "cc" and a card name for a combat card, "field", '
            '"stand", "press", "continue", "stop", "0e" for no Eyes, an attack such as "dale from '
            'erebor" (short forms and a combat card with it work too: "osg from si nc"), '
            'and the name of a gate when I ask; everything else is a move on the board '
            'or a click in the panel.',
    'skip': 'There is no skip any more: answer with a move on the board or a click in the panel, '
            'or type "back" and I put the board back as the game has it.',
    'reset': _GONE, 'rewind': _GONE, 'manual': _GONE, 'manual edit': _GONE,
    'yes': 'Nothing here is answered with yes or no: answer on the board or in the panel, and '
           'type "help" for the words I know.',
    'random': 'To let a random Companion take the Hunt damage, click that option in the panel.',
    'unpause': 'The chat cannot pause or resume the game: if you paused it in the old AI window, '
               'press RESUME there.',
}
_OLD.update(no=_OLD['yes'], pause=_OLD['unpause'], resume=_OLD['unpause'])


# the panel's levers this bridge does not have: (label, why not and what to do instead)
LEVERS_OFF = {
    'skip': ('Skip decision', 'Not in this version: answer with one of the options, or use '
                              '"Put the board back".'),
    'reset_resync': ('Reset and resync', 'Not in this version: use "Put the board back", or '
                                         '"Take back my last decision".'),
    'resync': ('Check the board', 'Use "Put the board back": it compares the board with the '
                                  'game and repairs it.'),
    'set_children': ('', 'The search width cannot be changed in this version: only the AI '
                         'strength can.'),
    'set_chance': ('', 'The chance outcomes per roll cannot be changed in this version: only '
                       'the AI strength can.'),
    'enforce': ('', 'Your moves are always checked for legality in this version.'),
    'whose_turn': ('', 'Whose turn it is cannot be overridden in this version: the game keeps '
                       'the order itself.'),
}


def lever(op, sims):
    """ONE sentence answering a click on a lever of the panel."""
    if op == 'set_sims':
        return 'The AI now plays with %d simulations per move.' % sims
    return LEVERS_OFF.get(op, ('', 'That lever does nothing in this version.'))[1]


def old_word(text):
    """ONE sentence for a word the old bridge knew and this one does not (what to do
    instead), for "help" the words that exist; None for anything else (talk)."""
    t = chat_key(text)
    if t.isdigit():
        return 'A number alone answers nothing here: click the option in the panel.'
    return _OLD.get(t)


def say_choices(state, side, menu, verdict):
    """ONE sentence for a word that picked no option, by the reader's verdict (heard): 'none'
    it fits no option, 'several' more than one, 'cards' several of his cards carry that combat
    name.  It says what can be typed here, and repeats neither his word nor a card of his hand."""
    if verdict == 'cards':
        return ('More than one of your cards carries that name: type "cc" and the title at '
                'the top of the card, or click the card in the panel.')
    names, several = [], verdict == 'several'
    for action in menu:
        name = None
        if action.action_type == 'combat_card_play':
            name = '"cc" and the name of the card'
        elif by_word(action):
            name = '"%s"' % word_for(state, side, action)
        if name and name not in names and name != '"None"':
            names.append(name)
    head = 'That %s' % ('fits more than one option here' if several else 'does not fit here')
    if not names:
        return head + ': this choice is made on the board or with a click in the panel.'
    more = ' and %d more' % (len(names) - 6) if len(names) > 6 else ''
    return '%s: you can type %s%s; everything else is a move on the board or a click in the panel.' % (
        head, ', '.join(names[:6]), more)


def not_by_word():
    return 'That is not answered by a chat word: do it on the board or click it in the panel.'


def word_kept(early):
    return 'Noted: it answers your %s when it is asked.' % (
        'combat card' if early == 'card' else 'choice after the round')


def nothing_asked():
    return 'That answers nothing now: no decision of yours is open.'


CLOSING = ('fp ok', 'pass', 'done', 'stop')     # words that close a decision (ruling 8)


def closes(text):
    return chat_key(text) in CLOSING


def confirm(kind, named_by_board):
    """Ruling 8: what a closing word does, by the matcher's verdict on the board of the
    open decision.  'board': it confirms the board's answer, which is committed at once
    (also inside its K / Q wait, a strict part included); 'word': the option without a
    move is committed (the board shows nothing; or it is unfinished or undecided and allows
    exactly that option, or shows nothing but a die or card laid down for an action: named
    by the board); 'say': nothing is committed and the sentence is said at once."""
    if kind == 'commit':
        return 'board'
    if kind in (None, 'nothing') or (kind in ('wait', 'undecided') and named_by_board):
        return 'word'
    return 'say'


def board_unclear(what, cannot=(), moved=None, more=False):
    """The board fits no legal move (what: the matcher's own words - the pieces that fit none,
    or what differs from the move he began; None: said here of what lies `moved` - more: each
    of those alone begins a legal move, together they are more moves than are owed), and the
    ways out.  cannot: the options of his decision that cannot be read from the board at all."""
    what = str(what or ('The board shows more moves than are owed here: %s.' if more
                        else 'No legal move moves: %s.') % moved).strip()
    return '%s %s "back" and I put the board back.%s' % (
        what[:1].upper() + what[1:] if what.endswith('.') else what + '.',
        'Type' if what.startswith('No legal move') else 'Take one of them back, or type' if more
        else 'Finish the move, or type',
        ' The board cannot answer "%s": use the word or the click.' % '"; "'.join(cannot) if cannot else '')


def tell_apart(names):
    """What to type for each of several choices: the words of its name that the others
    do not share ("minas morgul", "morannon"); the whole name when that leaves nothing."""
    keys = [chat_key(n).split() for n in names]
    a = b = 0
    while len(keys) > 1 and all(len(k) > a for k in keys) and len({k[a] for k in keys}) == 1:
        a += 1
    while len(keys) > 1 and all(len(k) - a > b for k in keys) and len({k[-1 - b] for k in keys}) == 1:
        b += 1
    own = [' '.join(k[a:len(k) - b]) for k in keys]
    return own if all(own) and len(set(own)) == len(own) else [' '.join(k) for k in keys]


def names_choice(text, names):
    """The index of the ONE choice a typed word names among choices the board cannot tell
    apart (its own words, or its whole name); None when it names none or several."""
    t, own = chat_key(text), tell_apart(names)
    hits = [i for i, n in enumerate(names) if t and t in (own[i], chat_key(n))]
    return hits[0] if len(hits) == 1 else None


def board_choices(names):
    """The board fits more than one move: ONE sentence that names them and says what to
    type (when a few words tell them apart) or click."""
    names = [n.rstrip('.') for n in names if n]
    own, more = tell_apart(names), len(names) - 6       # (the words names_choice hears)
    typed = len(names) > 1 and len(set(own)) == len(own) and all(len(t) <= 30 for t in own)
    return 'The board does not say which move you mean%s; %sclick it in the panel.' % (
        ': ' + ', or '.join(names[:6]) + (' and %d more' % more if more > 0 else '') if names else '',
        'type %s, or ' % ' or '.join('"%s"' % t for t in own[:6]) if typed else '')


def only_choice(label):
    return 'You had only one choice, so I made it for you: %s.' % label.rstrip('.')


def also_moved():
    return 'I also moved what the move brings with it for you.'


# why a piece the person moved cannot count and goes back (the reasons boardanswer.BoardAnswers.end
# is given that owe him a sentence): the ONE sentence of each reason, said by the pass of the
# comparison that carries the piece.  NEEDED: the game has it neither where it lies nor where he
# took it from any more (a step of the game took it); NOT_OWED: it lies in the play area and no
# action he still has can use it; BESIDE: he answered by a word or a click, and neither that
# answer nor his decision that follows at once uses what he had moved.  Each sentence NAMES what
# goes back (ruling J3)
NEEDED, NOT_OWED, BESIDE = 'needed', 'not owed', 'beside'
PUT_BACK = {
    NEEDED: 'A move of the game needed a piece you had moved, so I put it where the game has it: %s.',
    NOT_OWED: 'No move of yours can use what you laid down there, so I put it back: %s.',
    BESIDE: 'Your answer did not use everything you had moved, so I put the rest back: %s.',
}


def put_back(reason, what):
    """what: the pieces the pass carries back, in plain words (boardmodel.moved_words)."""
    return PUT_BACK[reason] % what


# the answers to "take back my last decision": done, or why it cannot be
TAKE_BACK = {
    'done': 'Your last decision was taken back. Choose again.',
    'nothing': 'Nothing can be taken back now.',
    'resolving': 'Your choice is still being resolved.',
    'moved on': 'The game has moved on since your last decision.',
    'rolled': 'Dice were rolled or cards drawn for it, so it stands.',
}


def take_back(answer):
    return TAKE_BACK[answer]


def not_carried_out(status=False):
    """His choice failed with an error: the reply to his click, or (status) the status line."""
    return 'Your choice could not be carried out. Please choose again.' if status else \
        'That could not be carried out. Please choose again, or choose another option.'


def put_back_asked(waits):
    """The answer to "back" at once; waits: a roll or a draw is under way, the board follows it."""
    return ('I put the board back as soon as the roll or the draw that is under way is done.'
            if waits else 'Comparing the board with the game now.')


def board_compared(n):
    """The answer to "put the board back": in how many places the board differed from the game."""
    return 'The board matches the game.' if not n else (
        'The board differed from the game in %d place%s; %s put back.'
        % (n, '' if n == 1 else 's', 'it is being' if n == 1 else 'they are being'))


def board_as_asked(moves):
    """The answer to "put the board back" while his decision is open; moves: pieces go back."""
    return 'I am putting the board back as it was when you were asked.' if moves else \
        'The board is as it was when you were asked.'


def putting_in_order():
    return 'Putting the board in order.'


def board_answer_failed(name):
    """An error of the bridge stopped the move he made on the board (name: the error's file):
    said as that - never as a rule of the game - with the two ways on."""
    return ('I could not carry out the move you made on the board: an error of mine, not a rule of the game '
            '(the details are in the file %s in the game folder). Click the move in the panel, or type '
            '"back" and I put the board back.' % name)


def another_game():
    return 'The client started another game. Close it and start the AI again.'


def waiting_for_you(kind, side, state):
    """What the game waits for, and how the person can answer it."""
    title, what = decision_words(kind, state)
    board, typed = (DECISIONS.get(_row(kind, state)) or ('', '', '', ''))[2:4]
    how = ['click an option in the panel']
    if board:
        how.insert(0, board + ' on the board')
    if typed:
        how.append('type ' + typed)
    return 'Waiting for you (%s): %s To answer: %s.' % (
        side_name(side), what, ', or '.join(how))


def _roll_for(ctx):
    ctx = str(ctx or '').lower()
    return (' for the Hunt' if 'hunt' in ctx else ' for the battle' if 'combat' in ctx
            or 'onslaught' in ctx else ' for the card' if 'card' in ctx else '')


def waiting_for_roll(n, ctx=''):
    return 'Waiting for you to roll %d %s%s with the dice button.' % (
        n, 'die' if n == 1 else 'dice', _roll_for(ctx))


def roll_request(side, n, ctx=''):
    """The one chat sentence asking the person for his dice."""
    return '%s (you): please roll %d %s%s.' % (side_name(side), n, 'die' if n == 1 else 'dice',
                                               _roll_for(ctx))


def by_hand(side, kind, decks=()):
    """His cards (decks: of each) or action dice, made with his own hand: the one chat sentence - (side None) the status."""
    what = 'roll your action dice' if kind == 'action_roll' else 'draw %s' % ' and '.join(
        '%d %s card%s' % (decks.count(d), d.capitalize(), '' if decks.count(d) == 1 else 's') for d in sorted(set(decks)))
    return 'Waiting for you to %s (or type "finish" and I do it for you).' % what if side is None else \
        '%s (you): please %s, or type "finish" and I do it for you.' % (side_name(side), what)


def left_to_you(what, status=False):
    """What a card of his own leaves for HIM to do (what: boardmodel.ops_words): the chat sentence, or the status."""
    return ('Waiting for you to do what your card says: %s (or type "finish" and I do it for you).' if status else
            'Your card leaves this for you to do on the board: %s - I go on when the board shows it, or type '
            '"finish" and I do it for you.') % what


def finishing(owed):
    """The answer to "finish": the bridge now makes what it had left to his own hand; or nothing was left."""
    return 'I do the rest for you.' if owed else 'Nothing is waiting for your own hand now: there is nothing to finish.'


def second_roll(n):
    """He rolled again right after the roll he was asked for: those dice wait for his next roll."""
    return 'You rolled again: I keep %s for your next roll.' % ('that die' if n == 1 else 'those %d dice' % n)


def unused_results(n):
    return ('I am ignoring %d result%s of a roll or a draw that the game does not use.'
            % (n, '' if n == 1 else 's'))


def card_not_playable(laid_down=False):
    """ONE sentence for a picked card that is no combat card here; laid_down: the client has
    put it on the table, and the server is putting it back into his hand."""
    return 'The card you chose cannot be played as a combat card here%s: please choose again.' % (
        ', so I am putting it back into your hand' if laid_down else ' and stays in your hand')


_CLIENT_DOES = {'d6': 'roll the dice', 'draws': 'draw the cards', 'tile': 'draw a Hunt tile',
                'action_roll': 'roll the action dice', 'randomcomp': 'draw a random Companion',
                'card_pick': 'discard a random card'}


def waiting_for_client(kind):
    return 'Waiting for the client to %s.' % _CLIENT_DOES.get(kind, 'answer')


def client_silent(kind):
    """The one sentence for a call the client carried out and gave no result for."""
    return ('I asked the game window to %s and no result has come: the game goes on as soon as one does, '
            'and when you type "retry" I ask again.' % _CLIENT_DOES.get(kind, 'answer'))


def waiting_for_board():
    return 'Waiting for the board of the client.'


def no_board():
    """Said after a time without a board that can be read: the next one that can is played on."""
    return ('I cannot read the board of the game window at the moment and play on as soon as I can; if it stays '
            'like this, please close the game window and start Play vs AI again.')


def internal_error(name):
    return ('I hit an internal error and keep trying. The details are in the file %s in the '
            'game folder.' % name)


_CARD_ID = re.compile(r'(FREE_PEOPLES|SHADOW)_(Character|Strategy)_\S*')


def no_cards(text):
    """A text for the console or an error file, which lie where the person may read them:
    no card is named in it (an id reads as its side and deck, a printed title as "a card")."""
    from cards import get_cards
    text = _CARD_ID.sub(lambda m: 'a %s %s card' % (side_name(m.group(1)), m.group(2)), str(text))
    for card in get_cards().values():
        text = text.replace(str(card.card_name), 'a card')
    return text


def ai_thinking(side, sims):
    return 'The %s is choosing its move (%s).' % (side_name(side), '%d simulations' % sims if sims else 'no search')


def ai_did(side, label):
    return 'The %s: %s.' % (side_name(side), label.rstrip('.'))


def you_did(label):
    return 'You: %s.' % label.rstrip('.')


def done(label):
    return 'Done: %s.' % label.rstrip('.')


def stale():
    return 'The position changed. Please choose again.'


def choice_busy():
    return 'Your previous choice is still being resolved.'


def chose():
    """Said in place of a sentence of the AI that would have named a card of his hand."""
    return 'I chose.'


def starting():
    return 'The game is starting.'


def blinding_violation(line, ai_side):
    """Why the feed may not be played on (the human's hand would be readable),
    or None.  ai_side: 'FREE_PEOPLES' / 'SHADOW', or None for AI vs AI."""
    if not isinstance(line, str) or not line.startswith(('#BLIND ', '#HAND')):
        return None
    human = None if ai_side is None else 'SA' if ai_side == 'FREE_PEOPLES' else 'FP'
    if line.startswith('#BLIND '):
        fields = dict(t.split('=', 1) for t in line.split()[1:] if '=' in t)
        if 'aiside' not in fields and 'hands' not in fields:
            return None
        hands = [h.strip().upper() for h in fields.get('hands', 'none').split(',')]
        if human is None:
            ok = fields.get('aiside', '').lower() == 'none' and hands == ['NONE']
        else:
            ok = human in hands
    else:
        toks = line.split()
        ok = toks[1:] == ['<blinded>'] or human is None or (
            len(toks) > 1 and toks[1].upper() != human)
    if ok:
        return None
    return ('The client was started for another kind of game: your hand would be readable '
            'by the AI. Nothing was played. Close the client and start it again from the '
            'launcher.')


def game_over(result):
    """One sentence for the end of the game."""
    who = 'Free Peoples' if str(result.get('victory', '')).startswith('free') else 'Shadow'
    how = {'ring_destroyed': 'the Ring was destroyed at Mount Doom',
           'ring': 'the Ring was destroyed at Mount Doom',
           'corruption': 'the Ring-bearers were corrupted',
           'ring_corruption': 'the Ring-bearers were corrupted',
           'military': 'a military victory'}.get(
               str(result.get('victory_type') or ''), _prose(result.get('victory_type') or ''))
    return 'Game over. Victory for the %s%s.' % (who, ': %s' % how if how else '')


def could_not_show(rows):
    """One sentence naming what the board still shows differently."""
    more = '' if len(rows) <= 2 else ' and %d more' % (len(rows) - 2)
    return ('I could not show this on the board and play on: %s%s.'
            % ('; '.join(rows[:2]), more))


# ---- what a player needs to follow the game: ONE plain line per fact, each built from
# ---- the engine's own state change (pre -> post of the committed step) ----
def _whose(side, human):
    return side_name(side) if human is None else 'you' if side is human else 'me'


def round_hits(pre, post, human):
    """The hits of the combat round fought in this step, from the engine's own result of
    that round (a state carries it only out of the step that fought it): what the attacker
    scored falls on the defender, and the other way round."""
    battle, result = pre.pending_combat, getattr(post, '_last_battle_result', None)
    if not isinstance(battle, dict) or result is None or getattr(result, 'round_paused', False):
        return None
    hits = [(battle.get('defender_side'), int(result.attacker_total_hits)),
            (battle.get('attacker_side'), int(result.defender_total_hits))]
    if not any(n for _side, n in hits):
        return 'This round: no hits.'
    if human is not None:
        hits.sort(key=lambda h: h[0] is not human)
    return 'This round: %s.' % ', '.join(
        '%d %s on %s' % (n, 'hit' if n == 1 else 'hits', _whose(s, human)) for s, n in hits)


def tile_drawn(face, tiles, damage, revealed):
    """A Hunt tile was drawn: what it is and what it does.  What depends on where the Fellowship
    is comes from the engine: the `damage` of the Hunt it opened (None: not known), `revealed`."""
    from .randomness import tile_face
    tile = next((t for t in tiles if tile_face(t) == tile_face(face)), None)
    if tile is None:
        return 'The Hunt tile is %s.' % place(face)
    reveal = ', and the Fellowship is revealed' if revealed else ''
    if getattr(tile, 'special_name', None):
        does = ('the damage is a die roll' if tile.is_dice_roll else
                'no damage' if not tile.damage else
                '%d damage' % tile.damage if tile.damage > 0 else
                '%d less Corruption' % -tile.damage)
        stop = ', and the Fellowship does not move' if getattr(tile, 'stops_movement', False) else ''
        return 'The Hunt tile is %s: %s%s%s.' % (tile.special_name, does, reveal, stop)
    if getattr(tile, 'is_eye', False):
        return 'The Hunt tile is an Eye%s%s.' % (
            '' if damage is None else ': %d damage' % damage, reveal if damage is not None
            else reveal.replace(', and', ':'))
    return 'The Hunt tile is a %d: %s%s.' % (
        tile.damage, '%d damage' % tile.damage if tile.damage else 'no damage', reveal)


def card_no_effect(title):
    return 'The card %s has no effect now.' % title


def _givers(state, side):
    names = (('gandalf_the_white', 'in_play', 'Gandalf the White'),
             ('aragorn_crowned', 'crowned', 'Aragorn')) if side.name == 'FREE_PEOPLES' else (
        ('witch_king', 'in_play', 'the Witch-king'), ('saruman', 'in_play', 'Saruman'),
        ('mouth_of_sauron', 'in_play', 'the Mouth of Sauron'))
    return [name for attr, flag, name in names
            if getattr(getattr(state, attr, None), flag, False)]


def dice_changed(pre, post, side, card, human):
    """A side has an action die more, or one fewer, and why."""
    before, after = len(pre.dice.get(side, ())), len(post.dice.get(side, ()))
    if before == after:
        return None
    n = abs(after - before)
    who = 'The %s now have' % side_name(side) if side.name == 'FREE_PEOPLES' else 'The Shadow now has'
    if human is not None:
        who = 'You now have' if side is human else 'I now have'
    if card:
        why = card
    else:
        now, was = _givers(post, side), _givers(pre, side)
        gone = [g for g in was if g not in now]
        why = ('%s left play' % ' and '.join(gone)) if after < before and gone else (
            '%s in play' % ' and '.join(now) if now else 'no character gives a die')
        why = why[:1].upper() + why[1:] if why.startswith('the ') else why
    return '%s %d action dice, %d %s than before (%s).' % (
        who, after, n, 'more' if after > before else 'fewer', why)


def sieges_ended(pre, post):
    out = ['The siege of %s has ended.' % place(r) for r in
           sorted(str(x) for x in set(pre.besieged_strongholds or ()) - set(post.besieged_strongholds or ()))]
    # the defender withdrew into the stronghold and the attacker did not follow: the engine
    # forms no siege, and the comparison takes the garrison out of the box again
    battle = pre.pending_combat if isinstance(pre.pending_combat, dict) else {}
    if battle.get('post_siege_retreat') and post.pending_combat is None \
            and set(post.besieged_strongholds or ()) == set(pre.besieged_strongholds or ()):
        out.append('The attacker did not advance: there is no siege, and the army stands in %s itself.'
                   % place(battle.get('defender_region')))
    return out


def enters_mordor(pre, post):
    if post.fellowship.mordor_entered and not pre.fellowship.mordor_entered:
        return 'The Fellowship enters Mordor.'
    return None


def gollum_guide(pre, post):
    def guide(state):
        return getattr(state.fellowship.guide, 'name', None)
    if guide(post) == 'Gollum' and guide(pre) != 'Gollum':
        return 'Gollum is now the Guide: no Companion is left in the Fellowship.'
    return None


def table_cards_gone(pre, post):
    def table(state):
        cards = getattr(state, 'persistent_cards', None)
        return {str(c.card_id): str(getattr(c, 'card_name', None) or card_title(c.card_id))
                for attr in ('fp_cards', 'sh_cards') for c in getattr(cards, attr, ())}
    now = table(post)
    return ['The card %s leaves the table.' % name
            for cid, name in sorted(table(pre).items()) if cid not in now]


def turn_over(turn):
    return 'Turn %d is over.' % turn


class Said(str):
    """A sentence that knows which function of this module made it (`id`: its name in the game's log)."""
    id = None


def _named(fn):
    @functools.wraps(fn)
    def made(*args, **kw):
        out = fn(*args, **kw)
        if type(out) is str:
            out = Said(out)
            out.id = fn.__name__
        return out
    return made


for _name, _fn in list(globals().items()):      # every sentence made here carries its name
    if isinstance(_fn, types.FunctionType) and not _name.startswith('_'):
        globals()[_name] = _named(_fn)
