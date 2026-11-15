"""The in-client panel's two files (schema l1104_panel_v1, read by the
unchanged AiPanel*.java):

    ai_panel_view.json      server -> panel: whose turn, the open decision with
                            EVERY legal option, the overview, replies
    ai_panel_request.json   panel -> server: one request slot (refresh /
                            preview / choose / recovery / perform_report)

Invariants:
  * every legal option of the open decision is in the view exactly once: as
    an option's `i`, as one of its dice choices, or in an `alts` list (dice
    of the same face, identical rows).  Nothing is capped or dropped;
  * an option is bound by `identity` (a hash of the action), so a click
    names the row the person read, whatever the menu order is by then;
  * a click is applied by the server and rendered by the server: the reply
    carries no steps for the panel to perform (DESIGN principle 5).
"""
import hashlib
import json
import os
import time
from collections import namedtuple

from . import words
from .transport import atomic_write

# what the view is built from: the game position, the seats, and what the loop says of itself
# (decision: the built `decision` of the person, or None; take_back: why his last decision
# cannot be taken back, or None)
Facts = namedtuple('Facts', 'pos human ai_sides sims net_tag session human_nick ai_nick decision '
                            'take_back how ai_last lever_said log_tail replies')
SCHEMA = 'l1104_panel_v1'
VIEW_NAME = 'ai_panel_view.json'
REPLIES_KEPT = 8
COMPOSER_MIN = 13               # smaller families stay plain buttons

# decide-tab groups in order: id, title, icon, action types
GROUPS = (
    ('g-quick', 'Quick answers', 'check', (
        'pass', 'declare_decline', 'combat_card_none', 'combat_continue', 'combat_cease',
        'siege_no_press', 'combat_no_retreat', 'victor_stay', 'rearguard_none', 'shine',
        'shine_none', 'followup_done', 'end_turn')),
    ('g-fellowship', 'Fellowship', 'ring', (
        'fellowship_move', 'fellowship_hide', 'fellowship_declare', 'fellowship_reveal_place',
        'choose_guide')),
    ('g-characters', 'Characters', 'crown', (
        'companion_separate', 'companion_move', 'transform_gandalf', 'crown_aragorn',
        'mouth_move', 'nazgul_move', 'muster_minion')),
    ('g-armies', 'Armies', 'sword', ('army_move', 'army_attack')),
    ('g-muster', 'Muster and politics', 'banner', (
        'muster_recruit', 'muster_elite', 'advance_political')),
    ('g-cards', 'Cards', 'card', (
        'play_event', 'draw_card', 'discard_table_card', 'use_token', 'discard_card')),
    ('g-battle', 'Battle', 'shield', (
        'combat_card_play', 'combat_effect_choice', 'casualty_allocate',
        'casualty_preserve_units', 'casualty_preserve_elites', 'siege_assault', 'siege_press',
        'combat_retreat', 'retreat_into_siege', 'victor_advance', 'siege_trim')),
    ('g-hunt', 'The Hunt', 'eye', ('eye_allocate',)),
    ('g-card-effect', 'Card effect', 'card', ('card_option',)),
    ('g-rings', 'Elven Rings, dice and passing', 'ring', ('elven_ring', 'dwarven_ring', 'skip')),
    ('g-other', 'Other', 'info', ()),
)
GROUP_OF = {t: g[0] for g in GROUPS for t in g[3]}
# big families become a picker: type -> (id, title, columns)
COMPOSERS = {
    'army_move': ('c-move', 'Move an army', (('from', 'From'), ('to', 'To'), ('units', 'Units'))),
    'companion_separate': ('c-separate', 'Separate Companions', (('who', 'Who'), ('to', 'To'))),
    'companion_move': ('c-companion', 'Move Companions',
                       (('from', 'From'), ('to', 'To'), ('who', 'Who'))),
    'nazgul_move': ('c-nazgul', 'Move Nazgul', (('from', 'From'), ('to', 'To'))),
    'victor_advance': ('c-advance', 'Advance after the battle', (('to', 'Into'), ('units', 'Units'))),
}


def identity(action):
    """16 hex characters naming one legal action (type, die, parameters)."""
    params = sorted((str(k), str(v)) for k, v in (action.params or {}).items())
    text = repr((action.action_type, action.die_idx, params))
    return hashlib.sha1(text.encode('utf-8', 'replace')).hexdigest()[:16]


def menu_sig(menu):
    h = hashlib.sha1()
    for action in menu:
        h.update(identity(action).encode('ascii') + b'|')
    return h.hexdigest()[:12]


def _die(state, side, action):
    """(face name, letter) of the die an action spends, or None."""
    from . import emitter
    i = action.die_idx
    if i is None or i < 0:
        return None
    try:
        face = state.dice[side][i].current.name
    except (IndexError, AttributeError):
        return None
    return face, emitter._FACE_LETTER.get(face, '?')


def _cells(action, columns):
    p = action.params or {}
    who = p.get('companions') or p.get('companion') or ''
    if isinstance(who, (list, tuple)):
        who = ', '.join(str(w) for w in who)
    cell = {'from': words.place(p.get('from') or p.get('from_region') or ''),
            'to': words.place(p.get('to') or p.get('to_region') or ''),
            'units': words.units_words(p), 'who': str(who)}
    return {key: cell.get(key, '') for key, _label in columns}


def decision(dec_id, kind, state, side, menu):
    """The view's `decision` for one open human decision."""
    options, by_key = [], {}
    for i, action in enumerate(menu):
        die = _die(state, side, action)
        key = (action.action_type, die is None, repr(sorted(
            (str(k), str(v)) for k, v in (action.params or {}).items())))
        choice = None if die is None else {
            'face': die[0], 'letter': die[1], 'i': i, 'identity': identity(action)}
        opt = by_key.get(key)
        if opt is None:
            opt = by_key[key] = {
                'id': 'o%d' % i, 'type': action.action_type,
                'label': words.option_label(state, side, action, side), 'detail': '', 'warn': '',
                'identity': identity(action), 'alts': [], 'dice': [], 'tags': [],
                '_action': action}
            options.append(opt)
            if die is None:
                opt['i'] = i
        elif die is None or any(d['face'] == die[0] for d in opt['dice']):
            opt['alts'].append(i)           # an identical row, or a die of the same face
            continue
        if choice is not None:
            choice['option'] = opt['id']
            opt['dice'].append(choice)
    # rows that read the same get their difference as the detail
    by_label = {}
    for opt in options:
        by_label.setdefault((opt['type'], opt['label']), []).append(opt)
    for same in by_label.values():
        if len(same) > 1:
            keys = sorted({k for o in same for k in (o['_action'].params or {})})
            vary = [k for k in keys if len({str((o['_action'].params or {}).get(k)) for o in same}) > 1]
            for o in same:
                o['detail'] = ', '.join('%s %s' % (k.replace('_', ' '), words.place(
                    (o['_action'].params or {}).get(k))) for k in vary)
    groups = {g[0]: {'id': g[0], 'title': g[1], 'icon': g[2], 'options': []} for g in GROUPS}
    composers = []
    by_type = {}
    for opt in options:
        by_type.setdefault(opt['type'], []).append(opt)
    for kind_of, rows in by_type.items():
        gid = GROUP_OF.get(kind_of, 'g-other')
        spec = COMPOSERS.get(kind_of)
        if spec is not None and len(rows) >= COMPOSER_MIN:
            composers.append({'id': spec[0], 'type': kind_of, 'title': spec[1], 'note': '',
                              'columns': [{'key': k, 'label': lab} for k, lab in spec[2]],
                              'rows': [{'option': _public(o), 'cells': _cells(o['_action'], spec[2])}
                                       for o in rows], 'row_count': len(rows)})
            groups.setdefault('g-' + spec[0], {'id': 'g-' + spec[0], 'title': spec[1],
                                               'icon': 'move', 'options': [],
                                               'composer': spec[0], 'count': len(rows)})
        else:
            groups[gid]['options'].extend(_public(o) for o in rows)
    shown = []
    for g in list(groups.values()):
        if g['options'] or g.get('composer'):
            g.setdefault('count', len(g['options']))
            shown.append(g)
    title, explain = words.decision_words(kind, state)
    return {'id': dec_id, 'kind': kind, 'family': words.family(kind), 'title': title,
            'explain': explain, 'board_hint': '', 'menu_sig': menu_sig(menu),
            'n_options': len(menu), 'quick': [o['id'] for o in groups['g-quick']['options']],
            'groups': shown, 'composers': composers, 'client_steps': [], 'owed': [],
            'partial': None, 'deferred': False}


def _public(opt):
    return {k: v for k, v in opt.items() if not k.startswith('_')}


def roll_decision(need, ask, ctx=''):
    """The pseudo decision shown while the game waits for the person's dice
    (`ask` numbers the requests, so each one is answered once)."""
    n = max(1, min(5, int(need)))
    step = {'id': 'cs-roll-%d' % n, 'label': 'Roll %d %s' % (n, 'die' if n == 1 else 'dice'),
            'why': 'The game waits for your dice.', 'step': {'t': 'cmd', 'cmd': 'roll%d' % n},
            'headline': words.waiting_for_roll(int(need), ctx)}
    return {'id': 'd-roll-%d-%d' % (int(need), int(ask)), 'kind': 'roll', 'family': 'action',
            'title': step['label'], 'explain': 'Each side rolls its own dice.', 'board_hint': '',
            'menu_sig': '', 'n_options': 0, 'quick': [], 'groups': [], 'composers': [],
            'client_steps': [step], 'owed': [], 'partial': None, 'deferred': False}


def indices(dec):
    """Every menu index the decision shows (the coverage the tests assert)."""
    out = []
    rows = [o for g in dec['groups'] for o in g['options']]
    rows += [r['option'] for c in dec['composers'] for r in c['rows']]
    for o in rows:
        if o.get('i', -1) >= 0:
            out.append(o['i'])
        out.extend(d['i'] for d in o['dice'])
        out.extend(o['alts'])
    return out


# ------------------------------------------------------------------ overview
def _overview(state, human, phase):
    import game_logic as gl
    import game_state as gs
    from . import emitter
    FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW
    ov = {'turn': int(state.turn), 'phase': phase}
    hunt_ids = [id(d) for d in state.hunt_box]
    dice = {'unused': {}}
    for side in (FP, SH):
        rows = []
        for i, d in enumerate(state.dice.get(side, [])):
            face = getattr(d.current, 'name', None)
            rows.append({'idx': i, 'face': face, 'letter': emitter._FACE_LETTER.get(face),
                         'label': (face or 'Not rolled').replace('_', ' ').title(),
                         'used': bool(d.used),
                         'where': 'hunt_box' if id(d) in hunt_ids else 'used' if d.used else 'pool'})
        dice[side.name] = rows
        dice['unused'][side.name] = sum(1 for r in rows if not r['used'] and r['face'])
    eyes = sum(1 for d in state.hunt_box if getattr(d.current, 'name', '') == 'EYE')
    dice['hunt_box'] = {'eyes': eyes, 'fp_dice': len(state.hunt_box) - eyes,
                        'text': '%d Eye(s), %d Free Peoples dice' % (eyes, len(state.hunt_box) - eyes)}
    ov['dice'] = dice
    fs = state.fellowship
    hidden = bool(fs.hidden)
    mordor = bool(fs.mordor_entered)
    where = None if (hidden and human is SH) else (
        'Mordor' if mordor else words.place(gl.fellowship_region(state)))
    guide = getattr(fs.guide, 'name', None)
    ov['fellowship'] = {
        'progress': int(fs.progress or 0), 'corruption': int(fs.corruption or 0),
        'corruption_max': 12, 'hidden': hidden, 'region': where, 'mordor': mordor,
        'mordor_step': int(fs.mordor_step or 0), 'mordor_step_label': '', 'guide': guide,
        'companions': [{'name': c.name, 'level': int(getattr(c, 'level', 0) or 0)}
                       for c in fs.companions], 'separated': [],
        'text': 'The Fellowship is %s%s with progress %d and %d Corruption.%s' % (
            'hidden' if hidden else 'revealed', ' at %s' % where if where else '',
            int(fs.progress or 0), int(fs.corruption or 0),
            ' %s is the Guide.' % guide if guide else '')}
    pool = list(state.hunt_pool or ())
    pool_eyes = sum(1 for t in pool if getattr(t, 'is_eye', False))
    ov['hunt'] = {'pool_total': len(pool), 'pool_eyes': pool_eyes,
                  'pool_text': '%d tiles in the Hunt Pool (%d Eyes)' % (len(pool), pool_eyes),
                  'used': [str(t) for t in state.hunt_discard or ()],
                  'removed': [str(t) for t in state.hunt_removed or ()],
                  'track': int(fs.progress or 0)}
    vc = gl.check_all_victory_conditions(state)
    ov['vp'] = {'FREE_PEOPLES': vc.get('free_peoples_vp'), 'SHADOW': vc.get('shadow_vp'),
                'fp_target': 4, 'shadow_target': 10}
    ov['politics'] = [
        {'nation': words.place(getattr(n, 'value', n)),
         'side': 'FREE_PEOPLES' if emitter._side_of_nation(n) == 'fp' else 'SHADOW',
         'step': max(0, 3 - int(step)), 'step_label': 'At War' if int(step) >= 3 else
         '%d step(s) from war' % (3 - int(step)), 'active': bool(state.political.active.get(n))}
        for n, step in state.political.boxes.items()]
    hands = {}
    for side in (FP, SH):
        hand = list(state.hands[side].hand)
        # counts only: the person sees his own cards in the client, and no
        # card he holds is named outside his own options
        hands[side.name] = {'count': len(hand), 'cards': []}
    ov['hands'] = hands
    ov['decks'] = {'fp_character': len(state.fp_char_deck), 'fp_strategy': len(state.fp_strat_deck),
                   'sh_character': len(state.sh_char_deck), 'sh_strategy': len(state.sh_strat_deck)}
    table = getattr(state, 'persistent_cards', None)
    ov['table_cards'] = [{'name': str(getattr(c, 'card_name', c.card_id)), 'side': tag}
                         for attr, tag in (('fp_cards', 'FREE_PEOPLES'), ('sh_cards', 'SHADOW'))
                         for c in getattr(table, attr, ())]
    fp_rings, sh_rings = int(state.fp_elven_rings), int(state.sh_elven_rings)
    ov['elven_rings'] = {'FREE_PEOPLES': fp_rings, 'dark_ring': sh_rings,
                         'text': 'The Free Peoples hold %d Elven Ring(s); %d on the Dark Ring box.'
                         % (fp_rings, sh_rings)}
    ov['besieged'] = [words.place(b) for b in sorted(str(b) for b in
                                                     (state.besieged_strongholds or ()))]
    ov['texts'] = ['%s is under siege.' % b for b in ov['besieged']]
    return ov


def _army(state, region, side, human):
    import combat as cb
    army = next((a for k, a in state.armies.items() if str(k).lower() == str(region).lower()), None)
    counts = {}
    for u in getattr(army, 'units', ()) or ():
        if cb.side_of_nation(u.nation) is side:
            key = '%s %s' % (words.place(getattr(u.nation, 'value', u.nation)),
                             str(getattr(u.kind, 'value', u.kind)).title())
            counts[key] = counts.get(key, 0) + 1
    units = [{'tkey': k.lower(), 'label': k, 'n': n} for k, n in sorted(counts.items())]
    return {'side': side.name, 'label': 'You' if side is human else words.side_name(side),
            'region': words.place(region), 'army': units,
            'army_text': ', '.join('%d %s' % (u['n'], u['label']) for u in units) or 'no units'}


def _tabs(state, human):
    """The battle / hunt / card tab contents while one is open."""
    pc = state.pending_combat if isinstance(state.pending_combat, dict) else None
    combat = hunt = card = None
    if pc is not None:
        phase = str(pc.get('phase') or '')
        current = ('cards' if 'card' in phase else 'casualties' if 'casualt' in phase else
                   'continue' if phase in ('combat_round_result', 'attacker_continue',
                                           'defender_retreat', 'victor_advance') else 'roll')
        order = ('cards', 'roll', 'casualties', 'continue')
        labels = {'cards': 'Combat cards', 'roll': 'Combat roll', 'casualties': 'Casualties',
                  'continue': 'Continue or stop'}
        combat = {
            'title': 'Battle for %s' % words.place(pc.get('defender_region')),
            'round': int(pc.get('round', pc.get('rounds_fought', 0)) or 0),
            'is_siege': bool(pc.get('is_siege')),
            'attacker': _army(state, pc.get('attacker_region'), pc.get('attacker_side'), human),
            'defender': _army(state, pc.get('defender_region'), pc.get('defender_side'), human),
            'steps': [{'id': s, 'label': labels[s], 'state': 'current' if s == current else
                       'done' if order.index(s) < order.index(current) else 'todo'} for s in order],
            'hits': {'text': ''}, 'cards': {}, 'rolls': {}}
    if state.pending_hunt is not None:
        damage = int((state.pending_hunt or {}).get('damage', 0) or 0)
        hunt = {'title': 'The Hunt', 'damage': damage,
                'text': 'The Hunt found the Fellowship: %d damage to take.' % damage,
                'in_mordor': bool(state.fellowship.mordor_entered),
                'corruption': int(state.fellowship.corruption or 0), 'corruption_max': 12}
    pco = getattr(state, 'pending_card_option', None) or getattr(state, 'pending_card_followup', None)
    if pco and not words.inner_step(pco):
        name = str(pco.get('card_name') or words.card_title(pco.get('card_id', '')))
        card = {'title': name, 'now': True, 'step_text': 'Choose how the card resolves.',
                'card': {'name': name, 'side': 'FREE_PEOPLES' if str(pco.get('side')) == 'fp'
                         else 'SHADOW', 'deck': ''}, 'chain': [], 'done': False}
    return combat, hunt, card


def view(facts, seq):
    """The whole ai_panel_view.json for the position and the loop's words in `facts` (Facts)."""
    pos = facts.pos
    state = pos.state
    human = facts.human
    over = None
    if pos.over:
        over = {'victory': str(pos.over.get('victory')), 'type': str(pos.over.get('victory_type')),
                'text': words.game_over(pos.over), 'turn': int(state.turn), 'final': True,
                'vp': {'FREE_PEOPLES': pos.over.get('free_peoples_vp'),
                       'SHADOW': pos.over.get('shadow_vp')}}
    dec = None if over else facts.decision
    combat, hunt, card = _tabs(state, human)
    if human is None:
        tone, text = 'info', 'Both sides are played by the AI.'
    elif over:
        tone, text = 'over', over['text']
    elif dec is not None:
        tone, text = 'you', facts.how
    else:
        tone, text = 'ai' if facts.how.startswith('The ') else 'wait', facts.how
    ai_side = next(iter(facts.ai_sides)) if human is not None else None
    levers = [] if human is None else [
        {'op': 'put_back', 'label': 'Put the board back', 'group': 'fix', 'enabled': True,
         'explain': 'Moves every piece to where the game has it.', 'disabled_reason': None,
         'danger': False, 'primary': True, 'client': False},
        {'op': 'take_back', 'label': 'Take back my last decision', 'group': 'fix',
         'explain': 'Returns to your last choice; possible until the AI has answered it.',
         'enabled': facts.take_back is None,
         'disabled_reason': facts.take_back, 'danger': False, 'primary': False,
         'client': False}] + [
        # buttons the panel shows whatever the server offers: published switched off, so
        # that it greys them and prints the reason
        {'op': op, 'label': words.LEVERS_OFF[op][0], 'group': 'other', 'enabled': False,
         'explain': '', 'disabled_reason': words.LEVERS_OFF[op][1], 'danger': False,
         'primary': False, 'client': False} for op in ('skip', 'reset_resync', 'resync')]
    return {
        'schema': SCHEMA, 'seq': seq, 'ts': time.time(), 'server_pid': os.getpid(),
        'session': str(facts.session or ''), 'mode': 'ai_vs_ai' if human is None else 'sided',
        'human': None if human is None else {
            'side': human.name, 'nick': facts.human_nick, 'label': words.side_name(human)},
        'ai': {'side': 'BOTH' if human is None else ai_side.name,
               'nick': None if human is None else facts.ai_nick,
               'label': 'Both sides' if human is None else words.side_name(ai_side),
               'thinking': False, 'thinking_since': None, 'overdue': False, 'sims': facts.sims,
               'net': facts.net_tag},
        'decision': dec,
        'headline': {'turn': int(state.turn), 'phase': pos.phase, 'phase_label': pos.phase,
                     'to_act': getattr(pos.active, 'name', None), 'you_act': dec is not None,
                     'tone': tone, 'text': text, 'waiting_on': facts.how},
        'overview': _overview(state, human, pos.phase),
        'combat': combat, 'hunt': hunt, 'card': card, 'undo': None, 'ai_last': facts.ai_last,
        'recovery': {'state': 'ok', 'headline': '', 'levers': levers, 'offer': None,
                     'settings': {'sims': facts.sims, 'max_children': 0, 'chance_k': 0,
                                  'enforce_human': True},
                     'last_result': facts.lever_said},
        'log_tail': list(facts.log_tail), 'replies': list(facts.replies),
        'reply': facts.replies[0] if facts.replies else None, 'game_over': over}


def publish(facts, client_dir, sent):
    """Write the view.  An unchanged view is written again only as a heartbeat.
    facts: a callable giving the Facts of now; sent: [number, text, time] of the view written
    last, kept by the caller and counted on here."""
    sent[0] += 1
    body = view(facts(), sent[0])
    same = json.dumps({k: v for k, v in body.items() if k not in ('seq', 'ts')}, default=str)
    if same == sent[1] and time.time() - sent[2] < 2.0:
        return
    try:
        atomic_write(os.path.join(client_dir, VIEW_NAME), json.dumps(body, default=str))
        sent[1:] = [same, time.time()]
    except OSError:     # the panel holds the file for the moment: the next publish writes it
        pass


def request(text):
    """The panel's request as its slot gave it (FileBridge.panel: moved aside, so the panel
    may write the next one), or None: nothing was there, or it is no request."""
    try:
        req = json.loads(text) if text else None
    except ValueError:
        return None
    return req if isinstance(req, dict) else None
