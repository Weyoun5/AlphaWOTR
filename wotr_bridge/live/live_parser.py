"""Rendered feed line -> record, LIVE channel.

The grammar itself is src/rendered_replay.py's parse_line — the
corpus-proven single implementation (10.8K games behind a quarantine
gate).  This module adds ONLY what live consumption needs (A15/A25-R2):

  * prefix normalization — live writes '$Nick ...' / '<auto> ...' /
    '<game> ...'; replay writes '%Nick ...' / '%auto> ...'.  parse_line
    expects the replay form with '%' pre-stripped, so live '$' is
    stripped and '<auto> ' becomes 'auto> ' (measured: parse_line's
    '<'-skip fires before its auto-move check; '<game> ... rolls' is
    matched BEFORE the skip and passes through untouched);
  * chat extraction — '<Nick> body' lines are the sided chat protocol,
    not board grammar; they become {'op':'chat'} records here instead
    of parse_line's None;
  * the unknown-shape alarm (design rule 12, gate-2 discipline): a
    board-authored line that parses to op:'other' is surfaced loudly,
    never dropped.

ATTRIBUTION (sided-play ruling): '$Nick' is NOT attribution — the
client stamps the LOCAL seat's nick on lines its own built-ins emit.
Records keep the nick for display only; ownership is derived from
ZONES downstream.  Nothing in this module may branch on the nick.
"""
import os
import re
import sys

# wotr_bridge/live/ -> project root is two levels up from this file
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, '..', '..', 'src'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import rendered_replay as rr
if __package__:
    from .action_die_zones import temporary as temporary_die_zone
else:  # The standalone native Undo model also imports this grammar.
    from action_die_zones import temporary as temporary_die_zone

# live chat line: "<Nick> body" — but NOT "<auto>"/"<game>" (board/system)
_CHAT_RE = re.compile(r'^<([^>]+)>\s?(.*)$')
_SYSTEM_TAGS = frozenset(('auto', 'game', 'replay', 'address'))

# gate-2 classified "other"-op shapes, lifted from the corpus extractor
# (rendered_extract.py:121-125) so the live alarm has the same whitelist
# the 10.8K-game quarantine proved: views/deselects are pure UI no-ops;
# die changes and random discards are RECOGNIZED decision-bearing shapes.
_NOOP_RE = re.compile(r'^\S+\s+(views\s+the\b|deselects\s+a\b)', re.I)
_DIECHANGE_RE = re.compile(
    r'^\S+\s+changes\s+(Shadow|Free)\s+Action\s+Die'
    r'(?:\s+\(([ACEHMPW])\)\s+to\s+([ACEHMPW]))?')
_RANDDISC_RE = re.compile(r'^\S+\s+randomly\s+discards\s+a\s+')


def normalize(raw):
    """Live line -> the form parse_line expects, or a chat/system tag.

    Returns (kind, text):
      kind 'board'  -> text ready for rr.parse_line
      kind 'chat'   -> text is the raw line (caller extracts nick/body)
      kind 'blank'  -> ignorable
    """
    s = rr.canonical_automatic_line(raw.strip())
    if not s:
        return ('blank', '')
    if s.startswith('$'):
        return ('board', s[1:])
    if s.startswith('%'):
        # replay channel: '%' prefixes EVERY line kind ('%Nick moves',
        # '%<game> ... rolls', '%<Nick> chat', '%auto> ...') — strip it
        # and classify the remainder by the same taxonomy
        return normalize(s[1:])
    if s.startswith('<auto> '):
        return ('board', 'auto> ' + s[len('<auto> '):])
    if s.startswith('<game> '):
        return ('board', s)                    # _ROLL_NUM_RE needs the tag
    m = _CHAT_RE.match(s)
    if m and m.group(1) not in _SYSTEM_TAGS:
        return ('chat', s)
    return ('board', s)


def parse(raw):
    """One feed line -> one record dict (never None for real content).

    Record ops: everything rr.parse_line emits (move/kill/flip/draw/
    roll_dice/roll_num/add_eye/alloc_eye/choose/undo/redo/turn/other)
    plus 'chat' {nick, body} and 'blank'.  op:'other' records carry
    unknown=True so the driver's alarm (rule 12) cannot miss them."""
    kind, text = normalize(raw)
    if kind == 'blank':
        return {'op': 'blank', 'raw': raw}
    if isinstance(raw, str) and raw.startswith('#BLIND '):
        # the agent's blinding attestation (AiFeed premain): consumed by
        # live_server's fail-closed check, never a gesture
        return {'op': 'system', 'raw': raw, 'attest': True}
    if kind == 'chat':
        m = _CHAT_RE.match(text)
        return {'op': 'chat', 'nick': m.group(1), 'body': m.group(2),
                'raw': raw}
    if text == '!!! Warning no units available in casualties, elite was downgraded from reinforcements':
        # Native Downgrade/Upgrade explains where it obtained the Regular.
        # The following kill and move records carry the actual payment; this
        # notice neither pays a hit nor changes the engine by itself.
        return {'op': 'system', 'raw': raw}
    rec = rr.parse_line(text)
    if rec is not None:
        rec.setdefault('raw', raw)
        rec.setdefault('norm', text)     # prefix-stripped board text
        # Native players may stage through either Current Card slot before
        # moving the same die into Current Action. This first hop opens an
        # unpaid window; the normal matching die terminator still completes
        # it. Keep the prefix in that window for cancellation and native Undo.
        if rec.get('op') == 'move' and not rec.get('die_stage'):
            staged = rr._DIE_TO_CARD_RE.match(text)
            if staged:
                shadow = staged.group(1).lower() == 'shadow'
                prefix = 'SA' if shadow else 'FP'
                if (rec['from'].lower() == (prefix + ' Dice').lower()
                        and temporary_die_zone(rec['to'])):
                    rec['die_stage'] = ('SHADOW' if shadow else 'FREE_PEOPLES',
                                        staged.group(2).upper())
    if rec is None:
        if text.startswith('<'):
            # parse_line's documented '<'-skip class: <game>/<replay>/
            # <address> tags that are neither rolls nor Turn markers
            return {'op': 'system', 'raw': raw}
        # a player-authored line the grammar cannot classify at all
        return {'op': 'other', 'raw': raw, 'unknown': True}
    if rec.get('op') == 'other':
        s = rec.get('raw', text)
        if _NOOP_RE.match(s):
            return {'op': 'noop', 'raw': raw}
        md = _DIECHANGE_RE.match(s)
        if md:
            rec2 = {'op': 'die_change', 'raw': raw, 'norm': text,
                    'player': text.split(None, 1)[0],
                    'side': ('SHADOW' if md.group(1) == 'Shadow'
                             else 'FREE_PEOPLES')}
            if md.group(2) and md.group(3):
                rec2['frm'] = md.group(2)
                rec2['to'] = md.group(3)
            return rec2
        if _RANDDISC_RE.match(s):
            return {'op': 'random_discard', 'raw': raw}
        rec['unknown'] = True
        rec['raw'] = raw
    return rec
