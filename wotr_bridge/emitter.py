# -*- coding: utf-8 -*-
"""Client-line emitter: the inverse of src/log_replay.py's line parser.

The Java client speaks one language — log lines.  To PLAY (not just
read), the AI must render its chosen engine actions back into that
language, byte-compatible with what a human click would have produced,
because the client's Talker (game.z.a) applies our line through the
same code path as a remote player's.

Two layers:

  L1  micro-op renderers  (this file, `classify_line` / `render`)
      A micro-op is one client gesture: drag pieces zone->zone, spend
      a die, draw a card, roll dice, allocate an eye, flip a token...
      `classify_line` decomposes a logged player line into
      (op, fields); `render` re-builds the line from (op, fields).
      Round-trip over the certified corpus (roundtrip_test.py) must be
      BYTE-equal (modulo the trailing checksum token, ' null' locally),
      which proves every formatter matches the client's own renderer.

  L2  action planners  (plan_* below)
      Translate one engine Action into an ordered list of micro-ops
      with CONCRETE piece ids, chosen from a live LogReplayer's
      piece_region / die_id_map tables.  Validated by type-equivalence
      against the human's logged pieces (the engine doesn't care WHICH
      of two interchangeable regulars moves, but the piece types, the
      zones and the grammar must match exactly).

Zone codes ('Game.2043') pass through untouched — the parser's
decode_location owns their meaning; planners obtain codes via
ZONE_BY_NAME (most-frequent code per decoded name over the corpus).
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
SRC = os.path.join(PROJECT, 'src')

_CHECKSUM_RE = re.compile(r'\s+(?:[A-F0-9]{6}|null)$')


def strip_tail(line):
    """Drop the trailing checksum token (networked) / 'null' (local)."""
    return _CHECKSUM_RE.sub('', line.rstrip('\r\n'))


# --------------------------------------------------------------- L1 ops
# Each op: (regex over the checksum-stripped line, python format string
# over the named fields).  Field conventions:
#   nick          issuing player ($ prefix)
#   pieces        list of piece-id tokens (client zero-padding kept)
#   frm/to        raw zone codes WITHOUT <~ ~> ('Game.2043',
#                 'Controls.46', ...)
#   die/piece     single piece-id token
# The format strings reproduce the client's exact spacing (trailing
# space after multi-piece groups, none after single-piece groups) —
# THE thing L1 round-trip exists to prove.

def _op(pattern, fmt, listfields=()):
    return {'re': re.compile(pattern + r'$'), 'fmt': fmt,
            'lists': listfields}


OPS = {
    # generic piece drag (armies, companions, markers, cards to play
    # areas, mordor track, political walks in Scribe builds, ...)
    'move': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Scribe\.141~>(?P<pieces>[\d ]+?) ?'
        r'<~Scribe\.142~><~(?P<frm>[\w.]+)~>'
        r'<~Scribe\.143~><~(?P<to>[\w.]+)~><~Scribe\.144~>\s*',
        '$%(nick)s<~Scribe.141~>%(pieces)s <~Scribe.142~><~%(frm)s~>'
        '<~Scribe.143~><~%(to)s~><~Scribe.144~>',
        listfields=('pieces',)),
    # face-up card reveal-drag (event card PLAY): commit hash + value
    'reveal_move': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Scribe\.141~>reveal (?P<piece>\d+) '
        r'(?P<hash>[A-F0-9]{40}) (?P<value>\d+) ?'
        r'<~Scribe\.142~><~(?P<frm>[\w.]+)~>'
        r'<~Scribe\.143~><~(?P<to>[\w.]+)~><~Scribe\.144~>\s*',
        '$%(nick)s<~Scribe.141~>reveal %(piece)s %(hash)s %(value)s '
        '<~Scribe.142~><~%(frm)s~><~Scribe.143~><~%(to)s~>'
        '<~Scribe.144~>'),
    # card reveal via the card-buffet (combat cards etc.)
    'reveal_buffet': _op(
        r'^\$(?P<nick>[^<>\s]+)<~CardBuffet\.14~>reveal (?P<piece>\d+) '
        r'(?P<hash>[A-F0-9]{40}) (?P<value>\d+)'
        r'<~CardBuffet\.15~><~(?P<frm>[\w.]+)~><~CardBuffet\.16~>\s*',
        '$%(nick)s<~CardBuffet.14~>reveal %(piece)s %(hash)s %(value)s'
        '<~CardBuffet.15~><~%(frm)s~><~CardBuffet.16~>'),
    # Shadow die spend: pool -> used (Game.2035)
    'die_spend_sh': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Controls\.276~>(?P<die>\d+)'
        r'<~Controls\.277~><~(?P<frm>[\w.]+)~>'
        r'<~Controls\.278~><~(?P<to>[\w.]+)~>',
        '$%(nick)s<~Controls.276~>%(die)s<~Controls.277~><~%(frm)s~>'
        '<~Controls.278~><~%(to)s~>'),
    # FP die spend: pool -> used
    'die_spend_fp': _op(
        r'^\$(?P<nick>[^<>\s]+)<~HighlightLabel\.86~>(?P<die>\d+)'
        r'<~HighlightLabel\.87~><~(?P<frm>[\w.]+)~>'
        r'<~HighlightLabel\.88~><~(?P<to>[\w.]+)~>',
        '$%(nick)s<~HighlightLabel.86~>%(die)s<~HighlightLabel.87~>'
        '<~%(frm)s~><~HighlightLabel.88~><~%(to)s~>'),
    # card draw: deck -> hand (Controls.45 FP / .46 SH)
    'card_draw': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Controls\.509~>(?P<piece>\d+)'
        r'<~Controls\.510~><~(?P<frm>[\w.]+)~>'
        r'<~Controls\.511~><~(?P<to>[\w.]+)~>',
        '$%(nick)s<~Controls.509~>%(piece)s<~Controls.510~><~%(frm)s~>'
        '<~Controls.511~><~%(to)s~>'),
    # face-up reveal THROUGH the hand-exit grammar (some builds)
    'reveal_hand_exit': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Controls\.281~>reveal (?P<piece>\d+) '
        r'(?P<hash>[A-F0-9]{40}) (?P<value>\d+) ?'
        r'<~Controls\.282~><~(?P<frm>[\w.]+)~>'
        r'<~Controls\.283~><~(?P<to>[\w.]+)~>',
        '$%(nick)s<~Controls.281~>reveal %(piece)s %(hash)s '
        '%(value)s <~Controls.282~><~%(frm)s~>'
        '<~Controls.283~><~%(to)s~>'),
    # face-down hand exit, grammar A: piece(s) from hand to zone
    'hand_exit': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Controls\.281~>(?P<pieces>[\d ]+?) ?'
        r'<~Controls\.282~><~(?P<frm>[\w.]+)~>'
        r'(?:<~Controls\.283~><~(?P<to>[\w.]+)~>)?\s*',
        None),  # asymmetric optional tail — rendered by _fmt_hand_exit
    # face-down hand exit, grammar B: (piece, hand)
    'hand_exit_b': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Controls\.296~>(?P<pieces>[\d ]+?) ?'
        r'<~Controls\.297~><~(?P<frm>[\w.]+)~>',
        '$%(nick)s<~Controls.296~>%(pieces)s <~Controls.297~>'
        '<~%(frm)s~>',
        listfields=('pieces',)),
    # die -> hunt box
    'die_to_hunt': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Controls\.795~>(?P<die>\d+)'
        r'<~Controls\.796~><~(?P<frm>[\w.]+)~>'
        r'<~Controls\.797~><~(?P<to>[\w.]+)~>',
        '$%(nick)s<~Controls.795~>%(die)s<~Controls.796~><~%(frm)s~>'
        '<~Controls.797~><~%(to)s~>'),
    # fellowship / corruption track marker
    'track_marker': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Controls\.790~>(?P<piece>\d+)'
        r'<~Controls\.791~><~(?P<frm>[\w.]+)~>'
        r'<~Controls\.792~><~(?P<to>[\w.]+)~>',
        '$%(nick)s<~Controls.790~>%(piece)s<~Controls.791~><~%(frm)s~>'
        '<~Controls.792~><~%(to)s~>'),
    # hunt tile: pool -> display
    'hunt_tile': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Controls\.624~>(?P<piece>\d+)'
        r'<~Controls\.625~><~(?P<frm>[\w.]+)~>'
        r'<~Controls\.626~><~(?P<to>[\w.]+)~>',
        '$%(nick)s<~Controls.624~>%(piece)s<~Controls.625~><~%(frm)s~>'
        '<~Controls.626~><~%(to)s~>'),
    # kill / cleanup (casualties to Game.XXXX; empty piece list = the
    # client's own no-op misclick line, kept for byte fidelity)
    'kill': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Controls\.644~>(?P<pieces>[\d ]*?) ?'
        r'<~Controls\.645~><~(?P<to>[\w.]+)~>',
        '$%(nick)s<~Controls.644~>%(pieces)s <~Controls.645~>'
        '<~%(to)s~>',
        listfields=('pieces',)),
    # dice roll: ids + values
    'dice_roll': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Controls\.345~>(?P<dice>[\d ]+?) ?'
        r'<~Controls\.346~>(?P<values>[\d ]+?) ?',
        '$%(nick)s<~Controls.345~>%(dice)s <~Controls.346~>%(values)s',
        listfields=('dice', 'values')),
    # dice pickup (companion of a roll)
    'dice_pickup': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Controls\.350~>(?P<dice>[\d ]+?) ?',
        '$%(nick)s<~Controls.350~>%(dice)s',
        listfields=('dice',)),
    # eye allocation before the roll
    'eye_alloc': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Controls\.372~>(?P<die>\d+)',
        '$%(nick)s<~Controls.372~>%(die)s'),
    # flip token(s) / character cards / settlement markers
    'flip': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Controls\.323~>(?P<pieces>[\d ]*?) ?',
        '$%(nick)s<~Controls.323~>%(pieces)s',
        listfields=('pieces',)),
    # FP die spend, HighlightLabel-81 builds (5th-die encoding)
    'die_spend_fp81': _op(
        r'^\$(?P<nick>[^<>\s]+)<~HighlightLabel\.81~>(?P<die>\d+)'
        r'<~HighlightLabel\.82~><~(?P<frm>[\w.]+)~>'
        r'<~HighlightLabel\.83~><~(?P<to>[\w.]+)~>',
        '$%(nick)s<~HighlightLabel.81~>%(die)s<~HighlightLabel.82~>'
        '<~%(frm)s~><~HighlightLabel.83~><~%(to)s~>'),
    # card drag between zones/hands (CardBuffet builds)
    'card_move': _op(
        r'^\$(?P<nick>[^<>\s]+)<~CardBuffet\.3~>(?P<pieces>[\d ]+?) ?'
        r'<~CardBuffet\.4~><~(?P<frm>[\w.]+)~>'
        r'<~CardBuffet\.5~><~(?P<to>[\w.]+)~>',
        '$%(nick)s<~CardBuffet.3~>%(pieces)s <~CardBuffet.4~>'
        '<~%(frm)s~><~CardBuffet.5~><~%(to)s~>',
        listfields=('pieces',)),
    # plaintext selection clear (no-op for state)
    'deselects': _op(
        r'^\$(?P<nick>[^<>\s]+) deselects\s*',
        '$%(nick)s deselects'),
    # cosmetic piece rename ('names 240 Gandalf2')
    'names': _op(
        r'^\$(?P<nick>[^<>\s]+) names (?P<piece>\d+) (?P<name>.*?) ?',
        '$%(nick)s names %(piece)s %(name)s'),
    # plaintext counter adjust ('adds 1 Dwarven Ring(s)')
    'counter_adjust': _op(
        r'^\$(?P<nick>[^<>\s]+) (?P<verb>adds|removes) '
        r'(?P<count>\d+) (?P<what>.*?) ?',
        '$%(nick)s %(verb)s %(count)s %(what)s'),
    # no-arg control button (Controls.179)
    'ctrl_179': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Controls\.179~>\s*',
        '$%(nick)s<~Controls.179~>'),
    # die refunds (Controls.736 / 767) and pool return (784)
    'die_refund_736': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Controls\.736~>(?P<pieces>[\d ]+?) ?'
        r'<~Controls\.737~><~(?P<frm>[\w.]+)~>'
        r'<~Controls\.738~><~(?P<to>[\w.]+)~>',
        '$%(nick)s<~Controls.736~>%(pieces)s <~Controls.737~>'
        '<~%(frm)s~><~Controls.738~><~%(to)s~>',
        listfields=('pieces',)),
    'die_refund_767': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Controls\.767~>(?P<pieces>[\d ]+?) ?'
        r'<~Controls\.768~><~(?P<frm>[\w.]+)~>'
        r'<~Controls\.769~><~(?P<to>[\w.]+)~>',
        '$%(nick)s<~Controls.767~>%(pieces)s <~Controls.768~>'
        '<~%(frm)s~><~Controls.769~><~%(to)s~>',
        listfields=('pieces',)),
    'die_refund_784': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Controls\.784~>(?P<pieces>[\d ]+?) ?'
        r'<~Controls\.785~><~(?P<frm>[\w.]+)~>'
        r'<~Controls\.786~><~(?P<to>[\w.]+)~>',
        '$%(nick)s<~Controls.784~>%(pieces)s <~Controls.785~>'
        '<~%(frm)s~><~Controls.786~><~%(to)s~>',
        listfields=('pieces',)),
    # card-buffet drag without reveal, grammar B
    'buffet_move14': _op(
        r'^\$(?P<nick>[^<>\s]+)<~CardBuffet\.14~>(?P<piece>\d+)'
        r'<~CardBuffet\.15~><~(?P<frm>[\w.]+)~><~CardBuffet\.16~>\s*',
        '$%(nick)s<~CardBuffet.14~>%(piece)s<~CardBuffet.15~>'
        '<~%(frm)s~><~CardBuffet.16~>'),
    # die face change (Elven Ring / Mouth of Sauron): plaintext grammar,
    # double space after the die id is the client's own formatting
    'die_change': _op(
        r'^\$(?P<nick>[^<>\s]+) changes (?P<die>\d+)  to (?P<face>\w+)',
        '$%(nick)s changes %(die)s  to %(face)s'),
    # reinforcement swap: place pieces onto a map zone from the pools
    'swap_place': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Controls\.304~>(?P<pieces>[\d ]+?) ?'
        r'<~Controls\.305~><~(?P<to>[\w.]+)~>',
        '$%(nick)s<~Controls.304~>%(pieces)s <~Controls.305~>'
        '<~%(to)s~>',
        listfields=('pieces',)),
    # reinforcement swap: remove pieces (sub-key 307/308/309 = pool
    # row; optional readable subtext like '(1)/casualties(1)')
    'swap_remove': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Controls\.313~>(?P<pieces>[\d ]+?) ?'
        r'<~Controls\.(?P<sub>30[789])~>(?P<subtext>[^<]*)'
        r'<~Controls\.314~><~(?P<to>[\w.]+)~>',
        '$%(nick)s<~Controls.313~>%(pieces)s <~Controls.%(sub)s~>'
        '%(subtext)s<~Controls.314~><~%(to)s~>',
        listfields=('pieces',)),
    # pieces -> zone (companion/minion cleanup to used box)
    'move_662': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Controls\.662~>(?P<pieces>[\d ]+?) ?'
        r'<~Controls\.663~><~(?P<to>[\w.]+)~>',
        '$%(nick)s<~Controls.662~>%(pieces)s <~Controls.663~>'
        '<~%(to)s~>',
        listfields=('pieces',)),
    # single piece quad variants seen across client builds
    'move_637': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Controls\.637~>(?P<piece>\d+)'
        r'<~Controls\.638~><~(?P<frm>[\w.]+)~>'
        r'<~Controls\.639~><~(?P<to>[\w.]+)~>',
        '$%(nick)s<~Controls.637~>%(piece)s<~Controls.638~><~%(frm)s~>'
        '<~Controls.639~><~%(to)s~>'),
    'marker_366': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Controls\.366~>(?P<piece>\d+)'
        r'<~Controls\.367~><~(?P<frm>[\w.]+)~>'
        r'<~Controls\.368~><~(?P<to>[\w.]+)~>',
        '$%(nick)s<~Controls.366~>%(piece)s<~Controls.367~><~%(frm)s~>'
        '<~Controls.368~><~%(to)s~>'),
    'token_271': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Controls\.271~>(?P<piece>\d+)'
        r'<~Controls\.272~><~(?P<frm>[\w.]+)~>'
        r'<~Controls\.273~><~(?P<to>[\w.]+)~>',
        '$%(nick)s<~Controls.271~>%(piece)s<~Controls.272~><~%(frm)s~>'
        '<~Controls.273~><~%(to)s~>'),
    # political-track marker walks (HighlightLabel builds)
    'political_91': _op(
        r'^\$(?P<nick>[^<>\s]+)<~HighlightLabel\.91~>(?P<piece>\d+)'
        r'<~HighlightLabel\.92~><~(?P<frm>[\w.]+)~>'
        r'<~HighlightLabel\.93~><~(?P<to>[\w.]+)~>',
        '$%(nick)s<~HighlightLabel.91~>%(piece)s<~HighlightLabel.92~>'
        '<~%(frm)s~><~HighlightLabel.93~><~%(to)s~>'),
    'political_96': _op(
        r'^\$(?P<nick>[^<>\s]+)<~HighlightLabel\.96~>(?P<piece>\d+)'
        r'<~HighlightLabel\.97~><~(?P<frm>[\w.]+)~>'
        r'<~HighlightLabel\.98~><~(?P<to>[\w.]+)~>',
        '$%(nick)s<~HighlightLabel.96~>%(piece)s<~HighlightLabel.97~>'
        '<~%(frm)s~><~HighlightLabel.98~><~%(to)s~>'),
    'political_101': _op(
        r'^\$(?P<nick>[^<>\s]+)<~HighlightLabel\.101~>(?P<piece>\d+)'
        r'<~HighlightLabel\.102~><~(?P<frm>[\w.]+)~>'
        r'<~HighlightLabel\.103~><~(?P<to>[\w.]+)~>',
        '$%(nick)s<~HighlightLabel.101~>%(piece)s'
        '<~HighlightLabel.102~><~%(frm)s~>'
        '<~HighlightLabel.103~><~%(to)s~>'),
    # face-down card play: piece from hand to a play zone
    'card_play_452': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Controls\.452~>(?P<piece>\d+)'
        r'<~Controls\.453~><~(?P<frm>[\w.]+)~>'
        r'<~Controls\.454~><~(?P<to>[\w.]+)~>',
        '$%(nick)s<~Controls.452~>%(piece)s<~Controls.453~><~%(frm)s~>'
        '<~Controls.454~><~%(to)s~>'),
    # card reveal, buffet grammar B
    'reveal_buffet9': _op(
        r'^\$(?P<nick>[^<>\s]+)<~CardBuffet\.9~>reveal (?P<piece>\d+) '
        r'(?P<hash>[A-F0-9]{40}) (?P<value>\d+)'
        r'<~CardBuffet\.10~><~(?P<frm>[\w.]+)~><~CardBuffet\.11~>\s*',
        '$%(nick)s<~CardBuffet.9~>reveal %(piece)s %(hash)s %(value)s'
        '<~CardBuffet.10~><~%(frm)s~><~CardBuffet.11~>'),
    # deck commitment publication (setup; payload passes through —
    # the AI's own commits come from wotr_crypto when networked play
    # lands, and local play never verifies them)
    'deck_commit': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Controls\.(?P<key>479|464)~>'
        r'(?P<payload>[A-F0-9 ]+?) ?',
        '$%(nick)s<~Controls.%(key)s~>%(payload)s'),
    # deck commitment, plaintext grammar ('sendTokens SIDE.hash ...')
    'send_tokens': _op(
        r'^\$(?P<nick>[^<>\s]+) sendTokens (?P<payload>\S.*?) ?',
        '$%(nick)s sendTokens %(payload)s'),
    # no-arg control button (Controls.16)
    'ctrl_16': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Controls\.16~>\s*',
        '$%(nick)s<~Controls.16~>'),
    # combat-card selection
    'combat_card_pick': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Controls\.840~>(?P<piece>\d+)',
        '$%(nick)s<~Controls.840~>%(piece)s'),
    # combat dice roll (a <game> SYSTEM line, not a player $ line): the
    # patched play-vs-AI client normally rolls these locally; when the AI
    # resolves the battle it injects them so the client shows the rolled
    # d6 values.  classify_line never reaches this (it bails on non-$
    # bodies) — the op exists so render() can emit the line.  log_replay
    # only checks for '<~Controls.1~>' (sets _rolls_since_die); it never
    # parses the values, so any faithful roll string round-trips.
    'combat_roll': _op(
        r'^<game> (?P<nick>[^<>\s]+)<~Controls\.1~>\s*(?P<vals>[\d ]+?) ?',
        '<game> %(nick)s<~Controls.1~> %(vals)s'),
    # undo button — parsed (prepass) but the AI never emits it
    'undo': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Controls\.819~>\s*',
        '$%(nick)s<~Controls.819~>'),
    # next-turn button (no $nick prefix in any build)
    'turn_marker': _op(
        r'^<~Game\.7~>(?P<turn>\d+)',
        '<~Game.7~>%(turn)s'),
    # dice-pool bookkeeping quads (FP 726/731, SH 757/762): dice
    # dragged between pool/rolled boxes at turn boundaries
    'dice_book_726': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Controls\.726~>(?P<pieces>[\d ]+?) ?'
        r'<~Controls\.727~><~(?P<frm>[\w.]+)~>'
        r'<~Controls\.728~><~(?P<to>[\w.]+)~>',
        '$%(nick)s<~Controls.726~>%(pieces)s <~Controls.727~>'
        '<~%(frm)s~><~Controls.728~><~%(to)s~>',
        listfields=('pieces',)),
    'dice_book_731': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Controls\.731~>(?P<pieces>[\d ]+?) ?'
        r'<~Controls\.732~><~(?P<frm>[\w.]+)~>'
        r'<~Controls\.733~><~(?P<to>[\w.]+)~>',
        '$%(nick)s<~Controls.731~>%(pieces)s <~Controls.732~>'
        '<~%(frm)s~><~Controls.733~><~%(to)s~>',
        listfields=('pieces',)),
    'dice_book_757': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Controls\.757~>(?P<pieces>[\d ]+?) ?'
        r'<~Controls\.758~><~(?P<frm>[\w.]+)~>'
        r'<~Controls\.759~><~(?P<to>[\w.]+)~>',
        '$%(nick)s<~Controls.757~>%(pieces)s <~Controls.758~>'
        '<~%(frm)s~><~Controls.759~><~%(to)s~>',
        listfields=('pieces',)),
    'dice_book_762': _op(
        r'^\$(?P<nick>[^<>\s]+)<~Controls\.762~>(?P<pieces>[\d ]+?) ?'
        r'<~Controls\.763~><~(?P<frm>[\w.]+)~>'
        r'<~Controls\.764~><~(?P<to>[\w.]+)~>',
        '$%(nick)s<~Controls.762~>%(pieces)s <~Controls.763~>'
        '<~%(frm)s~><~Controls.764~><~%(to)s~>',
        listfields=('pieces',)),
}


# ------------------------------------------- send-to-box routing
# THE CLIENT HAS TWO WAYS TO MOVE A PIECE and we were only ever using
# one.  A hand drag emits Scribe.141/142/143/144; the module's
# send-to-box command emits Controls.281/282/283.  Humans pick between
# them by DESTINATION, and the corpus is emphatic about which:
#
#     destination            drag     C281
#     into 2038..2046          22   11535   -> 99.8% C281
#     everywhere else      262314       0   -> 100.0% drag
#
# 2038..2046 are exactly the OFF-BOARD SUPPLY BOXES - FP/SA
# reinforcements, the two dead piles, the two dice pools, and hunt
# pool/used/removed.  Sending a piece to one of those is a menu
# command in the real client, so a human's log never shows a drag into
# them; ours showed nothing else.  That single difference accounted
# for FOUR of the five box-transition violations the gate reported
# (guide box -> dead, hunt removed -> used, region -> reinforcements,
# current-card -> dead): one root cause wearing four symptoms.
#
# Routed HERE, at render(): a chokepoint cannot be half-applied.
_SUPPLY_BOXES = frozenset('Game.%d' % n for n in range(2038, 2047))
# the six play-on-table card slots (Board1..Board6)
_TABLE_SLOTS = frozenset('Game.%d' % n for n in range(2029, 2035))


def sends_to_box(to):
    """True if this destination takes the send-to-box command."""
    return str(to) in _SUPPLY_BOXES


def _fmt_hand_exit(f):
    s = ('$%(nick)s<~Controls.281~>%(pieces)s <~Controls.282~>'
         '<~%(frm)s~>' % f)
    if f.get('to'):
        s += '<~Controls.283~><~%(to)s~>' % f
    return s


def classify_line(line):
    """(op, fields) for a player command line, else (None, None).
    fields['_trail'] preserves the original tail token so render()
    can reproduce the line byte-for-byte in round-trip mode."""
    raw = line.rstrip('\r\n')
    # chat lines are also emittable (pass / nc / ff declarations);
    # body always ends in ONE space, networked checksum directly after
    cm = re.match(r'^<([^<>]+?)> (.*?)( )?([A-F0-9]{6})?$', raw)
    if cm and cm.group(1) not in ('auto', 'game', 'replay') \
            and not cm.group(1).startswith('~'):
        return 'chat', {'nick': cm.group(1), 'msg': cm.group(2),
                        '_sp': cm.group(3) or '',
                        '_trail': cm.group(4) or ''}
    m = re.search(r'(\s+(?:[A-F0-9]{6}|null)|\s+)$', raw)
    trail = m.group(1) if m else ''
    body = raw[:len(raw) - len(trail)] if trail else raw
    if body.startswith(('$base', '$TEST')) or \
            not body.startswith(('$', '<~Game.7~>')):
        return None, None
    for op, spec in OPS.items():
        m = spec['re'].match(body)
        if not m:
            continue
        f = m.groupdict()
        for lf in spec['lists']:
            f[lf] = f[lf].split()
        f['_trail'] = trail
        return op, f
    return None, None


def render(op, fields, local=True):
    """Build the client line from (op, fields).  local=True stamps the
    non-networked ' null' tail; round-trip passes _trail through."""
    f = dict(fields)
    for k, v in list(f.items()):
        if isinstance(v, list):
            f[k] = ' '.join(v)
    if op == 'directive':
        # a bare client-executor command (!CALL ...) — routed to the
        # client's own handlers (its RNG, its deck); never a game
        # line, never a checksum tail (P1 2026-08-16: live card draws
        # emit through here instead of plan_draw_card picking)
        return f['line']
    if op == 'chat':
        # body normally ends in one space (kept for locally emitted
        # lines); '_sp' preserves the rare space-less system chat
        sp = f.get('_sp', ' ') if f.get('msg') else ''
        # CRITIQUE N-wave: sanitize() strips the apostrophe from
        # Helm's Deep into 'helm s deep' — the corpus chats say
        # 'helms deep' 694:1.  Display-text normalization only.
        if f.get('msg') and 'helm s deep' in f['msg']:
            f = dict(f)
            f['msg'] = f['msg'].replace('helm s deep', 'helms deep')
        return '<%(nick)s> %(msg)s' % f + sp + f.get('_trail', '')
    # THE KILL COMMAND'S ZONE IS A CONSTANT, not a destination.
    # Corpus, 1200 files: Controls.644/645 appears 111,766 times and
    # its zone token is Game.2035 in every single one - there is no
    # second value. Ours was 985 correct and 26 not (18 to Game.2036,
    # 4 to Game.2037, 2 to Game.2034, 2 to Game.2029), produced by
    # four different sites that each passed "where the piece is" as
    # though the token meant a destination.  Enforced here, at the
    # chokepoint, never at the producing sites.
    #
    # `_trail` marks a line that came from classify_line (round-trip),
    # where the bytes must be reproduced exactly as they were read -
    # so parsing a log never rewrites it, only our own emission is
    # normalised.
    # A play-on-table card dies IN ITS BOARD SLOT (USER 2026-08-20,
    # "keep track of that location and kill it there") - and the
    # client's grammar for a kill AT a slot is Controls.296/297, the
    # same (piece, zone) form a hand discard uses.  Raw corpus, 17,692
    # logs: Controls.297 with a Board1..6 token = 10,629 lines
    # ('$Sirrus233<~Controls.296~>167 <~Controls.297~><~Game.2029~>');
    # Controls.645 with a Board token = 0 of 1,721,217 (Game.2035 in
    # every one).  So a 'kill' aimed at a table slot is re-spelt as
    # hand_exit_b at that slot; every other kill keeps the constant.
    if op == 'kill' and '_trail' not in fields:
        if str(f.get('to')) in _TABLE_SLOTS:
            op = 'hand_exit_b'
            f['frm'] = f.pop('to')
        else:
            f['to'] = DICE_USED
    if op == 'hand_exit' or (op == 'move' and sends_to_box(f.get('to'))):
        # a drag into a supply box is not a form humans ever emit; the
        # module's send-to-box command is (see _SUPPLY_BOXES above).
        # The wire grammar is the one 'hand_exit' already renders, so
        # this reuses it rather than adding a second spelling of it -
        # and classify_line reads it straight back, keeping round-trip
        # byte-exact.
        body = _fmt_hand_exit(f)
    elif op == 'kill' and not f['pieces']:
        body = ('$%(nick)s<~Controls.644~><~Controls.645~><~%(to)s~>'
                % f)
    else:
        body = OPS[op]['fmt'] % f
    trail = f.get('_trail')
    if trail is None or trail == '':
        trail = ' null' if local else ''
    return body + trail


# ------------------------------------------------------- L2 planners
# Engine Action -> ordered micro-ops with concrete piece ids, drawn
# from a live LogReplayer.  Imported lazily so L1 round-trip needs no
# engine.  Populated incrementally; plan() raises NotImplementedError
# for uncovered action types so the server can fall back loudly.

# Mordor track boxes, step 0..5 (Mount Doom).  The codes are NOT
# contiguous, which is why this table exists in ONE place (the
# fellowship_move planner and live_server's Mordor choreography share
# it).
MORDOR_TRACK_ZONES = ('Game.524', 'Game.525', 'Game.528', 'Game.529',
                      'Game.532', 'Game.533')

# Set True by live.randomness.ClientRandomness.install() when the
# client-randomness funnels go in.  Two planners below can SOURCE
# randomness themselves (a die face, a drawn card identity): legitimate
# offline, where there is no client to ask, and forbidden in live play
# (USER 2026-08-16: a value the client never produced is injected
# randomness).  The planners cannot see the brain, so the brain tells
# the module.
LIVE_MODE = False

# fixed client zones (constant across the corpus)
FP_DICE_POOL = 'Game.2042'
SH_DICE_POOL = 'Game.2043'
DICE_USED = 'Game.2035'
HUNT_BOX = 'Game.1997'

_zone_by_key = None


def _lr():
    sys.path.insert(0, SRC)
    import log_replay
    return log_replay


def zone_code_for(region_key):
    """Board region key -> client zone code ('Game.NNNN').  Built by
    inverting the parser's own decode tables; sanitize() aligns the
    decoded client names to board keys."""
    global _zone_by_key
    if _zone_by_key is None:
        lr = _lr()
        from board import sanitize
        table = {}
        # AUTH first, then hand-tuned overrides win (they encode the
        # corrections the parser itself prefers)
        for src in (lr._AUTH_CODES, lr.LOCATION_CODES):
            for code, name in src.items():
                k = sanitize(str(name))
                if k:
                    table[k] = code
        _zone_by_key = table
    code = _zone_by_key.get(region_key)
    if code is None:
        raise LookupError('no client zone code for region %r'
                          % region_key)
    return code


def _norm_zone(z):
    from board import sanitize
    return sanitize(str(z)) if z else ''


_initial_positions = None


def _initial():
    """piece id -> DECODED start zone name.  Derived from the corpus
    (each piece's modal first drag origin over 400 games) — the
    client's fixed setup, which never appears in logs, so the live
    replayer's piece_region has no entry until a piece first moves."""
    global _initial_positions
    if _initial_positions is None:
        import json
        lr = _lr()
        with open(os.path.join(HERE, 'initial_positions.json')) as f:
            raw = json.load(f)
        _initial_positions = {p: lr.decode_location(c)
                              for p, c in raw.items()}
    return _initial_positions


def calibrate_initial_from_dump(dump, tracked_pids, replayer=None,
                                cap=12):
    """ONE-SHOT table calibration (USER live 2026-08-17, the erebor
    312/316/317 census hum x576): _initial() is a CORPUS-MODAL guess
    (modal first-drag origin), and it is provably wrong for pieces
    that rarely move from setup - dwarf elite 312 actually starts AT
    Erebor, dwarf regulars 316/317 in FP Reinf, inverted in the
    table.  The client's own board dump IS the module's setup truth
    for every piece that has never moved in-log: adopt it.  Heals
    all consumers at once (census, _pieces_in muster picks - the
    pid-316 resurrection class - and zone fallbacks).

    GUARDS (RED test_adv_event_die_draw exposed the naive version
    adopting 69 pids incl. DICE and POLITICAL markers from a
    mid-game dump): only UNIT-family pieces (regular/elite/leader/
    nazgul/companion/minion - dice/markers move via roll/kill/track
    grammar that piece_region never tracks, so 'never moved in-log'
    is meaningless for them), and if more than *cap* corrections
    would result the dump is stale/mid-game garbage - adopt NOTHING
    (the genuine table-error class is small).  Returns the
    correction list [(pid, old, new)]."""
    init = _initial()
    fixed = []
    for pid, zone in (dump or {}).items():
        p = str(pid)
        if p in tracked_pids:
            continue
        try:
            _nat, ut = _unit_type(p, replayer)
        except Exception:
            continue
        if ut not in ('regular', 'elite', 'leader', 'nazgul',
                      'companion', 'minion'):
            continue
        z = str(zone)
        cur = init.get(p)
        if cur is None or _norm_zone(str(cur)) != _norm_zone(z):
            fixed.append((p, cur, z))
    if len(fixed) > cap:
        print('    [census] initial-zone calibration SKIPPED: %d '
              'contradictions (> cap %d) - dump looks stale or '
              'mid-game, not setup truth' % (len(fixed), cap))
        return []
    for p, _cur, z in fixed:
        init[p] = z
    return fixed


def _pieces_in(replayer, region_key):
    """All piece ids currently in a region: live tracking first,
    initial-setup fallback for pieces that never moved."""
    keys = (region_key, region_key + ' stronghold')
    if region_key in ('sa dead', 'fp dead'):
        # ZONE-NAME ALIAS (USER 2026-08-14): the parser's kill/dkill
        # routing tracks dead pieces under the bare zone name 'Dead',
        # which the pool keys 'sa dead'/'fp dead' must also match.
        # Side discrimination happens in pick_units.
        keys = keys + ('dead', 'sa casualties' if region_key[0] == 's'
                       else 'fp casualties')
    if region_key == 'sa reinf':
        # Shadow units recycle. Native kills return them to reinforcements,
        # but historical logs and manual placement can leave physical pieces
        # in casualties. Include those pieces while retaining their actual
        # source. FP reinforcements never acquire this alias.
        keys = keys + ('sa dead', 'dead', 'sa casualties')
    out = []
    for pid, z in replayer.piece_region.items():
        if _norm_zone(z) in keys:
            out.append(pid)
    tracked = set(replayer.piece_region)
    for pid, z in _initial().items():
        if pid not in tracked and _norm_zone(z) in keys:
            out.append(pid)
    return sorted(out)


def _unit_type(pid, replayer, side_hint=None):
    """(nation, kind) with kind normalized to lowercase string
    ('regular'/'elite'/'leader'/'nazgul'/'companion'/'minion'...) —
    the parser returns UnitKind enums for units, bare strings for
    specials; companions/minions live in their own name tables."""
    lr = _lr()
    pid = str(pid)
    if pid in lr.COMPANIONS:
        return lr.COMPANIONS[pid], 'companion'
    if pid in lr.MINION_PIECES:
        return lr.MINION_PIECES[pid], 'minion'
    nat, ut = lr.get_unit_type_from_piece_id(
        pid, getattr(replayer, 'piece_types', None))
    if ut is None and side_hint:
        # reinforcement-pool ids (318+) only type with a side hint
        nat, ut = lr.get_unit_type_from_piece_id(
            pid, getattr(replayer, 'piece_types', None),
            side_hint=side_hint)
    if ut is not None:
        ut = str(getattr(ut, 'value', ut)).lower()
    return nat, ut


def _side_of_nation(nation):
    n = str(getattr(nation, 'value', nation) or '').lower()
    return 'shadow' if any(s in n for s in
                           ('sauron', 'isengard', 'southron',
                            'easterling')) else 'fp'


def pick_units(replayer, region_key, side, n_regular=0, n_elite=0,
               n_leader=0, n_nazgul=0, nation=None, exclude=(),
               side_hint=None):
    """Choose concrete unit piece ids of the requested composition
    currently sitting in region_key (per replayer.piece_region; both
    the base region and its stronghold box count as 'in' it)."""
    want = [('regular', n_regular), ('elite', n_elite),
            ('leader', n_leader), ('nazgul', n_nazgul)]
    pool = _pieces_in(replayer, region_key)
    chosen = []
    for kind, n in want:
        got = 0
        for pid in pool:
            if pid in chosen or pid in exclude or got >= n:
                continue
            nat, ut = _unit_type(pid, replayer, side_hint=side_hint)
            if ut != kind:
                continue
            # nazgul are shadow by definition (their nation slot is
            # unreliable); everything else classifies via nation
            if kind != 'nazgul' and _side_of_nation(nat) != side:
                continue
            if nation is not None and kind != 'nazgul':
                have = str(getattr(nat, 'value', nat) or '').lower()
                wantn = str(getattr(nation, 'value', nation)).lower()
                if have.replace(' ', '_') != wantn.replace(' ', '_'):
                    continue
            chosen.append(pid)
            got += 1
        if got < n:
            raise LookupError('need %d %s (%s) in %r, found %d'
                              % (n, kind, side, region_key, got))
    return chosen


def die_piece_for(replayer, side_name, die_idx):
    """The client die piece id for an engine die index.

    Mid-turn engine dice-list REMOVALS (The Day Without Dawn class:
    event_effects list.remove() of the unused WILL die) shift every
    later engine index down one while the parser's die_id_map keeps
    static registration indexes — attempt-7 game-3 dragged pid 211
    for the engine's 212-die and dead 213 for the 358-die.  The
    bridge stamps the removal points on the replayer
    (_die_idx_shift: side_name -> [removed registration indexes,
    cleared at the next action roll]); un-shift through them here so
    every mapping call site inherits the correction."""
    # THE Nth LIVE REGISTRATION SLOT, not compounding arithmetic:
    # walking the live slots is correct by construction and cannot run
    # off the end; when it runs out the roster genuinely disagrees and
    # the error says so with the whole picture.
    _dead = set((getattr(replayer, '_die_idx_shift', {}) or {}).get(
        side_name) or ())
    _by_reg = {}
    for pid, (s, idx) in replayer.die_id_map.items():
        if getattr(s, 'name', str(s)) == side_name:
            _by_reg[idx] = pid
    _live = [ix for ix in sorted(_by_reg) if ix not in _dead]
    if 0 <= die_idx < len(_live):
        pid = _by_reg[_live[die_idx]]
        return '%03d' % pid if isinstance(pid, int) else str(pid)
    raise LookupError(
        'no die piece for %s die %d: %d live slot(s) %s of registered '
        '%s, dead regs %s'
        % (side_name, die_idx, len(_live), _live, sorted(_by_reg),
           sorted(_dead)))


# face -> client roll value.  The Java client encodes each die's face
# as an ABSOLUTE index with a per-die base offset (decompiled b_0/c):
# FP action die (class i_0, base 0) faces 0..4; SHADOW action die
# (class bc_0, base 10) faces 10..15.  Injecting a Shadow roll with the
# relative 0..5 makes the client compute value-10 = negative and throw
# ArrayIndexOutOfBounds, so Shadow MUST carry the +10 base.  log_replay
# tolerantly parses BOTH ranges, so this stays round-trip compatible.
_FP_FACE_VALUE = {'WILL_OF_WEST': 0, 'CHARACTER': 1, 'EVENT': 2,
                  'MUSTER': 3, 'MUSTER_ARMY': 4}
_SH_FACE_VALUE = {'MUSTER': 10, 'MUSTER_ARMY': 11, 'ARMY': 12,
                  'CHARACTER': 13, 'EVENT': 14, 'EYE': 15}

# base action-die piece ids per side, then BONUS dice from special
# characters (FP: Aragorn-crowned + Gandalf-the-White = 358,359;
# SH: Witch-king, Saruman, Mouth = 360,361,362).  Bonus dice come from
# the reinforcement/character area and are rolled once the engine's pool
# grows past its base count (calculate_bonus_dice); log_replay registers
# whatever id first appears in a roll at the next pool index.
_FP_DIE_IDS = ['210', '211', '212', '213', '358', '359']
_SH_DIE_IDS = ['099', '100', '101', '102', '103', '104', '105',
               '360', '361', '362']


def plan_roll(replayer, nick, side_name, state, rng):
    """Roll THIS side's unrolled action dice: pick a fair face per die
    from its own face multiset, emit one Controls.345/346 line.  `rng`
    is a random.Random (the brain owns it so seeding is explicit).

    LIVE PLAY MAY NOT REACH HERE.  This picks the faces itself, so in
    live play it would be the bridge inventing dice the client never
    rolled.  The live head asks the client instead
    (randomness.directives_for -> '!CALL roll_action_dice', the
    client's own A.k() handler rolls); NO shipped path calls this.
    The HALT stays - if it really is dead the halt costs nothing, and
    if it is not, the halt says so immediately instead of letting
    self-generated dice pass for client dice.
    """
    if LIVE_MODE or rng is None:
        raise RuntimeError(
            'LIVE HALT: plan_roll would invent action-die faces in '
            'live play. Action dice come from the client '
            "(!CALL roll_action_dice -> the client's own roller) and "
            'the engine INGESTS them; it never sources them. No '
            'shipped path calls plan_roll - the live head asks via '
            'randomness.directives_for.')
    from game_state import Side
    side = Side.FREE_PEOPLES if side_name == 'FREE_PEOPLES' \
        else Side.SHADOW
    dice = state.dice[side]
    unrolled = [i for i, d in enumerate(dice)
                if d.current is None and not d.used]
    if not unrolled:
        raise LookupError('no unrolled %s dice to roll' % side_name)
    base_ids = _FP_DIE_IDS if side == Side.FREE_PEOPLES else _SH_DIE_IDS
    vmap = _FP_FACE_VALUE if side == Side.FREE_PEOPLES else _SH_FACE_VALUE
    ids, vals = [], []
    for i in unrolled:
        if i >= len(base_ids):
            raise LookupError('die index %d beyond known %s piece ids'
                              % (i, side_name))
        face = rng.choice(dice[i].faces)          # fair: from real die
        fname = getattr(face, 'name', str(face))
        if fname not in vmap:
            raise LookupError('no roll value for face %r' % fname)
        ids.append(base_ids[i])
        vals.append(str(vmap[fname]))
    return ('dice_roll', {'nick': nick, 'dice': ids, 'values': vals})


def plan_die_spend(replayer, nick, side_name, die_idx):
    pid = die_piece_for(replayer, side_name, die_idx)
    if side_name == 'SHADOW':
        return ('die_spend_sh', {'nick': nick, 'die': pid,
                                 'frm': SH_DICE_POOL, 'to': DICE_USED})
    return ('die_spend_fp', {'nick': nick, 'die': pid,
                             'frm': FP_DICE_POOL, 'to': DICE_USED})


# reinforcement pools / dead piles (fixed client zones)
FP_POOL = 'Game.2038'
SH_POOL = 'Game.2040'
FP_DEAD = 'Game.2039'
SH_DEAD = 'Game.2041'

# political track: internal box b (0 passive .. 3 at-war) maps to
# VASSAL zone Game.(2000+b); each advance step has its OWN walk
# grammar in the client (the corpus shows 91: 2000->2001,
# 96: 2001->2002, 101: 2002->2003)
_POLITICAL_STEP_OPS = ('political_91', 'political_96', 'political_101')


def _nation_value(n):
    return str(getattr(n, 'value', n) or '').lower().replace(' ', '_')


def plan_advance_political(replayer, nick, side_name, params):
    """advance_political params: nation.  One walk line of the
    nation's political marker (256-263) to the next track box."""
    lr = _lr()
    nval = _nation_value(params.get('nation'))
    pid = next((p for p, n in lr.POLITICAL_MARKERS.items()
                if _nation_value(n) == nval), None)
    if pid is None:
        raise LookupError('no political marker for nation %r'
                          % (params.get('nation'),))
    # engine actions carry no box (state = PRE-advance value); parser
    # records fire mid-execution and carry the pre-advance 'from'
    box = params.get('from')
    if box is None:
        box = next((v for k, v in
                    replayer.state.political.boxes.items()
                    if _nation_value(k) == nval), None)
    if box is None or not 0 <= int(box) <= 2:
        raise LookupError('nation %r cannot advance (box=%r)'
                          % (nval, box))
    box = int(box)
    return (_POLITICAL_STEP_OPS[box],
            {'nick': nick, 'piece': pid,
             'frm': 'Game.%d' % (2000 + box),
             'to': 'Game.%d' % (2001 + box)})


def plan_muster(replayer, nick, side_name, params, kind):
    """One recruit drag: a pool piece of (nation, kind) from the side's
    reinforcement zone onto the settlement.  kind: 'regular' | 'elite'
    | 'leader' (Shadow leader-recruits are nazgul)."""
    shadow = side_name == 'SHADOW'
    side = 'shadow' if shadow else 'fp'     # pick_units side vocab
    kw = {'n_regular': 0, 'n_elite': 0, 'n_leader': 0, 'n_nazgul': 0}
    if kind == 'leader' and shadow:
        kw['n_nazgul'] = 1
    else:
        kw['n_' + kind] = 1
    # Shadow casualties are reusable. Native kills return army units to
    # reinforcements; historical/manual physical casualties can still be
    # selected, provided the emitted source matches the piece's location.
    #
    # FP DEAD ARE PERMANENT (rules; USER precedent, comms 50 Part 35
    # Class A: "FP reinf explicitly NEVER gets this alias"): the FP
    # branch has ONE pool, and a genuine physical shortfall raises
    # LookupError - never a resurrection.
    pools =[('sa reinf', SH_POOL)] if shadow else \
        [('fp reinf', FP_POOL)]
    if shadow:
        pools.append(('sa dead', SH_DEAD))
    # SIEGE (USER 2026-08-14): reinforcements mustered in a BESIEGED
    # settlement join the GARRISON — the drag targets the stronghold
    # sub-box, not the field zone (same zone retreat-into-siege uses).
    # The server injects '_besieged_garrison' from the ENGINE state
    # (the engine keeps garrison+besieger co-located under one key, so
    # only the emission knows the client's two zones apart).
    _dest9 = _norm_zone(params['region'])
    if params.get('_besieged_garrison'):
        try:
            zone_code_for(_dest9 + ' stronghold')
            _dest9 = _dest9 + ' stronghold'
        except LookupError:
            pass
    last = None
    for pool_key, pool_zc in pools:
        try:
            pieces = pick_units(replayer, pool_key, side,
                                nation=params.get('nation'),
                                side_hint=side, **kw)
            # SOURCE FROM THE PIECE, NOT THE POOL: `_pieces_in('sa
            # reinf')` aliases in the Dead pile (SH casualties
            # recycle), so the piece's OWN tracked zone is the
            # discriminating source; the nominal pool zone is the
            # fallback for an untracked piece (the pid-316
            # resurrection class).
            if pieces:
                _pz9 = _piece_zone_code(replayer, pieces[0], default=None)
                if _pz9 and _pz9 != pool_zc:
                    _same9 = all(
                        _piece_zone_code(replayer, _p9, default=None)
                        == _pz9 for _p9 in pieces)
                    if _same9:
                        pool_zc = _pz9
                    else:
                        raise LookupError(
                            'muster pick spans more than one source '
                            'zone (%s) - refusing to emit one drag '
                            'with a single frm'
                            % ' '.join(sorted({
                                str(_piece_zone_code(replayer, _p9,
                                                     default='?'))
                                for _p9 in pieces})))
            return ('move', {'nick': nick, 'pieces': pieces,
                             'frm': pool_zc,
                             'to': zone_code_for(_dest9)})
        except LookupError as e:
            last = e
    raise last


def plan_muster_minion(replayer, nick, side_name, params):
    """muster_minion params: minion name, region.  Drag the minion
    piece (253/254/255) from wherever the client holds it (SA Reinf
    for fresh musters) onto the region."""
    lr = _lr()
    name = str(params.get('minion', ''))
    pid = next((p for p, n in lr.MINION_PIECES.items()
                if str(n).lower() == name.lower()), None)
    if pid is None:
        raise LookupError('unknown minion %r' % (name,))
    zone = replayer.piece_region.get(pid) or _initial().get(pid)
    if not zone:
        frm = SH_POOL          # never seen -> still in the pool
    else:
        frm = zone_code_for(_norm_zone(zone))
    return ('move', {'nick': nick, 'pieces': [pid], 'frm': frm,
                     'to': zone_code_for(_norm_zone(params['region']))})


# card decks/hands: fixed client zones + fixed piece-id membership
# (pids 106-157 FP / 158-209 SA, typed C/S/X in log_replay's tables;
# X = expansion-only cards, absent from base-game decks)
_DECK_ZONES = {('fp', 'character'): 'Game.2004',
               ('fp', 'strategy'): 'Game.2005',
               ('sh', 'character'): 'Game.2006',
               ('sh', 'strategy'): 'Game.2007'}
_HAND_ZONES = {'fp': 'Controls.45', 'sh': 'Controls.46'}


def plan_draw_card(replayer, nick, side_name, params, rng=None):
    """draw_card params: deck='character'|'strategy'.  Picks a random
    card piece still in that deck (fixed membership minus every pid
    ever seen leaving a deck — the server stashes that scan on the
    replayer as ai_drawn_pids).  Offline the client has no committed
    deck order we could follow, and only this side draws from its own
    decks, so a uniform pick over the remainder IS the fair shuffle.

    LIVE PLAY MAY NOT REACH HERE - guarded at entry, below."""
    # SAME RULE AS plan_roll, guarded at ENTRY (before the deck and
    # ledger lookups, which raise LookupError first - a guard behind
    # them is not a guard).
    if LIVE_MODE:
        # THE FUNNEL, not a halt (P1, USER 2026-08-16): the CLIENT
        # chooses the card - this returns the !CALL directive; the
        # client draws with its own RNG and the next tick's log replay
        # ingests the card it named.  Same grammar as the turn-start
        # draws.  plan_use_token's draw half heals through here too.
        _tag9 = 'sa' if side_name == 'SHADOW' else 'fp'
        _dk9 = ('strat' if str(params.get('deck', 'character')
                               ).lower().startswith('strat')
                else 'char')
        return ('directive', {'line': '!CALL draw_%s_%s'
                              % (_tag9, _dk9)})
    lr = _lr()
    side = 'sh' if side_name == 'SHADOW' else 'fp'
    deck = 'strategy' if str(params.get('deck', 'character')
                             ).lower().startswith('strat') \
        else 'character'
    want = 'C' if deck == 'character' else 'S'
    if side == 'fp':
        pids = [106 + i for i, t in enumerate(lr._FP_PID_TYPES)
                if t == want]
    else:
        pids = [158 + i for i, t in enumerate(lr._SA_PID_TYPES)
                if t == want]
    # THE LEDGER MUST EXIST (PERMANENT item 7): a missing ai_drawn_pids
    # is "we cannot know", never "no cards drawn yet" (an empty default
    # would return the same pid forever).
    if not hasattr(replayer, 'ai_drawn_pids'):
        raise LookupError(
            'cannot plan a %s %s draw: the replayer has no '
            'ai_drawn_pids ledger, so every draw would return the '
            'same piece' % (side, deck))
    drawn = {str(p) for p in (replayer.ai_drawn_pids or ())}
    left = [p for p in pids if str(p) not in drawn]
    if not left:
        raise LookupError('no cards left in %s %s deck' % (side, deck))
    # (live guard is at function entry - see the top of this function)
    pid = rng.choice(left) if rng is not None else left[0]
    return ('card_draw', {'nick': nick, 'piece': str(pid),
                          'frm': _DECK_ZONES[(side, deck)],
                          'to': _HAND_ZONES[side]})


def plan_army_move(replayer, nick, side_name, params):
    """army_move params: from/to board keys, n_regular/n_elite,
    take_leaders or n_leader, optional nation."""
    frm_key = _norm_zone(params.get('from'))
    to_key = _norm_zone(params.get('to'))
    side = 'shadow' if side_name == 'SHADOW' else 'fp'
    # FLD-5b (par5b_g5#13): the engine's LA-8 per-count variant rides
    # key 'n_leaders' (plural, ai.py:5485); reading only the legacy
    # singular left the explicit count invisible, so take_leaders
    # swept ALL leader pieces while the executor moved exactly
    # int(n_leaders) (ai.py move_leaders) - render == engine.
    n_l = int(params.get('n_leaders',
                         params.get('n_leader', 0)) or 0)
    n_naz = int(params.get('n_nazgul', 0) or 0)
    if params.get('take_leaders') and not n_l \
            and params.get('n_nazgul') is None:
        # take ALL plain-leader pieces of this side at the source; for
        # Shadow that includes NAZGUL (typed 'nazgul' - the engine
        # moves them under take_leaders) - UNLESS the action carries an
        # EXPLICIT 'n_nazgul' split: the engine then passes
        # move_leaders=None to move_army, and its take() treats None
        # as ZERO (the documented 1.150 quirk), so NO plain leader
        # moves either.  Corpus logs never carry n_nazgul
        # (engine-enumeration variant only), so the L1/L2 roundtrip
        # gates are byte-identical either way.
        for pid in _pieces_in(replayer, frm_key):
            nat, ut = _unit_type(pid, replayer, side_hint=side)
            if ut == 'leader' and _side_of_nation(nat) == side:
                n_l += 1
            elif ut == 'nazgul' and side == 'shadow':
                n_naz += 1
    chars = [str(c) for c in params.get('characters', ())]
    if params.get('take_leaders') and params.get('n_nazgul') is None \
            and params.get('n_leaders') is None:
        # FLD-5b: an explicit n_leaders pins the leader FIGURE count
        # precisely (ai.py 1.211 comment) - the character sweep only
        # rides the plain take-all form
        # the engine's take_leaders also carries CHARACTER leaders: a
        # minion (Witch-king / Mouth - Saruman never leaves Orthanc)
        # or a separated companion leading the army.  With an EXPLICIT
        # n_nazgul the engine moves ONLY those nazgul (move_leaders
        # None = zero - see above), so minions/companions stay home.
        for pid in _pieces_in(replayer, frm_key):
            nat, ut = _unit_type(pid, replayer, side_hint=side)
            if ut == 'minion' and side == 'shadow' and pid != '254' \
                    and pid not in chars:
                chars.append(pid)
            elif ut == 'companion' and side == 'fp' \
                    and pid not in chars:
                chars.append(pid)
    mbn = params.get('moved_by_nation')
    if mbn:
        # the ENGINE's exact per-(nation, kind) moved composition (a
        # nation-blind pick would ship the wrong nation's regular from
        # a MIXED army)
        pieces = []
        used = set(chars)
        for key9, n9 in mbn.items():
            natv9, kind9 = str(key9).split(':', 1)
            if kind9 not in ('regular', 'elite'):
                continue
            got9 = pick_units(replayer, frm_key, side,
                              nation=natv9, exclude=list(used),
                              side_hint=side,
                              **{'n_%s' % kind9: int(n9)})
            pieces += got9
            used.update(got9)
        got9 = pick_units(replayer, frm_key, side, n_leader=n_l,
                          n_nazgul=n_naz, exclude=list(used),
                          side_hint=side)
        pieces += got9
    else:
        pieces = pick_units(
            replayer, frm_key, side,
            n_regular=int(params.get('n_regular', 0) or 0),
            n_elite=int(params.get('n_elite', 0) or 0),
            n_leader=n_l,
            n_nazgul=n_naz,
            nation=params.get('nation'), exclude=chars,
            side_hint=side)
    # SIEGE: units CAN move in to join a besieged garrison (game_logic:
    # the 5-unit siege cap applies only to units joining the garrison
    # side inside a besieged stronghold), so the arrival zone is the
    # garrison question.  victor_advance delegates here.
    return ('move', {'nick': nick, 'pieces': pieces + chars,
                     'frm': zone_code_for(frm_key),
                     'to': _garrison_dest(replayer, to_key,
                                          side_name)})


# fellowship / guide boxes and die-change face letters
FELLOWSHIP_BOX = 'Game.1998'
GUIDE_BOX = 'Game.1999'
_FACE_LETTER = {'ARMY': 'A', 'CHARACTER': 'C', 'MUSTER': 'M',
                'MUSTER_ARMY': 'H', 'EVENT': 'P', 'EYE': 'E',
                'WILL_OF_WEST': 'W'}   # parser DIE_CHANGE_LETTERS inverse

# pendings that are IMPLICIT in the client: a human simply does not do
# the thing; the parser infers the decision from subsequent lines.  The
# planner returns ZERO lines for them (the server treats [] from these
# as a made decision, not a failure).
IMPLICIT_ACTIONS = frozenset((
    'combat_cease', 'combat_no_retreat', 'combat_continue',
    'victor_stay', 'siege_no_press', 'siege_continue',
    'rearguard_none', 'followup_done', 'casualty_preserve_units',
    'casualty_preserve_elites', 'hunt_gollum_no_reveal', 'card_option',
))


def _companion_pid(name):
    lr = _lr()
    want = str(name).strip().lower()
    for p, n in lr.COMPANIONS.items():
        if str(n).strip().lower() == want:
            return p
    # DUAL-FACE figures change NAME mid-game while the pid stays
    # constant (240 Grey->White, 241 Strider->Aragorn); the parser's
    # COMPANIONS table renames 241 at crowning but never 240, so every
    # alias face must resolve to its pid.
    for p, faces in getattr(lr, 'PIECE_NAME_ALIASES', {}).items():
        for f in faces:
            if str(f).strip().lower() == want:
                return p
    # FLA-6 (par5 g3#5): MINIONS resolve too - the named echo
    # performer (human_gestures.play_human_echo_kills) resolves every
    # named row through this ONE table, and 'Witch King' raised here
    # so the owed drag was never auto-performed.  Side attribution
    # stays with emission_card._named_pid_side (which checks the
    # tables itself, never this function's success).
    for p, n in getattr(lr, 'MINION_PIECES', {}).items():
        if str(n).strip().lower() == want:
            return str(p)
    raise LookupError('unknown companion %r' % (name,))


def _guide_departure_zone(replayer, pid):
    """Client zone code a companion LEAVES the fellowship from: wherever
    the client holds the piece (Guide Box or Fellowship Box both
    occur).  A piece tracked in FP Casualties cannot be the source of
    a departure (stale tracking): fall back to its setup zone, else the
    Guide Box."""
    frm = _piece_zone_code(replayer, pid, default=GUIDE_BOX)
    if frm != FP_DEAD:
        return frm
    frm = GUIDE_BOX
    iz = _initial().get(str(pid))
    if iz:
        try:
            frm = zone_code_for(_norm_zone(str(iz)))
        except LookupError:
            frm = GUIDE_BOX
    return frm


_zone_name_to_code = None


def _holding_zone_code(replayer, pid, default=None):
    """Resolve where the replayer holds a piece to a client zone code,
    inverting the FULL location table (regions AND off-map boxes like
    FP Dead / FP Card Play / Reinf).  Falls back to `default` if the
    piece is untracked or sits somewhere with no invertible code."""
    global _zone_name_to_code
    if _zone_name_to_code is None:
        lr = _lr()
        from board import sanitize
        m = {}
        for code, name in lr.LOCATION_CODES.items():
            k = sanitize(str(name))
            if k and k not in m:          # first (lowest) code wins
                m[k] = code
        _zone_name_to_code = m
    tracked = replayer.piece_region.get(str(pid)) \
        or _initial().get(str(pid))
    if tracked:
        code = _zone_name_to_code.get(_norm_zone(tracked))
        if code:
            return code
    return default


def _native_kill_zone(replayer, pid, default='Dead'):
    """Java's destination for a newly killed piece, not a historical pile.

    Shadow army units and ordinary Nazgul return to reinforcements. Named
    minions, FP pieces and other tokens retain the casualty/dead handling.
    All live projection paths must agree before the next board dump arrives.
    Never use this to reinterpret an explicitly observed casualty location.
    """
    nation, kind = _unit_type(str(pid), replayer)
    if kind == 'nazgul' or (kind in ('regular', 'elite')
                           and _side_of_nation(nation) == 'shadow'):
        return 'sa reinf'
    return default


def _piece_zone_code(replayer, pid, default=None):
    """Client zone code where the replayer believes a piece sits."""
    slot = (getattr(replayer, 'shadow_stronghold_regions', None) or {}).get(str(pid))
    if slot and _norm_zone(replayer.piece_region.get(str(pid))) == slot[0] + ' stronghold':
        return slot[1]
    zone = replayer.piece_region.get(str(pid)) \
        or _initial().get(str(pid))
    if not zone:
        if default:
            return default
        raise LookupError('no known zone for piece %r' % (pid,))
    zk = _norm_zone(zone)
    if zk == 'dead':
        # BARE-'Dead' TRACKING: the parser's kill/dkill routing stores
        # casualties under the bare zone name 'Dead', which the client
        # zone table has no key for; the pid decides WHICH dead pile
        # (FP pieces -> FP Casualties, Shadow -> SA Casualties).
        nat, _ut9 = _unit_type(pid, replayer)
        if _ut9 in ('nazgul', 'minion'):
            return SH_DEAD
        if _ut9 == 'companion':
            return FP_DEAD
        return FP_DEAD if _side_of_nation(nat) == 'fp' else SH_DEAD
    return zone_code_for(zk)


def _side_pieces_in(replayer, region_key, side):
    """Every piece of a side in a region: units, leaders, nazgul,
    companions (fp) / minions (sh).  side: 'fp' | 'shadow'."""
    out = []
    for pid in _pieces_in(replayer, region_key):
        nat, ut = _unit_type(pid, replayer, side_hint=side)
        if ut in ('regular', 'elite', 'leader'):
            if _side_of_nation(nat) == side:
                out.append(pid)
        elif ut in ('nazgul', 'minion'):
            if side == 'shadow':
                out.append(pid)
        elif ut == 'companion':
            if side == 'fp':
                out.append(pid)
    return out


def plan_army_attack(replayer, nick, side_name, params):
    """No declare grammar exists: humans announce the target in chat
    ('X from Y' / bare target for a sortie) and the battle unfolds as
    pending-combat decisions."""
    frm = _norm_zone(params.get('from'))
    to = _norm_zone(params.get('to'))
    msg = to if frm == to else '%s from %s' % (to, frm)
    return ('chat', {'nick': nick, 'msg': msg})


# ================================================= THE GARRISON QUESTION
# "A piece is arriving at region R.  Does it land on the FIELD zone or
# inside the stronghold sub-box?"  Answer: the sub-box iff R is under
# siege AND this piece's side holds the garrison.
#
# WHICH STATE ANSWERS.  These take `state` explicitly rather than
# reaching for one, because the honest answer differs by caller: the
# server holds the ENGINE's state and must use it (engine is master,
# client renders), while an emitter planner only ever holds the
# replayer's client-side mirror.  Passing it in lets both ask the SAME
# function without pretending the emitter has access it does not have.
#
# NO BLANKET except: a bare `except Exception: pass` turns
# "combat.garrison_side_for is broken" into "the piece quietly lands in
# the field zone".  Only the genuinely expected failure is caught: a
# region with no ' stronghold' sub-box defined.
_SIDE_ALIASES = {'fp': 'FREE_PEOPLES', 'free_peoples': 'FREE_PEOPLES',
                 'freepeoples': 'FREE_PEOPLES', 'free': 'FREE_PEOPLES',
                 'shadow': 'SHADOW', 'sh': 'SHADOW'}


def canon_side(side):
    """One spelling for a side, whatever vocabulary the caller uses."""
    s = str(getattr(side, 'name', side) or '').strip()
    return _SIDE_ALIASES.get(s.lower(), s.upper())


def _battle_region_of(pc, side_name):
    """The region `side_name` fights FROM in an open pending_combat:
    defender_region when it is the defender, else attacker_region
    (the engine's own keys; a pending never carries 'region')."""
    if not pc:
        return None
    if canon_side(pc.get('defender_side')) == canon_side(side_name):
        return pc.get('defender_region')
    return pc.get('attacker_region')


def garrison_side_of(state, region):
    """'FREE_PEOPLES' | 'SHADOW' for the garrison holding a BESIEGED
    stronghold, or None when `region` is not under siege at all."""
    reg = _norm_zone(str(region))
    bs = {_norm_zone(str(b)) for b in
          (getattr(state, 'besieged_strongholds', set()) or set())}
    if reg not in bs:
        return None
    import combat as _cmb
    g = _cmb.garrison_side_for(state, reg)
    return canon_side(g) if g is not None else None


def garrison_arrival_zone(state, region, side):
    """THE arrival-zone answer.  Every arrival path calls this."""
    reg = _norm_zone(str(region))
    held = garrison_side_of(state, reg)
    if held is not None and canon_side(side) == held:
        try:
            return zone_code_for(reg + ' stronghold')
        except LookupError:
            pass          # region has no stronghold sub-box
    return zone_code_for(reg)


def _garrison_dest(replayer, to_key, side_name):
    """Emitter-side shorthand: ask with the replayer's state."""
    return garrison_arrival_zone(replayer.state, to_key, side_name)


def plan_companion_move(replayer, nick, side_name, params):
    lr = _lr()
    frm = _norm_zone(params['from'])
    names = params.get('companions')
    present = _pieces_in(replayer, frm)
    if names:
        # every NAMED figure resolves through the alias-aware lookup
        # (a crowned Aragorn / Gandalf the White keep their pids while
        # the static table still says Strider / the Grey) and must be
        # in the source region: a drag that silently drops a named
        # figure is never emitted
        pids = []
        for n in names:
            p = _companion_pid(n)
            if p not in present:
                raise LookupError('companion %r (piece %s) is not in %r'
                                  % (n, p, frm))
            pids.append(p)
    else:
        pids = [p for p in present if p in lr.COMPANIONS]
    if not pids:
        raise LookupError('no companions to move in %r' % (frm,))
    # SIEGE: a companion arriving at a BESIEGED friendly stronghold
    # joins the garrison INSIDE - the field zone belongs to the
    # besieger (the ONE question, garrison_arrival_zone; same as
    # plan_nazgul_move below).
    return ('move', {'nick': nick, 'pieces': pids,
                     'frm': zone_code_for(frm),
                     'to': _garrison_dest(replayer,
                                          _norm_zone(params['to']),
                                          'FREE_PEOPLES')})


def plan_nazgul_move(replayer, nick, side_name, params):
    frm = _norm_zone(params['from'])
    naz = [p for p in _pieces_in(replayer, frm)
           if _unit_type(p, replayer, side_hint='shadow')[1] == 'nazgul']
    have_wk = '253' in _pieces_in(replayer, frm)
    # engine actions say 'count'; parser records say 'moved' (which can
    # be a composition dict counting the Witch-king among the riders)
    n = params.get('count', params.get('moved'))
    if isinstance(n, dict):
        n = sum(v for v in n.values() if isinstance(v, int))
    if n is None:
        pids = list(naz)
    else:
        pids = naz[:int(n)]
        if len(pids) < int(n) and have_wk:
            pids.append('253')     # humans drag the WK as 'a nazgul'
        if len(pids) < int(n):
            raise LookupError('need %d nazgul in %r, found %d'
                              % (int(n), frm, len(pids)))
    if params.get('include_wk') and '253' not in pids:
        if not have_wk:
            raise LookupError('Witch-king not in %r' % (frm,))
        pids.append('253')
    if not pids and have_wk:
        pids = ['253']             # a lone Witch-king ride
    if not pids:
        raise LookupError('no nazgul in %r' % (frm,))
    _to9 = _norm_zone(params['to'])
    # SIEGE: nazgul arriving at a BESIEGED friendly stronghold join the
    # garrison INSIDE the sub-box - the field zone belongs to the
    # besieger (the ONE question; the side is a constant, the question
    # is not).
    _toz9 = garrison_arrival_zone(replayer.state, _to9, 'SHADOW')
    groups = {}
    for pid in pids:
        source = _piece_zone_code(replayer, pid, default=zone_code_for(frm))
        groups.setdefault(source, []).append(pid)
    return [('move', {'nick': nick, 'pieces': pieces, 'frm': source, 'to': _toz9})
            for source, pieces in groups.items()]


def plan_companion_separate(replayer, nick, side_name, params):
    """Separate a companion to a region.  The guide reassignment (when
    the guide is the one leaving) is IMPLICIT in the client -- humans
    emit no extra line -- so the parser auto-reassigns it (log_replay
    change_guide, mirroring the engine); we emit only the move."""
    # THE GROUP FORM IS THE COMMON ONE: ai.py builds a SINGLE separation
    # as params['companion'] and a group separation (the rulebook's
    # normal case - companions of equal level leave together) as
    # params['companions'] (a LIST); both shapes are read.
    names = params.get('companions')
    if names is None:
        names = [params['companion']]
    elif isinstance(names, str):
        names = [names]
    pids = [_companion_pid(n) for n in names]
    to = _garrison_dest(replayer, _norm_zone(params['to_region']),
                        'FREE_PEOPLES')
    # figures can sit in different zones (one already separated, one
    # still in the box) - one drag per source zone, corpus order
    groups = []
    for pid in pids:
        frm = _piece_zone_code(replayer, pid, default=FELLOWSHIP_BOX)
        for g in groups:
            if g[0] == frm:
                g[1].append(pid)
                break
        else:
            groups.append((frm, [pid]))
    # SIEGE: a companion separated into a BESIEGED friendly stronghold
    # joins the garrison, not the besiegers (same question as
    # plan_companion_move).
    return [('move', {'nick': nick, 'pieces': list(gp), 'frm': gf,
                      'to': to}) for gf, gp in groups]


def plan_eye_bank(replayer, nick, die_piece):
    """The physical consequence of a Shadow die BECOMING an Eye: the
    die banks in the Hunt Box.  ONE implementation, two callers - the
    elven_ring ACTION planner below and the server's card-effect
    die-face delta (SIBLING ENUMERATION, offline rig 2026-08-17: the
    ring fix covered only the action; card effects that set a die to
    the Eye - The Lidless Eye, Give it to Uss!, the Nazgul-search
    family - reach the same physical state through the other door and
    were emitting nothing, leaving the parser's die rolled-and-free
    while the engine had it used and banked)."""
    _frm = _piece_zone_code(replayer, die_piece, default=SH_DICE_POOL)
    return ('move', {'nick': nick, 'pieces': [die_piece],
                     'frm': _frm, 'to': HUNT_BOX})


RING_ACTIVE_SLOTS = ('Game.2008', 'Game.2009', 'Game.2010')
RING_USED_SLOTS = ('Game.2011', 'Game.2012', 'Game.2013')


def pick_fp_ring_walk(replayer, paid=None, used=None):
    """WHICH FP elven ring marker walks, OUT OF which slot, INTO which
    slot - read from the BOARD, never computed from a count.

    USER RULING 2026-08-19, verbatim: *"which ring is irrelevant as
    long as it is cleanly tracked."*  The three rings are functionally
    interchangeable, so WHICH ring is chosen is not a rule question and
    no preference order is implied here.  What matters is that the
    client walks a NAMED PHYSICAL MARKER: naming a ring that is not in
    the named slot emits a gesture that CANNOT BE PERFORMED, and the
    board and the engine then disagree with nothing able to heal them.

    THE DEFECT THIS REPLACES (BRIDGE-11 2026-08-19, driven, RED):

        board : 249 Used Ring 2 | 250 Nenya Slot | 251 Used Ring 1
        count : fp_elven_rings = 1  ->  used = 2
        emits : piece 251 out of Game.2010 (Narya Slot)
        truth : 251 is in Used Ring 1; only 250 is still in a slot

    Both the marker and its source slot were `('249','250','251')[used]`
    / `'Game.%d' % (2008+used)` arithmetic off a state counter, and the
    `_ring_paid` ledger - the only guard - did NOT refuse: it catches a
    ring walked twice by THIS caller, never a ring that left its slot
    by any other route.

    THE SIBLING WAS ALREADY CURED THIS WAY, which is why this is a
    defect and not a design choice.  The SHADOW ring-dispose path had
    the identical disease (auto69 game 5: pid 251 disposed twice, 250
    orphaned in Used Ring 2 forever, the divergence unhealable) and was
    fixed to select from the board with, in its own test's words, "a
    ledger a ring can leave exactly once, and NO arithmetic fallback".
    This is the FP twin taking the same cure.

    EQUIVALENCE, so the risk to shared emission code is bounded rather
    than assumed: on every board where the count and the board AGREE -
    every trajectory that is not already broken - this returns exactly
    what the old arithmetic returned, pinned for 0, 1 and 2 rings used
    (test_ring_gesture_names_present_marker.py leg 5).  It can differ
    ONLY where the old code was already emitting an impossible gesture.
    """
    where = {}
    for _p in ('249', '250', '251'):
        try:
            where[_p] = _piece_zone_code(replayer, _p, default=None)
        except LookupError:
            where[_p] = None
    # THE REPEAT GUARD IS A LEDGER-vs-COUNT CHECK, NOT A BOARD CHECK
    # (BRIDGE-11): it refuses only when the LEDGER HAS RUN AHEAD OF THE
    # COUNT - the caller is re-emitting one logical use.  A board that
    # merely lags the ledger (the round-trip rig lags BY DESIGN) must
    # never refuse.  WORDING IS PART OF THE CONTRACT:
    # test_permanent_class.py asserts the phrase 'already been walked'.
    if paid is not None and used is not None and len(paid) >= used + 1:
        raise LookupError(
            'an FP elven ring has already been walked to its Used Ring '
            'slot for this use (ledger=%s, but the engine has spent '
            'only %d) - refusing to walk a second ring for one use'
            % (' '.join(sorted(paid)) or '(none)', used))
    avail = [(p, z) for p, z in sorted(where.items())
             if z in RING_ACTIVE_SLOTS and not (paid and p in paid)]
    if not avail:
        # NAME THE REAL CAUSE: "no ring is in a slot" and "every ring
        # is already in my ledger" are different facts with different
        # fixes.
        if paid and all(p in paid for p in ('249', '250', '251')):
            raise LookupError(
                'all three FP elven rings have already been walked '
                '(ledger=%s) - there is no fourth ring'
                % ' '.join(sorted(paid)))
        raise LookupError(
            'no FP elven ring is in an Elven Ring slot (board: %s; '
            'ledger=%s) - refusing to emit a walk of a marker that is '
            'not there'
            % (', '.join('%s=%s' % (p, where.get(p))
                         for p in ('249', '250', '251')),
               ' '.join(sorted(paid)) if paid else '(none)'))
    pid, frm = avail[0]
    # DESTINATION: the first Used Ring slot that is free ON THE BOARD,
    # skipping the walks this caller has emitted that the board has not
    # shown yet (lag 0 = "the first free slot"; a lagging board
    # reproduces the `2011+used` arithmetic exactly).
    occupied = set(z for z in where.values() if z in RING_USED_SLOTS)
    lag = sum(1 for p in (paid or ())
              if where.get(p) in RING_ACTIVE_SLOTS)
    free = [z for z in RING_USED_SLOTS if z not in occupied]
    if lag >= len(free):
        raise LookupError(
            'no free Used Ring slot for %s (board occupied=%s, '
            'un-landed walks=%d)'
            % (pid, ', '.join(sorted(occupied)) or '(none)', lag))
    return pid, frm, free[lag]


def pick_sh_ring_dispose(replayer, disposed=None):
    """WHICH collected ring the Shadow disposes, and WHICH slot it is
    ACTUALLY in.  The twin of `pick_fp_ring_walk`, and the second half
    of the same cure.

    THE AUTO69 DEFECT IS NOT WHAT THIS FIXES - that one is already
    cured.  auto69 game 5 disposed pid 251 TWICE and orphaned 250 in
    Used Ring 2 forever; the caller's `_ring_disposed` ledger (a ring
    may leave exactly once, and a halt when it is exhausted) makes that
    impossible now.

    WHAT THIS FIXES is one layer in.  Having CHOSEN the ring from the
    board, the site computed where to drag it FROM by arithmetic -
    `'Game.%d' % (2011 + index_of(pid))` - which assumes RING N IS IN
    USED RING N.  Nothing enforces that.  Driven RED
    (probe_ring_dispose_frm_arithmetic.py), through the real shipped
    FP planner rather than a hand-built board:

        FP walks 249 -> Used Ring 1, 250 -> Used Ring 2
        Shadow disposes 250, freeing Used Ring 2
        FP walks 251 -> Used Ring 2   (the first FREE slot)
        Shadow disposes 251 -> shipped frm = Game.2013 (Used Ring 3)
                               board says  = Game.2012 (Used Ring 2)

    i.e. a drag out of a slot the ring is not in.  `pick_fp_ring_walk`
    sending a ring to the first FREE used slot is what makes the index
    coupling breakable, so these two must agree about the board or
    neither can be trusted - ONE ANSWERER for "where is this ring".

    THE STALE-TRACKING CASE IS DELIBERATELY LEFT ALONE.  When the board
    cannot say where any undisposed ring is, there is no better answer
    than the ledger + the index, and the caller already prints that it
    is guessing.  Returning frm=None hands that case back unchanged, so
    behaviour differs ONLY where the board knows the slot and the old
    arithmetic named a different one - which is exactly where the old
    code was emitting an impossible gesture.

    Returns (pid, frm_or_None, stale).
    """
    disposed = disposed or set()
    where = {}
    for _p in ('249', '250', '251'):
        try:
            where[_p] = _piece_zone_code(replayer, _p, default=None)
        except LookupError:
            where[_p] = None
    cands = [p for p in ('251', '250', '249') if p not in disposed]
    if not cands:
        raise LookupError(
            'all three ring pieces are already disposed (%s)'
            % ' '.join(sorted(disposed)))
    for p in cands:
        if where.get(p) in RING_USED_SLOTS:
            return p, where[p], False
    return cands[-1], None, True


def plan_elven_ring(replayer, nick, side_name, params):
    """Transfer/dispose one actual ring and change the selected physical die."""
    ops = []
    state = replayer.state
    if side_name == 'FREE_PEOPLES':
        _paid = params.get('_ring_paid')
        _used9 = 3 - int(getattr(state, 'fp_elven_rings', 3))
        pid, _frm, _to = pick_fp_ring_walk(replayer, _paid, _used9)
        # COUNTS MUST AGREE (USER 2026-08-19, invariant 2: no phantom
        # ring, none spent twice, none stranded): a count that points
        # at a ring the board does NOT have in a slot is a tracking
        # break, SAID not raised (refusing would block a legal ring
        # use).  Narrow on purpose: only when the count's ring is
        # ABSENT from the slots - a lagging board is normal (the
        # round-trip rig lags by design) and a different CHOICE is not
        # a defect ("which ring is irrelevant as long as it is cleanly
        # tracked").
        if 0 <= _used9 <= 2:
            _want9 = ('249', '250', '251')[_used9]
            _z9 = None
            try:
                _z9 = _piece_zone_code(replayer, _want9, default=None)
            except LookupError:
                _z9 = None
            if _z9 not in RING_ACTIVE_SLOTS:
                print('    [ring] COUNT/BOARD DISAGREE: '
                      'fp_elven_rings=%s implies %s, but the board has '
                      '%s in %s - walking %s, which is actually in a '
                      'slot'
                      % (getattr(state, 'fp_elven_rings', '?'), _want9,
                         _want9, _z9, pid))
        if _paid is not None:
            _paid.add(pid)
        ops.append(('marker_366', {'nick': nick, 'piece': pid,
                                   'frm': _frm, 'to': _to}))
    else:
        pid, frm, stale = pick_sh_ring_dispose(replayer)
        if stale or frm not in RING_USED_SLOTS:
            raise LookupError('no Shadow-held Elven Ring is on the client board')
        ops.append(('kill', {'nick': nick, 'pieces': [pid], 'to': SH_DEAD}))
    face = str(getattr(params.get('new_face'), 'value',
                       params.get('new_face'))).upper().replace(' ', '_').replace('/', '_')
    face = {'EYE_OF_SAURON': 'EYE', 'WILL_OF_THE_WEST': 'WILL_OF_WEST'}.get(face, face)
    letter = _FACE_LETTER.get(face)
    if letter is None:
        raise LookupError('no die-change letter for face %r' % face)
    dp = die_piece_for(replayer, side_name, params['target_die_idx'])
    ops.append(('die_change', {'nick': nick, 'die': dp, 'face': letter}))
    if side_name == 'SHADOW' and face == 'EYE':
        ops.append(plan_eye_bank(replayer, nick, dp))
    return ops


def plan_use_token(replayer, nick, side_name, params, rng=None):
    """Token drag into the dice area + the token's effect line."""
    muster = str(params.get('token_type')) == 'muster'
    pid = '426' if muster else '427'
    frm = _piece_zone_code(replayer, pid, default=FP_POOL)
    # tokens are spent INTO the Used-dice box (corpus: Game.2035)
    ops = [('move', {'nick': nick, 'pieces': [pid], 'frm': frm,
                     'to': DICE_USED})]
    if muster:
        ops.append(plan_advance_political(replayer, nick, side_name,
                                          params))
    else:
        ops.append(plan_draw_card(replayer, nick, side_name,
                                  {'deck': params.get('deck',
                                                      'character')},
                                  rng=rng))
    return ops


def casualty_context(state):
    """The active owner, region and remaining budget for either casualty source."""
    card = getattr(state, 'pending_card_option', None) or {}
    if card.get('card_casualty'):
        return {'kind': 'card', 'stage': 'card', 'region': card['cas_region'],
                'side': card['cas_side'], 'hits': int(card['cas_hits']),
                'card_id': card['card_id']}
    pc = getattr(state, 'pending_combat', None) or {}
    stage = pc.get('alloc_stage')
    if pc.get('phase') == 'casualty_allocation' and stage in ('attacker', 'defender'):
        return {'kind': 'combat', 'stage': stage, 'region': pc[stage + '_region'],
                'side': pc[stage + '_side'],
                'hits': int(pc.get('atk_hits' if stage == 'attacker' else 'def_hits', 0) or 0),
                'card_id': None}
    return None


def casualty_action_params(action):
    """Normalize only explicit ordinary/card casualty actions, never generic cards."""
    if action.action_type == 'casualty_allocate':
        return dict(action.params)
    choices = (action.params or {}).get('choices') or {}
    if action.action_type == 'card_option' and 'casualty_operation' in choices:
        return {'operation': choices['casualty_operation'], 'nation': choices.get('nation')}
    return None


def plan_casualty_operation(params, replayer, nick, side_name):
    """One source-stamped casualty operation, shared by combat and event cards."""
    state = replayer.state
    context = casualty_context(state)
    if context is None:
        raise LookupError('casualty operation requires an active allocation')
    operation, nation = params.get('operation'), params.get('nation')
    if operation not in ('remove_regular', 'reduce_elite', 'remove_elite') or nation is None:
        raise LookupError('casualty operation requires an explicit operation and nation')
    owner = context['side']
    if getattr(owner, 'name', None) != side_name:
        raise LookupError('casualty operation belongs to another player')
    side = 'shadow' if side_name == 'SHADOW' else 'fp'
    region = _norm_zone(context['region'])
    zc = garrison_arrival_zone(state, region, side)
    # Validate against the same participating roster as the engine before
    # choosing a physical figure; peaceful rearguard cannot pay a hit.
    import ai
    hits = context['hits']
    valid = ai._casualty_operations(state, context['region'], owner, hits)
    if not any(a.params.get('nation') == nation and
               a.params.get('operation') == operation for a in valid):
        raise LookupError('casualty operation is not legal in this position')
    units = pick_units(replayer, region, side,
                       n_regular=int(operation == 'remove_regular'),
                       n_elite=int(operation != 'remove_regular'),
                       nation=nation, side_hint=side)
    if operation != 'reduce_elite':
        return [('hand_exit_b', {'nick': nick, 'pieces': units, 'frm': zc})]
    # The first stamp belongs to this operation. Later stamps can belong
    # to forced follow-through and must not choose this replacement.
    stamps = params.get('_dg_srcs') or []
    if not stamps:
        raise LookupError('Elite reduction has no engine source stamp')
    stamped_nation, source = stamps[0]
    norm = lambda n: str(getattr(n, 'value', n)).lower().replace(' ', '_')
    if norm(stamped_nation) != norm(nation) or source not in ('cas', 'pool', None):
        raise LookupError('Elite reduction source stamp does not match the action')
    ops = [('swap_place', {'nick': nick, 'pieces': units, 'to': zc})]
    if source is not None:
        zone = (('sa dead' if side == 'shadow' else 'fp dead')
                if source == 'cas' else
                ('sa reinf' if side == 'shadow' else 'fp reinf'))
        replacement = pick_units(replayer, zone, side, n_regular=1,
                                 nation=nation, side_hint=side)
        ops.append(('swap_remove', {'nick': nick, 'pieces': replacement,
                    'sub': '308', 'subtext': '', 'to': zc}))
    return ops



def plan_siege_press(params, replayer, nick, side_name):
    """Render the engine's one paid Elite, using its nation and supply stamp."""
    state = replayer.state
    pc = getattr(state, 'pending_combat', None) or {}
    if getattr(pc.get('attacker_side'), 'name', None) != side_name:
        raise LookupError('siege press belongs to another player')
    region = _norm_zone(pc.get('attacker_region') or pc.get('region', ''))
    if not region:
        raise LookupError('siege press has no attacker region')
    stamps = params.get('_dg_srcs') or []
    if not stamps or not isinstance(stamps[0], (tuple, list)) or len(stamps[0]) != 2:
        raise LookupError('siege press requires its engine downgrade source stamp')
    nation, source = stamps[0]
    side = 'shadow' if side_name == 'SHADOW' else 'fp'
    if nation is None or _side_of_nation(nation) != side or source not in ('cas', 'pool', None):
        raise LookupError('siege press has an invalid nation or replacement source')
    requested = params.get('nation')
    norm = lambda n: str(getattr(n, 'value', n)).lower().replace(' ', '_')
    if requested is not None and norm(requested) != norm(nation):
        raise LookupError('siege press nation disagrees with engine payment stamp')
    elite = pick_units(replayer, region, side, n_elite=1,
                       nation=nation, side_hint=side)
    # Both halves use the same concrete piece's physical location, including
    # a previously witnessed generic stronghold slot.
    zone = _piece_zone_code(replayer, elite[0])
    ops = [('swap_place', {'nick': nick, 'pieces': elite, 'to': zone})]
    if source is not None:
        supply = (('sa dead' if side == 'shadow' else 'fp dead')
                  if source == 'cas' else
                  ('sa reinf' if side == 'shadow' else 'fp reinf'))
        regular = pick_units(replayer, supply, side, n_regular=1,
                             nation=nation, side_hint=side)
        ops.append(('swap_remove', {'nick': nick, 'pieces': regular,
                    'sub': '308', 'subtext': '', 'to': zone}))
    return ops

def plan(action_type, params, replayer, nick, side_name,
         die_idx=None, rng=None):
    """One engine action -> ordered [(op, fields), ...] micro-ops.
    Raises NotImplementedError for uncovered types (server falls back
    loudly), LookupError when the boards disagree (bug — surface it)."""
    ops = []
    if die_idx is not None and die_idx >= 0:
        ops.append(plan_die_spend(replayer, nick, side_name, die_idx))
    if action_type == 'pass':
        return [('chat', {'nick': nick, 'msg': 'pass'})]
    if action_type == 'army_move':
        # A legal paid move may activate a nation without changing region
        # (Almanac p.41). Its political delta is emitted by the commit sweep.
        if (params.get('from') is None or params.get('to') is None
                or _norm_zone(params['from']) != _norm_zone(params['to'])):
            ops.append(plan_army_move(replayer, nick, side_name, params))
        return ops
    if action_type == 'draw_card':
        ops.append(plan_draw_card(replayer, nick, side_name, params,
                                  rng=rng))
        return ops
    if action_type == 'advance_political':
        ops.append(plan_advance_political(replayer, nick, side_name,
                                          params))
        return ops
    if action_type == 'muster_recruit':
        ops.append(plan_muster(replayer, nick, side_name, params,
                               str(params.get('type', 'regular'))))
        return ops
    if action_type == 'muster_elite':
        ops.append(plan_muster(replayer, nick, side_name, params,
                               'elite'))
        return ops
    if action_type == 'muster_minion':
        ops.append(plan_muster_minion(replayer, nick, side_name,
                                      params))
        return ops
    if action_type in IMPLICIT_ACTIONS:
        # decision made by NOT acting; parser infers from later lines
        return ops
    if action_type == 'army_attack':
        ops.append(plan_army_attack(replayer, nick, side_name, params))
        return ops
    if action_type == 'victor_advance':
        # post-battle advance = an army move into the vacated region
        ops.append(plan_army_move(replayer, nick, side_name, params))
        return ops
    if action_type == 'companion_move':
        if (params.get('from') is None or params.get('to') is None
                or _norm_zone(params['from']) != _norm_zone(params['to'])):
            ops.append(plan_companion_move(replayer, nick, side_name,
                                           params))
        return ops
    if action_type == 'companion_separate':
        ops.extend(plan_companion_separate(replayer, nick, side_name,
                                           params))
        return ops
    if action_type == 'nazgul_move':
        ops.extend(plan_nazgul_move(replayer, nick, side_name, params))
        return ops
    if action_type == 'fellowship_move':
        fs = replayer.state.fellowship
        if getattr(fs, 'mordor_entered', False):
            # Mordor track: pawn 247 walks box N -> N+1.  Table hoisted
            # to MORDOR_TRACK_ZONES so the reconciler shares it.
            mz = MORDOR_TRACK_ZONES
            step = int(params.get('from', params.get(
                'from_step', fs.mordor_step)))
            post_step = params.get('mordor_post_step')
            if post_step is not None and int(post_step) <= step:
                # the ENGINE did not advance (stop tile / reveal kept
                # the pawn) — emitting a walk anyway desyncs the
                # client pawn one box ahead (t116 seeds 11/23).  The
                # spent die STILL funds the hunt: drop it in the box.
                if die_idx is not None and die_idx >= 0:
                    dp = die_piece_for(replayer, side_name, die_idx)
                    ops.append(('die_to_hunt', {
                        'nick': nick, 'die': dp,
                        'frm': DICE_USED, 'to': HUNT_BOX}))
                return ops
            if not 0 <= step <= 4:
                raise LookupError('mordor step %r cannot advance'
                                  % (step,))
            ops.append(('move', {'nick': nick, 'pieces': ['247'],
                                 'frm': mz[step], 'to': mz[step + 1]}))
            if die_idx is not None and die_idx >= 0:
                # the spent FP die still funds the hunt on the Mordor
                # track — drop it in the Hunt Box exactly like the
                # pre-Mordor branch (missing drop left the client
                # showing the die in Used; live census t116)
                dp = die_piece_for(replayer, side_name, die_idx)
                ops.append(('die_to_hunt', {'nick': nick, 'die': dp,
                                            'frm': DICE_USED,
                                            'to': HUNT_BOX}))
            return ops
        # engine action params are {} (state.progress = PRE-move value);
        # parser records carry {'from','to'} and fire mid-execution
        # (progress already advanced) — honor the recorded 'from'
        p = int(params.get('from', fs.progress))
        ops.append(('track_marker', {'nick': nick, 'piece': '248',
                                     'frm': 'Game.%d' % (2014 + p),
                                     'to': 'Game.%d' % (2015 + p)}))
        if die_idx is not None and die_idx >= 0:
            # the SAME just-spent die goes Used -> Hunt Box (this drop
            # is what hands the Shadow player the hunt roll)
            dp = die_piece_for(replayer, side_name, die_idx)
            ops.append(('die_to_hunt', {'nick': nick, 'die': dp,
                                        'frm': DICE_USED,
                                        'to': HUNT_BOX}))
        return ops
    if action_type == 'fellowship_hide':
        ops.append(('flip', {'nick': nick, 'pieces': ['247']}))
        return ops
    if action_type in ('fellowship_declare', 'fellowship_reveal_place'):
        # engine says 'region'; parser records say 'to' (+ 'from')
        dest = params.get('region', params.get('to'))
        if dest is None:
            raise LookupError('no destination region in %r' % (params,))
        # G13 / PAWN CONTINUITY (USER 2026-08-14): the pawn's TRUE
        # region comes from the parser's fellowship tracker FIRST (the
        # 247 walks ride the fellowship handler, never piece_region;
        # the engine's 'from' is its last DECLARED position and would
        # teleport the pawn back on every reveal)
        cur = getattr(replayer, 'fellowship_location', None) \
            or params.get('from') \
            or replayer.piece_region.get('247') or _initial().get('247')
        if not cur:
            raise LookupError('fellowship pawn zone unknown')
        frm_c = zone_code_for(_norm_zone(cur))
        to_c = zone_code_for(_norm_zone(dest))
        if frm_c != to_c:
            # human grammar: ONE line per region stepped (H01 64-65,
            # 696) — BFS the board for the shortest hop chain; an
            # unpathable pair (off-map boxes) falls back to one line
            hops = None
            try:
                from collections import deque
                b9 = replayer.state.board
                src9, dst9 = _norm_zone(cur), _norm_zone(dest)
                seen9 = {src9: None}
                q9 = deque([src9])
                while q9:
                    r9 = q9.popleft()
                    if r9 == dst9:
                        break
                    for adj9 in b9.adjacent(r9):
                        a9 = _norm_zone(adj9)
                        if a9 not in seen9:
                            seen9[a9] = r9
                            q9.append(a9)
                if dst9 in seen9:
                    hops = []
                    node9 = dst9
                    while node9 is not None and seen9[node9] is not None:
                        hops.append(node9)
                        node9 = seen9[node9]
                    hops.reverse()
            except Exception:
                hops = None
            if hops and 1 < len(hops) <= 6:
                prev9 = _norm_zone(cur)
                for h9 in hops:
                    ops.append(('move', {
                        'nick': nick, 'pieces': ['247'],
                        'frm': zone_code_for(prev9),
                        'to': zone_code_for(h9)}))
                    prev9 = h9
            else:
                ops.append(('move', {'nick': nick, 'pieces': ['247'],
                                     'frm': frm_c, 'to': to_c}))
        if action_type == 'fellowship_declare':
            p = int(replayer.state.fellowship.progress)
            if p > 0:
                # corpus grammar: a progress RESET is a Scribe DRAG back
                # down the track; no human log contains a backward
                # Controls.790 "advance" (that form also leaves
                # log_replay's _pre_reset_progress stash dangling)
                ops.append(('move', {'nick': nick, 'pieces': ['248'],
                                     'frm': 'Game.%d' % (2014 + p),
                                     'to': 'Game.2014'}))
            # marker dance: humans flip-and-unflip the fellowship marker
            # when declaring (O.1 bookkeeping pair) and log_replay's
            # declare branch arms _skip_next_fsp_flip=2 for EXACTLY
            # this pair - without it the parser eats the next two
            # GENUINE flips
            ops.append(('flip', {'nick': nick, 'pieces': ['247']}))
            ops.append(('flip', {'nick': nick, 'pieces': ['247']}))
        else:
            # hunt-forced reveal placement: corpus choreography is reset
            # -> walk -> flip.  The reset rides the planner so the
            # MANUAL decide() path stays in sync; from_progress is
            # injected by the AUTO driver (engine truth), the parser's
            # counter is the fallback for decide().
            p = int(params.get('from_progress',
                               replayer.state.fellowship.progress))
            if p > 0:
                ops.insert(0, ('move', {'nick': nick, 'pieces': ['248'],
                                        'frm': 'Game.%d' % (2014 + p),
                                        'to': 'Game.2014'}))
            ops.append(('flip', {'nick': nick, 'pieces': ['247']}))
        return ops
    if action_type == 'crown_aragorn':
        ops.append(('flip', {'nick': nick, 'pieces': ['241']}))
        return ops
    if action_type == 'transform_gandalf':
        # dead Gandalf is held in an off-map box that varies by build
        # (FP Dead 2039 or FP Card Play 2037); honor the tracked holding
        # box, else default to Dead (piece_region can stale-point at his
        # death region when the kill line didn't re-track him)
        frm = _holding_zone_code(replayer, '240', default=FP_DEAD)
        if frm not in (FP_DEAD, 'Game.2037'):
            # BL-75: Gandalf the GREY may be ALIVE and SEPARATED on the
            # map when the transformation fires ('no longer in the
            # Fellowship' does not require dead) - a tracked MAP REGION
            # is then the TRUE source; only a NON-region stale zone
            # (Guide Box, tracks) falls back to the dead pile.
            _is_reg75 = False
            try:
                import board as _bd75
                from board import sanitize as _sz75
                _z75 = str(replayer.piece_region.get('240', '')
                           or '')
                _is_reg75 = _sz75(_z75) in _bd75.regions
            except Exception:
                _is_reg75 = False
            if not _is_reg75:
                frm = FP_DEAD
        # ASK THE GARRISON QUESTION: Gandalf the White returns to an FP
        # Elven stronghold, which can be BESIEGED when he does - he
        # joins the garrison INSIDE, never the besiegers.  The server
        # injects '_besieged_garrison' as it does for musters, because
        # only the ENGINE knows the siege state.
        _gdest9 = _norm_zone(params['region'])
        if params.get('_besieged_garrison'):
            try:
                zone_code_for(_gdest9 + ' stronghold')
                _gdest9 = _gdest9 + ' stronghold'
            except LookupError:
                pass
        ops.append(('move', {'nick': nick, 'pieces': ['240'],
                             'frm': frm,
                             'to': zone_code_for(_gdest9)}))
        # corpus grammar (GenericPie L215): the transformation FLIPS
        # piece 240 to its Gandalf-the-White side right after the walk
        ops.append(('flip', {'nick': nick, 'pieces': ['240']}))
        return ops
    if action_type == 'elven_ring':
        return ops + plan_elven_ring(replayer, nick, side_name, params)
    if action_type == 'use_token':
        return ops + plan_use_token(replayer, nick, side_name, params,
                                    rng=rng)
    if action_type == 'choose_guide':
        pid = _companion_pid(params['companion'])
        frm = _piece_zone_code(replayer, pid, default=FELLOWSHIP_BOX)
        if frm == GUIDE_BOX:
            return ops  # Keeping the current Guide is a real choice, no drag.
        # The Java Guide zone accepts multiple figures. A nomination must
        # return the previous living Guide before seating the chosen one.
        living = [c.name for c in replayer.state.fellowship.companions]
        for name in living:
            old_pid = _companion_pid(name)
            if old_pid != pid and _piece_zone_code(replayer, old_pid, default=FELLOWSHIP_BOX) == GUIDE_BOX:
                ops.append(('move', {'nick': nick, 'pieces': [old_pid],
                                     'frm': GUIDE_BOX, 'to': FELLOWSHIP_BOX}))
        ops.append(('move', {'nick': nick, 'pieces': [pid],
                             'frm': frm, 'to': GUIDE_BOX}))
        return ops
    if action_type == 'combat_card_none':
        ops.append(('chat', {'nick': nick, 'msg': 'nc'}))
        return ops
    if action_type == 'combat_effect_choice':
        # Payment gestures are emitted from the committed board delta.
        if all(key in params for key in ('nation', 'kind', 'operation')):
            return ops
        from live.chat import combat_effect_choice_note
        ops.append(('chat', {'nick': nick, 'msg': combat_effect_choice_note(params)}))
        return ops
    if action_type == 'combat_retreat':
        # engine shape: params carry ONLY to_region; the retreating
        # side's region is its combat-role region in pending_combat
        # (attacker_region / defender_region - there is no 'region').
        # Parser records carry 'from' and win when present.
        pc = getattr(replayer.state, 'pending_combat', None) or {}
        region = params.get('from') or params.get('region')
        if not region:
            region = _battle_region_of(pc, side_name)
        region = _norm_zone(region)
        dest = params.get('to_region', params.get('to'))
        if not region or dest is None:
            raise LookupError('no battle region/destination (%r, %r)'
                              % (params, sorted(pc)))
        side = 'shadow' if side_name == 'SHADOW' else 'fp'
        pids = _side_pieces_in(replayer, region, side)
        if not pids:
            raise LookupError('no %s pieces in %r to retreat'
                              % (side, region))
        ops.append(('move', {'nick': nick, 'pieces': pids,
                             'frm': zone_code_for(region),
                             'to': zone_code_for(_norm_zone(dest))}))
        return ops
    if action_type == 'retreat_into_siege':
        # engine shape: params == {}; the besieged stronghold is the
        # DEFENDER's region (only the garrison side may withdraw)
        pc = getattr(replayer.state, 'pending_combat', None) or {}
        region = _norm_zone(params.get('region') or params.get('from')
                            or pc.get('defender_region', ''))
        if not region:
            raise LookupError('no pending_combat defender_region for '
                              'siege (%r)' % (sorted(pc),))
        side = 'shadow' if side_name == 'SHADOW' else 'fp'
        pids = _side_pieces_in(replayer, region, side)
        if not pids:
            raise LookupError('no %s pieces in %r to besiege'
                              % (side, region))
        ops.append(('move', {'nick': nick, 'pieces': pids,
                             'frm': zone_code_for(region),
                             'to': zone_code_for(region
                                                 + ' stronghold')}))
        return ops
    if action_type == 'siege_trim':
        # the withdrawing owner sends the overstack excess back to
        # REINFORCEMENTS: k elites ('elites_back') + (excess - k)
        # regulars.  The exact (nation, kind) rows the engine removed
        # ride '_trim_by_nation' (threaded from pre/post by the live
        # emission); without them the pick is nation-blind off k.
        pc = getattr(replayer.state, 'pending_combat', None) or {}
        region = _norm_zone(params.get('region')
                            or pc.get('trim_region')
                            or pc.get('defender_region', ''))
        if not region:
            raise LookupError('no pending_combat trim_region (%r)'
                              % (sorted(pc),))
        side = 'shadow' if side_name == 'SHADOW' else 'fp'
        rows = params.get('_trim_by_nation')
        if rows:
            want = [(key.split(':', 1)[0], key.split(':', 1)[1], int(n))
                    for key, n in sorted(rows.items())]
        else:
            k = int(params.get('elites_back', 0) or 0)
            exc = int(params.get('excess', 0) or 0)
            want = [(None, 'elite', k), (None, 'regular', exc - k)]
        chosen = []
        for nat, kind, n in want:
            if n <= 0:
                continue
            chosen += pick_units(replayer, region, side,
                                 n_regular=(n if kind == 'regular'
                                            else 0),
                                 n_elite=(n if kind == 'elite' else 0),
                                 nation=nat, exclude=chosen,
                                 side_hint=side)
        if not chosen:
            raise LookupError('siege_trim removes no unit (%r)'
                              % (params,))
        groups = []
        for pid in chosen:
            frm = _piece_zone_code(replayer, pid,
                                   default=zone_code_for(region))
            for g in groups:
                if g[0] == frm:
                    g[1].append(pid)
                    break
            else:
                groups.append((frm, [pid]))
        for gf, gp in groups:
            ops.append(('move', {'nick': nick, 'pieces': list(gp),
                                 'frm': gf,
                                 'to': (SH_POOL if side == 'shadow'
                                        else FP_POOL)}))
        return ops
    if action_type == 'casualty_allocate':
        return plan_casualty_operation(params, replayer, nick, side_name)
    if action_type == 'siege_press':
        return plan_siege_press(params, replayer, nick, side_name)
    if action_type == 'hunt_take_corruption':
        # FROM = where the client's marker physically sits (parser);
        # TO = the ENGINE's landing spot (corruption_target injected by
        # the server; the pending_hunt damage is the fallback).  The
        # engine target also self-heals any earlier marker drift.
        c = int(replayer.state.fellowship.corruption)
        new = params.get('corruption_target')
        if new is None:
            ph = getattr(replayer.state, 'pending_hunt', None) or {}
            new = c + int(ph.get('damage', 1) or 1)
        new = max(0, min(12, int(new)))
        if new == c:
            return ops
        # corpus grammar: the corruption marker 252 moves ONLY via
        # Scribe drags on the shared Track 0-12 zones (Game.2014+N) —
        # zero Controls.790 lines on 252 in the entire 400-game corpus.
        # Multi-step single drags are attested (GenericPie 2023-01-02
        # L147: '252 Game.2014 -> Game.2016' = +2 corruption).
        ops.append(('move', {'nick': nick, 'pieces': ['252'],
                             'frm': 'Game.%d' % (2014 + c),
                             'to': 'Game.%d' % (2014 + new)}))
        return ops
    if action_type == 'hunt_sacrifice_guide':
        name = params.get('companion')
        if not name:
            guide = getattr(replayer.state.fellowship, 'guide', None)
            if guide is None:
                raise LookupError('no guide to sacrifice')
            name = guide.name
        pid = _companion_pid(name)
        frm = _guide_departure_zone(replayer, pid)
        # ALIVE SEPARATION (USER ruling 2026-08-15): a HOBBIT guide
        # sacrificing OUTSIDE Mordor separates onto the map, not into
        # the dead pile.  The server injects the engine's true
        # placement region as to_region; the fellowship's tracked
        # region is the fallback mirror of the engine rule.
        dest = params.get('to_region')
        if dest is None and name in ('Meriadoc', 'Peregrin') \
                and not getattr(replayer.state.fellowship,
                                'mordor_entered', False):
            dest = getattr(replayer, 'fellowship_location', None)
        if dest:
            ops.append(('move', {'nick': nick, 'pieces': [pid],
                                 'frm': frm,
                                 'to': zone_code_for(
                                     _norm_zone(str(dest)))}))
            return ops
        ops.append(('move', {'nick': nick, 'pieces': [pid],
                             'frm': frm, 'to': FP_DEAD}))
        return ops
    if action_type == 'hunt_guide_separate':
        # the hobbit guide's ability: outside Mordor the hobbit is
        # QUEUED for the hobbit_place pending, whose companion_separate
        # action carries the placement drag (rendered there, once);
        # on the Mordor track the separation kills the hobbit - the
        # death drag is the only gesture.  The guide change that
        # follows is implicit in the client (as for every separation).
        fs = replayer.state.fellowship
        if not getattr(fs, 'mordor_entered', False):
            return ops
        name = params.get('companion')
        if not name:
            guide = getattr(fs, 'guide', None)
            if guide is None:
                raise LookupError('no guide to separate')
            name = guide.name
        pid = _companion_pid(name)
        ops.append(('move', {'nick': nick, 'pieces': [pid],
                             'frm': _guide_departure_zone(replayer, pid),
                             'to': FP_DEAD}))
        return ops
    if action_type == 'mouth_move':
        # the Mouth of Sauron walks alone on a Character die (range 3):
        # ONE drag of his piece from the engine's 'from' to 'to'
        pid = next((p for p, n in _lr().MINION_PIECES.items()
                    if 'mouth' in str(n).lower()), None)
        if pid is None:
            raise LookupError('no Mouth of Sauron piece id')
        frm = _norm_zone(params['from'])
        to = _norm_zone(params['to'])
        tracked = replayer.piece_region.get(pid) or _initial().get(pid)
        if tracked and _norm_zone(tracked) in (frm, frm + ' stronghold'):
            frm_code = _piece_zone_code(replayer, pid)
        else:
            frm_code = zone_code_for(frm)
        ops.append(('move', {'nick': nick, 'pieces': [pid],
                             'frm': frm_code,
                             'to': _garrison_dest(replayer, to,
                                                  'SHADOW')}))
        return ops
    if action_type == 'dwarven_ring':
        # a Dwarven Ring marker leaves the side's dice pool for the
        # Used box (the use); the rerolled die's NEW face - the
        # engine's, landed from the client's own d6 - is stated with
        # the face-change line.  new_face is threaded from the post
        # state by the live emission; without it only the ring walks.
        is_fp = str(params.get('side', 'fp')).lower() != 'sh'
        pool_zone = 'fp dice' if is_fp else 'sa dice'
        ring = next((p for p in sorted(_lr().DWARVEN_RING_PIECES)
                     if _norm_zone(replayer.piece_region.get(p)
                                   or _initial().get(p) or '')
                     == pool_zone), None)
        if ring is None:
            raise LookupError('no Dwarven Ring piece in %r' % (pool_zone,))
        ops.append(('move', {'nick': nick, 'pieces': [ring],
                             'frm': FP_DICE_POOL if is_fp
                             else SH_DICE_POOL,
                             'to': DICE_USED}))
        face = params.get('new_face')
        if face is not None:
            face = str(getattr(face, 'name', face)).upper() \
                .replace(' ', '_').replace('/', '_')
            face = {'EYE_OF_SAURON': 'EYE',
                    'WILL_OF_THE_WEST': 'WILL_OF_WEST'}.get(face, face)
            letter = _FACE_LETTER.get(face)
            if letter is None:
                raise LookupError('no die-change letter for face %r'
                                  % (face,))
            dp = die_piece_for(replayer,
                               'FREE_PEOPLES' if is_fp else 'SHADOW',
                               int(params['die_idx']))
            ops.append(('die_change', {'nick': nick, 'die': dp,
                                       'face': letter}))
        return ops
    if action_type == 'hunt_gollum_reveal':
        # FLIP ONLY: the engine KEEPS the progress counter on a Gollum
        # reveal (only fellowship_reveal_place/declare reset it) - a
        # reset drag here would misstate the engine state and be
        # "corrected" back by consecutive Controls.790 lines, which
        # corpus grammar reads as fellowship MOVE presses.  A reveal
        # that keeps progress renders as the marker flip alone.
        ops.append(('flip', {'nick': nick, 'pieces': ['247']}))
        return ops
    if action_type == 'eye_allocate':
        # one line per eye die moved to the hunt box.  count=0 is the
        # engine's LEGAL zero-eye choice -> zero ops.
        _n_eyes = params.get('count')
        _n_eyes = 1 if _n_eyes is None else int(_n_eyes)
        for i in range(_n_eyes):
            idx = params['die_indices'][i] \
                if params.get('die_indices') else None
            if idx is None:
                raise NotImplementedError('eye_allocate needs die ids')
            pid = die_piece_for(replayer, 'SHADOW', idx)
            ops.append(('eye_alloc', {'nick': nick, 'die': pid}))
        return ops
    # no L2 'die_to_hunt' planner exists ON PURPOSE: no engine action
    # of that type exists, and the SHADOW-pool-sourced Controls.795 form
    # it once rendered is a V1 grammar violation humans never emit
    # (they drag Scribe.141 2043->1997).  Controls.795 is correct only
    # for the FP FELLOWSHIP die out of Game.2035 (DICE_USED), which the
    # four live sites source.
    raise NotImplementedError('no L2 planner for %r yet' % action_type)
