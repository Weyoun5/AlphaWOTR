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
# Routed HERE, at render(), rather than at the ~dozen call sites that
# build these ops - a call-site sweep is exactly the shape of edit I
# got wrong twice today (fixed one of three hunt sites; fixed one of
# sixteen garrison sites), and a chokepoint cannot be half-applied.
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
    # though the token meant a destination.
    #
    # Enforced here rather than at those four sites: this is the third
    # time today a fix has had to cover N producers of one act, and the
    # first two only worked because they were done at a chokepoint.
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
# contiguous, which is exactly why this must exist in ONE place: a
# second transcription of 524/525/528/529/532/533 is a table nobody
# would notice going stale.  Used by the fellowship_move planner and
# by the per-action Mordor reconciler in ai_move_server.
MORDOR_TRACK_ZONES = ('Game.524', 'Game.525', 'Game.528', 'Game.529',
                      'Game.532', 'Game.533')

# Set True by AiBrain._install_live_patches when the client-randomness
# funnels go in.  Two planners below can SOURCE randomness themselves
# (a die face, a drawn card identity); that is legitimate offline,
# where there is no client to ask, and forbidden in live play, where
# inventing a value the client never produced is the injected
# randomness the USER ruled out on 2026-08-16.  The planners cannot
# see the brain, so the brain tells the module.
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
        # ZONE-NAME ALIAS (game 4 'Shadows on the Misty Mountains',
        # USER 2026-08-14): the parser's kill/dkill routing tracks
        # dead pieces under the bare zone name 'Dead', which the pool
        # fallback's 'sa dead'/'fp dead' keys never matched — recycled
        # Shadow elites became unpickable and card musters silently
        # dropped.  Side discrimination happens in pick_units.
        keys = keys + ('dead', 'sa casualties' if region_key[0] == 's'
                       else 'fp casualties')
    if region_key == 'sa reinf':
        # SH casualties RECYCLE to reinforcements (rules) but the
        # client kill physically routes them to the Dead pile — the
        # POOL pick must see them there (attempt-25 Many Kings: the
        # near harad placement dropped whole while the recycled
        # southron sat in 'Dead'; census: pool PHANTOM + region
        # short).  FP reinf must NEVER get this alias — FP dead are
        # permanent.  Side discrimination happens in pick_units.
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
    # THE Nth LIVE REGISTRATION SLOT, not compounding arithmetic.
    # The old form bumped die_idx by one per dead reg <= it, which is
    # only equivalent when every removal sits BELOW the query: with
    # dead regs [0, 4] an engine index of 3 became 4 (>=0) and then 5
    # (>=4) — one past a map that ends at 4, and the emission died with
    # 'no die piece for FREE_PEOPLES die 5' mid-battle (attempt 48).
    # Walking the live slots is correct by construction and cannot run
    # off the end; when it does run out, the roster genuinely
    # disagrees, and the error now says so with the whole picture.
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
    rolled.  Both shipped paths already ask the client instead
    (decide_auto and decide_sided emit '!CALL roll_action_dice' and the
    client's own A.k() handler rolls); only the legacy AiBrain.decide
    path still calls this.  Rather than rebuild a path nobody uses,
    HALT - if it really is dead the halt costs nothing, and if it is
    not, the halt says so immediately instead of letting
    self-generated dice pass for client dice.
    """
    if LIVE_MODE or rng is None:
        raise RuntimeError(
            'LIVE HALT: plan_roll would invent action-die faces in '
            'live play. Action dice come from the client '
            "(!CALL roll_action_dice -> the client's own roller) and "
            'the engine INGESTS them; it never sources them. This is '
            'the legacy AiBrain.decide path - use decide_auto (AUTO) '
            'or decide_sided (LIVE_<side>).')
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
    # SHADOW casualties return to the pool (rulebook p.30) — on the
    # client they sit in the SA Dead pile (Game.2041, the corpus kill
    # destination), so a muster may legitimately pull from there once
    # the fresh Reinf pieces of that type are exhausted.
    #
    # FP DEAD ARE PERMANENT (rules; USER precedent, comms 50 Part 35
    # Class A: "FP reinf explicitly NEVER gets this alias").  The old
    # code appended ('fp dead', FP_DEAD) here as a physical-supply
    # borrow — and it RESURRECTED A DEAD FP LEADER onto the board
    # (RULES-agent finding: attempt11_clean_game01 rendered L610,
    # 'moves Gondor Leader from FP Casualties to Dol Amroth'; one
    # occurrence in 70 archived games).  A rules-illegal render is
    # never the lesser evil: the FP branch now has ONE pool, and a
    # genuine physical shortfall raises LookupError, which
    # _emit_engine_action already routes to the board-diff renderer
    # (whose own pool pick is FP-correct by the same Part-35 ruling)
    # with a loud [emit-fallback] line.  No halt, no resurrection.
    pools = [('sa reinf', SH_POOL)] if shadow else \
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
            # SOURCE FROM THE PIECE, NOT THE POOL (structural item).
            # `_pieces_in('sa reinf')` deliberately aliases in the
            # Dead pile because SH casualties recycle to
            # reinforcements - but the emission then claimed
            # `frm = SH_POOL` while the client physically holds the
            # piece at SH_DEAD, so one pid was resolvable from two
            # pools with two different source zones.  That is the
            # pid-316 resurrection class.  The piece's OWN tracked
            # zone is the discriminating predicate, so use it and
            # only fall back to the nominal pool zone when the piece
            # is untracked.
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
    # SAME RULE AS plan_roll, and guarded at ENTRY.  My first placement
    # of this check was further down, after the deck and ledger
    # lookups - which raise LookupError first, so the live guard was
    # unreachable and the test "proving" the halt was actually seeing
    # an unrelated error.  A guard you have to survive three other
    # checks to reach is not a guard.
    if LIVE_MODE:
        # THE FUNNEL, not a halt (P1, USER live session 2 2026-08-16:
        # the AI's own mid-turn event-card draw reached here via the
        # sided AI-slot and the old guard HALTED the game on the
        # human's 'pass').  The guard's invariant is PRESERVED: the
        # CLIENT still chooses the card - this returns the !CALL
        # directive; the client draws with its own RNG and the next
        # tick's log replay ingests the card it named.  Same grammar
        # as the turn-start draws.  plan_use_token's draw half heals
        # through here too.
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
    # THE LEDGER MUST EXIST (PERMANENT item 7).
    # `getattr(replayer, 'ai_drawn_pids', ())` defaulted to an EMPTY
    # TUPLE, which filters nothing - so if the replayer was never
    # stamped, every draw returned `left[0]`: the SAME pid, forever,
    # with the client asked to draw a card it already holds.  A
    # missing ledger is not "no cards drawn yet", it is "we cannot
    # know", and those must not look alike.
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
    n_l = int(params.get('n_leader', 0) or 0)
    n_naz = int(params.get('n_nazgul', 0) or 0)
    if params.get('take_leaders') and not n_l \
            and params.get('n_nazgul') is None:
        # take ALL plain-leader pieces of this side at the source.  For
        # Shadow that includes NAZGUL (their leaders, typed 'nazgul' —
        # the engine moves them under take_leaders, so a move that omits
        # them leaves a phantom leader behind on the client board) —
        # UNLESS the action carries an EXPLICIT 'n_nazgul' split: the
        # engine then passes move_leaders=None to move_army, and its
        # take() treats None as ZERO (the documented 1.150 quirk), so
        # NO plain leader moves either — auto-counting here dragged
        # figures the engine left behind (t115: T1 Barad-Dur move with
        # n_nazgul=0 stranded a phantom engine-side nazgul army; t116
        # seed-7 T6: the Mouth dragged out of Dol Guldur while the
        # engine kept him home).  Corpus logs never carry n_nazgul
        # (engine-enumeration variant only), so the L1/L2 roundtrip
        # gates are byte-identical either way.
        for pid in _pieces_in(replayer, frm_key):
            nat, ut = _unit_type(pid, replayer, side_hint=side)
            if ut == 'leader' and _side_of_nation(nat) == side:
                n_l += 1
            elif ut == 'nazgul' and side == 'shadow':
                n_naz += 1
    chars = [str(c) for c in params.get('characters', ())]
    if params.get('take_leaders') and params.get('n_nazgul') is None:
        # the engine's take_leaders also carries CHARACTER leaders: a
        # minion (Witch-king / Mouth — Saruman never leaves Orthanc)
        # or a separated companion leading the army.  Omitting them
        # left a phantom leader behind on the client (WK stranded at
        # Osgiliath while the engine marched him on).  With an EXPLICIT
        # n_nazgul the engine moves ONLY those nazgul (move_leaders
        # None = zero — see above), so minions/companions stay home.
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
        # the ENGINE's exact per-(nation, kind) moved composition —
        # splitting a MIXED army with a nation-blind pick shipped an
        # Isengard regular where the engine moved a Sauron one (t131
        # game 10: Parth Celebrant swap, engine-vs-client 4 issues)
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
    # SIEGE (attempt-77 audit): units CAN move in to join a besieged
    # garrison - game_logic's own comment says "the 5-unit siege cap
    # only applies to units joining the garrison side inside a
    # besieged stronghold" - so the field zone is not automatically
    # right here.  The bridge's board-diff path already asks via
    # _siege_dest_zone; this planner did not.  victor_advance
    # delegates here, so it is covered by the same change.
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
    # constant (240 Grey->White, 241 Strider->Aragorn).  The parser's
    # COMPANIONS table renames 241 at crowning but NEVER renames 240,
    # so 'Gandalf the White' fell through this lookup forever and the
    # by-name reconciler silently skipped his moves (attempt-9 game-1:
    # client Gandalf stranded at dol guldur while the engine walked
    # him to parth celebrant, with nation leader 068 shipped in his
    # place).  Resolve every alias face to its pid.
    for p, faces in getattr(lr, 'PIECE_NAME_ALIASES', {}).items():
        for f in faces:
            if str(f).strip().lower() == want:
                return p
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


def _piece_zone_code(replayer, pid, default=None):
    """Client zone code where the replayer believes a piece sits."""
    zone = replayer.piece_region.get(str(pid)) \
        or _initial().get(str(pid))
    if not zone:
        if default:
            return default
        raise LookupError('no known zone for piece %r' % (pid,))
    zk = _norm_zone(zone)
    if zk == 'dead':
        # BARE-'Dead' TRACKING (synth probe18 wedge, 2026-08-16): the
        # parser's kill/dkill routing stores casualties under the
        # bare zone name 'Dead', which the client zone table (built
        # from the decode tables) has no key for - so a muster that
        # legally recycles a Shadow casualty (rulebook p.30) died on
        # 'no client zone code for region dead'.  The pid decides
        # WHICH dead pile: FP pieces -> FP Casualties, Shadow ->
        # SA Casualties.
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
# That single sentence was written out longhand in EIGHT places across
# two modules, and the copies had drifted:
#
#   * six were faithful transcriptions - the bug-per-copy risk without
#     even the excuse of a difference;
#   * one asked it by COMBAT ROLE ("defender and besieged") instead of
#     by garrison side, which inverts during a SORTIE - when the
#     garrison attacks, the defender is the besieger out in the field,
#     and that copy routed the besieger's casualties into the
#     stronghold box (see plan(), casualty selection);
#   * one arrival site (transform_gandalf) never asked at all, which
#     put Gandalf the White among the besiegers of Lorien and ran 83
#     settled census rows before it was caught;
#   * and the two modules spoke different vocabularies for the same
#     answer - 'FREE_PEOPLES'/'SHADOW' here, 'fp'/'shadow' there - so
#     even a correct copy could compare unequal-but-identical answers.
#
# WHICH STATE ANSWERS.  These take `state` explicitly rather than
# reaching for one, because the honest answer differs by caller: the
# server holds the ENGINE's state and must use it (engine is master,
# client renders), while an emitter planner only ever holds the
# replayer's client-side mirror.  Passing it in lets both ask the SAME
# function without pretending the emitter has access it does not have.
#
# NO BLANKET except: - every copy above ended in `except Exception:
# pass`, which turns "combat.garrison_side_for is broken" into "the
# piece quietly lands in the field zone" - a divergence that then has
# to be rediscovered from a census row.  Only the genuinely expected
# failure is caught: a region with no ' stronghold' sub-box defined.
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
    # SIEGE (attempt-77 audit): a companion arriving at a BESIEGED
    # friendly stronghold joins the garrison INSIDE - the field zone
    # belongs to the besieger.  The BRIDGE's own companion path
    # already asks this (ai_move_server _companion_move_lines ->
    # _siege_dest_zone); this planner did not, so the two
    # implementations of one question disagreed.  Same pattern as
    # plan_nazgul_move directly below.
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
    # SIEGE (attempt-25): nazgul arriving at a BESIEGED friendly
    # stronghold join the garrison INSIDE the sub-box - the field zone
    # belongs to the besieger.  Nazgul are always Shadow, so the side
    # is a constant here; the QUESTION is not, and it is asked in the
    # one place now.
    _toz9 = garrison_arrival_zone(replayer.state, _to9, 'SHADOW')
    return ('move', {'nick': nick, 'pieces': pids,
                     'frm': zone_code_for(frm),
                     'to': _toz9})


def plan_companion_separate(replayer, nick, side_name, params):
    """Separate a companion to a region.  The guide reassignment (when
    the guide is the one leaving) is IMPLICIT in the client -- humans
    emit no extra line -- so the parser auto-reassigns it (log_replay
    change_guide, mirroring the engine); we emit only the move."""
    # THE GROUP FORM IS THE COMMON ONE (offline rig instrument,
    # 2026-08-17: 761 of 1119 emit-fallbacks over 900 chaos games were
    # this KeyError).  ai.py builds a SINGLE separation as
    # params['companion'] and a group separation - the rulebook's
    # normal case, since companions of equal level leave together - as
    # params['companions'] (a LIST).  This planner only ever read the
    # singular key, so every group separation raised KeyError and fell
    # back to the raw board diff: the corpus gesture (one drag, all
    # figures) was never emitted for the majority of separations.
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
    # SIEGE (attempt-77 audit): same question as plan_companion_move -
    # a companion separated into a BESIEGED friendly stronghold joins
    # the garrison, not the besiegers.
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
    # THE REPEAT GUARD IS A LEDGER-vs-COUNT CHECK, NOT A BOARD CHECK.
    # (BRIDGE-11, fixing a regression I caused and the round-trip rig
    # caught within one battery.)  My first cut refused whenever ANY
    # already-paid ring still showed in an active slot - i.e. whenever
    # the board lagged the ledger.  In the rig the board lags BY
    # DESIGN, so every ring use after the first raised, the action
    # emitted nothing, and the engine advanced while the board did not:
    #     [emit-fallback] elven_ring: LookupError(...) -> 0 diff lines
    #     seed 824000 T2 elven_ring | SILENT-STEP dice FREE_PEOPLES
    # The OLD guard never claimed that.  It refused only when the ring
    # it was ABOUT TO WALK was already paid - which, because the pid
    # came from the count, happens exactly when the LEDGER HAS RUN
    # AHEAD OF THE COUNT: the caller is re-emitting one logical use.
    # That is the real condition, so it is stated directly.  WORDING IS
    # PART OF THE CONTRACT: test_permanent_class.py's "a repeat walk of
    # the SAME ring is refused" asserts the phrase 'already been
    # walked', and the pin is HONOURED, never edited to fit a rewrite.
    if paid is not None and used is not None and len(paid) >= used + 1:
        raise LookupError(
            'an FP elven ring has already been walked to its Used Ring '
            'slot for this use (ledger=%s, but the engine has spent '
            'only %d) - refusing to walk a second ring for one use'
            % (' '.join(sorted(paid)) or '(none)', used))
    avail = [(p, z) for p, z in sorted(where.items())
             if z in RING_ACTIVE_SLOTS and not (paid and p in paid)]
    if not avail:
        # NAME THE REAL CAUSE.  "no ring is in a slot" and "every ring
        # is already in my ledger" are different facts with different
        # fixes, and a halt message that states the wrong one costs the
        # next reader the whole diagnosis.
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
    # shown yet.  With a board that tracks, lag is 0 and this is simply
    # "the first free slot"; with a board that lags it reproduces the
    # old `2011+used` arithmetic exactly.  Both worlds, one expression.
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
    """Ring active->used walk (FP only; a Shadow collected-ring use is
    proven by the die_change alone) + the die-face change line."""
    ops = []
    state = replayer.state
    if side_name == 'FREE_PEOPLES':
        _paid = params.get('_ring_paid')
        _used9 = 3 - int(getattr(state, 'fp_elven_rings', 3))
        pid, _frm, _to = pick_fp_ring_walk(replayer, _paid, _used9)
        # COUNTS MUST AGREE (USER 2026-08-19, invariant 2: no phantom
        # ring, none spent twice, none stranded).  The gesture follows
        # the BOARD because only the board can be walked - but a count
        # that points at a ring the board does NOT have in a slot is a
        # tracking break, and a tracking break never said out loud is
        # how auto69 game 5 ran to the end.  Said, not raised: refusing
        # would block a legal ring use over bookkeeping the gesture
        # itself is not wrong about.
        # Narrow on purpose: only when the count's ring is ABSENT from
        # the slots, never merely when the choice differs.  A board
        # that simply lags an emitted walk is normal (the round-trip
        # rig lags by design) and must not print a false alarm - and
        # under the USER's ruling a different CHOICE is not a defect at
        # all: "which ring is irrelevant as long as it is cleanly
        # tracked".
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
    # A7 (USER ruling 2026-08-19): THE RING NO LONGER CHANGES A DIE'S
    # FACE, so this planner must no longer render one.  The engine
    # raises a legality waiver and leaves the die as rolled; forging a
    # die-face line here would tell the client - and our own parser -
    # about a change that did not happen.  CAUGHT BY THE ROUND-TRIP
    # RIG, not by reading: `test_roundtrip_rig_gate.py` went red the
    # moment the engine stopped changing the face -
    #     seed 824000 T2 elven_ring {'new_face': 'Character'}
    #     dice FREE_PEOPLES: engine=(('EVENT','EVENT'),2)
    #                     roundtrip=(('CHARACTER','EVENT'),2)
    # which is the emitted face-change line being replayed by the
    # parser against an engine that never made it.
    # AND IT WAS NEVER THE CLIENT'S GRAMMAR EITHER: the client's ring
    # command emits a marker walk and an optional flip - no die pid, no
    # face - and over 300 human logs, 540 ring uses produced ONE nearby
    # die-face re-render against 2.6 expected by chance.
    #
    # THE ONE SURVIVOR is the Shadow ring-to-EYE use, where the engine
    # still changes the face because the die's move into the Hunt Box
    # is a ZONE MOVE the client does render.
    #
    # FP legality use  -> the marker walk above IS the whole gesture.
    # SH legality use  -> no ops here; the per-action ELVEN-RING DELTA
    #   RECONCILER in ai_move_server (`[ring-dispose]`) emits the Dark
    #   Ring -> SA Casualties walk off the `sh_elven_rings` delta, and
    #   it is the ONE ledger for that departure.  Emitting a second
    #   gesture here would be the auto69 double-departure again.
    face = str(getattr(params.get('new_face'), 'value',
                       params.get('new_face'))).upper() \
        .replace(' ', '_').replace('/', '_')
    # CRITIQUE NEW-2: the hybrid face's engine VALUE is 'Muster/Army'
    # — the slash survived the space normalization and the H letter
    # (992 corpus 'to H' occurrences) was unreachable; the full name
    # forms alias to the map keys too
    face = {'EYE_OF_SAURON': 'EYE',
            'WILL_OF_THE_WEST': 'WILL_OF_WEST'}.get(face, face)
    # RIG CLASS-A (155x at 300-game scale, 2026-08-17): an Eye result
    # BANKS - the engine puts the changed die in the Hunt Box, so the
    # face-change line alone leaves the parser's die in the pool
    # (dice SHADOW: engine=(...), N+1 used vs roundtrip=(...'EYE'...),
    # N).  Emit the physical drag too, the same pool->huntbox grammar
    # as the eye allocation.  SHADOW only - no FP face is the Eye.
    if side_name == 'SHADOW' and face == 'EYE':
        letter = _FACE_LETTER.get(face)
        if letter is None:
            raise LookupError('no die-change letter for face %r'
                              % (face,))
        dp = die_piece_for(replayer, side_name,
                           params['target_die_idx'])
        ops.append(('die_change', {'nick': nick, 'die': dp,
                                   'face': letter}))
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
        ops.append(plan_companion_move(replayer, nick, side_name,
                                       params))
        return ops
    if action_type == 'companion_separate':
        ops.extend(plan_companion_separate(replayer, nick, side_name,
                                           params))
        return ops
    if action_type == 'nazgul_move':
        ops.append(plan_nazgul_move(replayer, nick, side_name, params))
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
        # G13 (USER 2026-08-14 gap wave): the pawn's TRUE region first —
        # the parser's fellowship tracker, not piece_region (the 247
        # walks ride the fellowship handler, which never updates
        # piece_region, so the old chain fell through to _initial() =
        # 'from Rivendell' forever)
        # PAWN CONTINUITY (CRITIQUE #4): the LIVE engine's 'from' is
        # its last DECLARED position (Rivendell for whole stretches),
        # so preferring it teleported the pawn back to Rivendell on
        # every reveal (a14 renders 92/181: frm=Game.1911 forever).
        # The parser's fellowship_location follows the pawn's rendered
        # walks and equals 'from' at parser-record time — it leads.
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
                # down the track (2023-08-13 HombreLobo L544: '248
                # Game.2016 -> Game.2014' right after the declare walk).
                # No human log contains a backward Controls.790
                # "advance"; that form also leaves log_replay's
                # _pre_reset_progress stash dangling, inflating the next
                # declare's progress_consumed.
                ops.append(('move', {'nick': nick, 'pieces': ['248'],
                                     'frm': 'Game.%d' % (2014 + p),
                                     'to': 'Game.2014'}))
            # marker dance: humans flip-and-unflip the fellowship marker
            # when declaring (O.1 bookkeeping pair; ketchuptime
            # 2023-11-03 L102-103) and log_replay's declare branch arms
            # _skip_next_fsp_flip=2 for EXACTLY this pair.  Without it
            # the parser silently eats the next two GENUINE flips, so a
            # later hunt/card reveal showed hidden for many ticks.
            ops.append(('flip', {'nick': nick, 'pieces': ['247']}))
            ops.append(('flip', {'nick': nick, 'pieces': ['247']}))
        else:
            # hunt-forced reveal placement: corpus choreography is reset
            # -> walk -> flip (2024-02-05 Polijuna_The board is set
            # L1326-1328: '248 Game.2015->Game.2014', '247
            # Game.1935->Game.1943', Controls.323 flip).  The reset must
            # ride the planner so the MANUAL decide() path (which never
            # runs _track_delta_lines) stays in sync too.  from_progress
            # is injected by the AUTO driver (engine truth); fall back
            # to the parser's counter for decide().
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
            # the transformation is BY DEFINITION a return from the
            # dead pile (GenericPie L213 drags 240 FROM Game.2039); a
            # stale tracked zone like the Guide Box routes the drag
            # into the parser's fellowship-companion handler, where
            # dead Gandalf matches nothing and the White never appears
            # (t116 ga11 T123: fangorn entry missing parser-side)
            frm = FP_DEAD
        # ASK THE GARRISON QUESTION (attempt 77).
        # This was a bare field-zone lookup: Gandalf the White returns
        # to an FP Elven stronghold, Lorien can be BESIEGED when he
        # does, and he must join the garrison INSIDE rather than
        # appear among the besiegers.  The site did not ask the
        # garrison-side question at all - it is not a second copy that
        # disagreed, it is a third site that never consulted the rule.
        # The server injects '_besieged_garrison' the same way it does
        # for musters, because only the ENGINE knows the siege state.
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
        ops.append(('move', {'nick': nick, 'pieces': [pid],
                             'frm': frm, 'to': GUIDE_BOX}))
        return ops
    if action_type == 'combat_card_none':
        ops.append(('chat', {'nick': nick, 'msg': 'nc'}))
        return ops
    if action_type == 'combat_effect_choice':
        # only the Dread & Despair leadership forfeit has a dedicated
        # chat grammar ('ffN'); other magnitude choices are drags that
        # are not wired yet
        cid = str(params.get('card_id', '')).lower()
        if 'dread' in cid:
            ops.append(('chat', {'nick': nick,
                                 'msg': 'ff%d' % int(
                                     params.get('count', 1) or 1)}))
            return ops
        raise NotImplementedError(
            'combat_effect_choice drags for %r not wired' % (cid,))
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
    if action_type in ('casualty_allocate', 'siege_press'):
        state = replayer.state
        pc = getattr(state, 'pending_combat', None) or {}
        side = 'shadow' if side_name == 'SHADOW' else 'fp'
        defender = False
        if action_type == 'siege_press':
            # press = downgrade ONE attacker elite to keep assaulting
            region = _norm_zone(pc.get('attacker_region')
                                or pc.get('region', ''))
            k, hits = 1, 0
        else:
            stage = pc.get('alloc_stage')
            if stage == 'attacker':
                region = _norm_zone(pc.get('attacker_region', ''))
                hits = int(pc.get('atk_hits', 0) or 0)
            else:
                defender = True
                region = _norm_zone(pc.get('defender_region', ''))
                hits = int(pc.get('def_hits', 0) or 0)
            k = int(params.get('count', 0) or 0)
        if not region:
            raise LookupError('no pending_combat region for %s'
                              % action_type)
        # A besieged GARRISON's units live in the stronghold sub-box.
        #
        # CORRECTED while unifying: this asked `defender and besieged`,
        # i.e. it inferred the garrison from the COMBAT ROLE.  That
        # holds for an assault, where the garrison defends - but it
        # inverts during a SORTIE, where the garrison attacks and the
        # DEFENDER is the besieging army standing in the field.  On a
        # sortie the old test sent the besieger's casualties into the
        # stronghold sub-box, selecting pieces from the wrong zone.
        # garrison_arrival_zone asks by garrison SIDE, which is right
        # for both (combat.garrison_side_for reads the settlement's
        # controller, not who happens to be attacking).
        zc = garrison_arrival_zone(state, region, side)
        chosen = []
        if k:
            elites = pick_units(replayer, region, side, n_elite=k,
                                side_hint=side)
            # AUTO-96 game-4 (2026-08-17, the physical-supply
            # phantom): the ENGINE's downgrade-source decision is
            # THE answer (combat._consume_downgrade_source stamps
            # (nation, 'cas'|'pool'|None) per downgrade; the server
            # threads them in as _dg_srcs).  The old blind zone
            # order ('dead first, any nation') disagreed with an
            # engine 'pool' sourcing and left the client pool one
            # HIGH forever (pool fp rohan/regular engine=4 client=5
            # PHANTOM).  Obey when told; the legacy order remains
            # only as the un-threaded fallback (teacher/live paths
            # follow the human's physical drags instead).
            regs = None
            _dgs9 = params.get('_dg_srcs') or []
            if _dgs9:
                regs = []
                _dz9 = ('sa dead' if side == 'shadow' else 'fp dead')
                _rz9 = ('sa reinf' if side == 'shadow'
                        else 'fp reinf')
                for _nat9, _src9 in _dgs9[:k]:
                    if _src9 is None:
                        print('[emitter] downgrade with NO source '
                              '(%s): elite dies outright - no '
                              'replacement placed' % _nat9)
                        continue
                    _zone9 = _dz9 if _src9 == 'cas' else _rz9
                    try:
                        regs += pick_units(replayer, _zone9, side,
                                           n_regular=1,
                                           side_hint=side,
                                           nation=_nat9,
                                           exclude=regs)
                    except LookupError as e:
                        # the engine and the client board disagree on
                        # the replacement supply: surface it (a
                        # nation-blind pick here places a phantom)
                        raise LookupError(
                            'engine-stamped downgrade source %r has '
                            'no %s regular on the client board: %s'
                            % (_zone9, _nat9, e))
            if regs is None:
                for _pk9 in (('sa dead', 'sa reinf')
                             if side == 'shadow'
                             else ('fp dead', 'fp reinf')):
                    try:
                        regs = pick_units(replayer, _pk9, side,
                                          n_regular=k,
                                          side_hint=side)
                        break
                    except LookupError:
                        continue
            if regs is None:
                raise LookupError(
                    'no replacement regulars for %s downgrade' % side)
            # op-name convention (matches ai_move_server's downgrade
            # path and log_replay's dispatch): swap_place renders
            # Controls.304 = REMOVE the on-board elites; swap_remove
            # renders Controls.313 = PLACE the reserve regulars.
            # (FIX 2026-07-27: this branch had the two inverted.)
            ops.append(('swap_place', {'nick': nick, 'pieces': elites,
                                       'to': zc}))
            if regs:
                # all-None sources leave regs empty (elites died
                # outright, nothing placed) - never emit a bare
                # swap_remove
                ops.append(('swap_remove', {'nick': nick,
                                            'pieces': regs,
                                            'sub': '308',
                                            'subtext': '',
                                            'to': zc}))
            chosen.extend(elites)
        h = max(0, hits - k)
        if h:
            # THE engine allocation order (ai._apply_allocation): the
            # remaining hits kill REGULARS first; then each elite
            # absorbs 2 hits as an outright kill; a single leftover hit
            # on an elite is a DOWNGRADE (the swap, sourced from the
            # engine's next _dg_src_log stamp; a None source = the
            # elite dies outright, nothing placed - USER R7).  Kinds
            # must match the engine's removal exactly: a missing piece
            # HOLDS (LookupError), never substitutes another kind.
            def _avail(kind):
                out_ = []
                for pid in _pieces_in(replayer, region):
                    if pid in chosen or pid in kills or pid in swap_out:
                        continue
                    nat, ut = _unit_type(pid, replayer, side_hint=side)
                    if ut == kind and _side_of_nation(nat) == side:
                        out_.append(pid)
                return out_
            kills, swap_out, swap_in = [], [], []
            for pid in _avail('regular'):
                if h <= 0:
                    break
                kills.append(pid)
                h -= 1
            while h > 0:
                el = _avail('elite')
                if not el:
                    raise LookupError('need %d more casualt%s (%s) in '
                                      '%r, found no %s'
                                      % (h, 'y' if h == 1 else 'ies',
                                         side, region,
                                         'elite' if _avail('regular')
                                         == [] else 'regular'))
                if h >= 2:
                    kills.append(el[0])
                    h -= 2
                    continue
                # leftover single hit: the downgrade swap
                _dgs9 = list(params.get('_dg_srcs') or [])[k:]
                swap_out.append(el[0])
                if _dgs9:
                    _nat9, _src9 = _dgs9[0]
                    if _src9 is None:
                        print('[emitter] downgrade with NO source '
                              '(%s): elite dies outright - no '
                              'replacement placed' % _nat9)
                    else:
                        _zone9 = (('sa dead' if side == 'shadow'
                                   else 'fp dead') if _src9 == 'cas'
                                  else ('sa reinf' if side == 'shadow'
                                        else 'fp reinf'))
                        try:
                            swap_in += pick_units(replayer, _zone9, side,
                                                  n_regular=1,
                                                  side_hint=side,
                                                  nation=_nat9,
                                                  exclude=swap_in)
                        except LookupError as e:
                            raise LookupError(
                                'engine-stamped downgrade source %r '
                                'has no %s regular on the client '
                                'board: %s' % (_zone9, _nat9, e))
                else:
                    raise LookupError('leftover-hit elite downgrade at '
                                      '%r has no engine source stamp '
                                      '(_dg_srcs) - holding' % region)
                h -= 1
            if kills:
                ops.append(('hand_exit_b', {'nick': nick,
                                            'pieces': kills, 'frm': zc}))
            if swap_out:
                ops.append(('swap_place', {'nick': nick,
                                           'pieces': swap_out, 'to': zc}))
                if swap_in:
                    ops.append(('swap_remove', {'nick': nick,
                                                'pieces': swap_in,
                                                'sub': '308',
                                                'subtext': '',
                                                'to': zc}))
        return ops
    if action_type == 'hunt_take_corruption':
        # FROM = where the client's marker physically sits (parser);
        # TO = the ENGINE's landing spot when the server injects
        # corruption_target (the parser never reconstructs
        # pending_hunt, so the old damage fallback always walked ONE
        # step — a '3' tile rendered as +1).  The engine target also
        # self-heals any earlier marker drift.
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
        # FLIP ONLY (t123).  The engine KEEPS the progress counter on a
        # Gollum reveal (ai.py 'hunt_gollum_reveal': hidden=False, no
        # progress touch — only fellowship_reveal_place/declare reset).
        # The old reset drag here ('248 -> Game.2014') misstated the
        # engine state, and _track_delta_lines then "corrected" it back
        # up with consecutive Controls.790 lines — which corpus grammar
        # defines as fellowship MOVE button presses (wotr2026_7_17_11_26
        # T7 L464-467 / T8 L540-544: 9 phantom fellowship_move records
        # refused by the certifier).  A reveal that keeps progress
        # renders as the marker flip alone.
        ops.append(('flip', {'nick': nick, 'pieces': ['247']}))
        return ops
    if action_type == 'eye_allocate':
        # one line per eye die moved to the hunt box.  count=0 is the
        # engine's LEGAL zero-eye choice -> zero ops (the old `or 1`
        # coerced it to a phantom eye line).
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
    # REMOVED 2026-08-16: the L2 'die_to_hunt' planner.
    #
    # It rendered Controls.795 with frm=SH_DICE_POOL (Game.2043) - the
    # V1 grammar violation, x89 across the certified run, which humans
    # never emit (they drag Scribe.141 2043->1997). The two LIVE
    # producers were converted earlier today; this one survived because
    # nothing calls it: no engine action of type 'die_to_hunt' exists
    # (verified across src/ai.py, game_logic.py, game_state.py - zero
    # hits), so it was parser-record-only and could never fire.
    #
    # Deleted rather than converted: an unreachable producer of a known
    # bad form is a loaded gun for whoever next adds that action type,
    # and "it is dormant" is the same argument that left the dwarven
    # ring inventing its own randomness for months.
    #
    # Controls.795 is still correct and still emitted for the FP
    # FELLOWSHIP die out of Game.2035 - the four live sites all source
    # from DICE_USED, which is the human form. The violation was only
    # ever the SHADOW-pool source.
    raise NotImplementedError('no L2 planner for %r yet' % action_type)
