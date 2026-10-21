#!/usr/bin/env python3
"""Per-game curation registry (fidelity-campaign waiver metadata).

Keyed by log filename.  The teacher-era schedule machinery
(schedule_fixups / parser_gap_suppress / card_choices / ...) was
retired with closed_loop_test.py; the LIVE consumer is
wotr_bridge/engine_check.py, which reads exactly two keys:

  play_requirement_waivers: set[(line:int, normalized_card_name)]
      plays executed DESPITE a failed printed requirement -- logged
      human misplays; the log is ground truth.

  minion_entry_waivers: set[(line:int, normalized_minion_name)]
      minion musters whose engine entry condition does not hold at
      that state -- logged human mistakes (user ruling 2026-07-04).

The remaining keys (concession, casualty_spec_regions,
combat_hit_caps, starting_resources, casualty_policy_overrides,
curated_discards, companion_queue_default) are retained as inert
per-game evidence records; nothing reads them today.
get_curation() fills fresh empty defaults per call.
"""
import os


# ===================================================================
# Game 0: 2023-01-02_144127_FP_GenericPie_SP_Sirrus233
# (T9 Shadow military win, VP 10, corr 8, mordor_step 3 — certified)
# ===================================================================


CURATION = {
    '2023-01-02_144127_FP_GenericPie_SP_Sirrus233.log': {
        # Each entry verified against data/game0_readable_reference.txt:
        #   (241, fellowship_move) - T3 move that the human UNDID ("3 undos"
        #         in the log; undo pre-pass drops the piece line but the
        #         progress-track record survives)
        #   (833, mordor_move) - T7 mordor step-back record: Shelob's Lair
        #         stop tile — the engine reverts the step itself inside
        #         move_fellowship
        #   (838/842, play_event) - T7 duplicated 'There is Another Way'
        #         plays from the logged out-of-turn undo confusion (the
        #         real play comes later)
        #   (443, army_move) - T4 'West Harondor -> Pelargir': the victor's
        #         post-battle advance of the (injected) Pelargir attack —
        #         the engine advances the attacker itself after the
        #         defender retreats
        #   (1265, play_event) - T9 'The Lidless Eye': played die-less
        #         AFTER the final Dol Amroth casualty had already sealed
        #         the Shadow military victory at 10 VP — a post-victory
        #         formality, not a game decision
        'parser_gap_suppress': {
            (241, 'fellowship_move'), (443, 'army_move'),
            (833, 'mordor_move'), (838, 'play_event'),
            (842, 'play_event'), (1265, 'play_event'),
        },
        # The human's actual in-game choices for engine card effects
        'card_choices': {
            'anewpowerisrising': {'unit_combo': '2elite'},
            'celebornsgaladhrim': {'unit_type': 'elite'},
            'imrahilofdolamroth': {'unit_type': 'elite'},
            'eomersonofeomund': {'region': 'Westemnet',
                                 'unit_type': 'elite'},
            'cirdansships': {'region': 'Dol Amroth',
                             'unit_combo': '2elite'},
            'rageofthedunlendings': {'recruit_region': 'Gap of Rohan'},
        },
        # Verified against data/game0_readable_reference.txt:
        # - T3 MT assault2: "SP kills Sauron Elite, returns Regular" — the
        #   Shadow DOWNGRADED (battle #3 at minas tirith; #1 = field
        #   attack, #2 = assault1 where SP killed 3 regulars = default
        #   preserve_elites).
        # - T5 Fords of Isen: "FP loses 2 Rohan E ... retreats 2L+3R" —
        #   4 hits on 3R+2E = two downgrades + two regular kills
        #   (preserve_units).
        # - T5 DA assault1: "1 hit each" then end-of-battle Shadow
        #   composition 3R+0E requires the assault1 hit to DOWNGRADE
        #   (6R2E->7R1E), press (->8R0E), then 5 kills (->3R0E).  Battle
        #   #2 at dol amroth (#1 = the T4 attack that put DA under siege).
        # - T6 Lorien assault (battle #2 there): Durin's Bane pre-combat
        #   3 hits -> "FP kills E, returns R, kills E" = one full elite
        #   removal (2 hits) + one downgrade = 'spend_elites' (FP rolls 3
        #   dice in the main round).
        # - T7 Helm's Deep assault (battle #2): "SH loses Isengard E" on
        #   2 hits with 5 regulars alive = full elite removal =
        #   'spend_elites'.
        'casualty_policy_overrides': {
            ('minas tirith', 'SHADOW', 3): 'preserve_units',
            ('fords of isen', 'FREE_PEOPLES', None): 'preserve_units',
            ('dol amroth', 'SHADOW', 2): 'preserve_units',
            ('lorien', 'FREE_PEOPLES', 2): 'spend_elites',
            ('helm s deep', 'SHADOW', 2): 'spend_elites',
        },
        # T5 "SH kills Wormtongue in hand (discard)" — the only genuine
        # hand-limit discard in the log's hand-diff stream
        'curated_discards': ('wormtongue',),
        'companion_queue_default': ['Meriadoc', 'Boromir', 'Legolas'],
    },
}


# ===================================================================
# Game 1: 2023-12-07_001647_FP_Mol_SP_Pineirin
# (report: Shadow Ring victory; report 'turns'=1 is a DATA ERROR — the
#  log plays 9 full turns, ending T9 when Isildur's Bane's hunt-tile
#  corruption seals the Shadow ring victory)
# ===================================================================


CURATION['2023-12-07_001647_FP_Mol_SP_Pineirin.log'] = {
    # T7 Westemnet field battle (lines 565-585): SH scored 2+3 hits
    # (Shield-Wall cancels 1); the FP force 2R+2E+1L survived as 1R+1L
    # and RETREATED — only possible if the FP DOWNGRADED elites
    # (preserve_units): 2 downgrades + 2 kills + Great Host's post-
    # casualty auto-hit = 1R+1L remaining, matching the logged retreat.
    # SH took FP's 2 hits as downgrades likewise (the advance keeps 4
    # army units).
    'casualty_policy_overrides': {
        ('westemnet', 'FREE_PEOPLES', None): 'preserve_units',
        ('westemnet', 'SHADOW', None): 'preserve_units',
    },
    # T7 Westemnet: SH rolled 2+3 hits, Shield-Wall cancels 1 -> 4; the
    # humans did NOT apply Great Host's post-casualty automatic hit (the
    # FP survivor 1R+1L retreated) — cap the round at 4.
    'combat_hit_caps': {
        ('westemnet', 'SHADOW', None): 4,
    },
    # (763, mordor_move) - T9 step-back record 3->2 after the Shelob's
    #     Lair stop tile at step 3: the engine reverts the step itself
    #     inside move_fellowship (same parser gap as game 0 line 833)
    # (309, eye_allocate) + (312, die_change) - T5 logged MISCLICK: the FP
    #     player pressed the eye-allocate button on SH die 103, hit UNDO
    #     (line 310, Controls.819) with chat "my bad", and restored the
    #     face with "changes 103 to P".  The line-by-line replay paths do
    #     not process undo buttons — suppress both residue records.
    # (632, combat_start) - duplicate callout of the never-fought FoI
    #     battle (see the line-632 rewrite in _game1_schedule_fixups)
    # (459, play_event) - T6 'Morgul Wound' play RETRACTED: Mol replied
    #     "not revealed" (line 460) and Pineirin played Half-orcs instead;
    #     the same card 193 is legally played at T8 line 720
    'parser_gap_suppress': {
        (763, 'mordor_move'),
        (309, 'eye_allocate'),
        (312, 'die_change'),
        (632, 'combat_start'),
        (459, 'play_event'),
    },
    # The human's actual in-game choices (each verified in the decoded
    # record stream, data/_g1_records.txt):
    #   Half-Orcs T6 line 465: "muster Lorien 1 elite" (card_effect)
    #   Dreadful Spells T5 line 362: the only Nazgul/FP contact is the
    #       Lorien siege (SH besiegers with Nazgul + FP garrison)
    'card_choices': {
        'halforcsandgoblinmen': {'region': 'Lorien', 'unit_type': 'elite'},
        'dreadfulspells': {'target_region': 'Lorien',
                           'target_fp_army': 'Lorien'},
        # T2 line 136-138: "2R Dol Guldur / 2R Moria / 2R Mount Gundabad"
        # (engine default is Barad-dur/DG/MM — human chose differently)
        'pitsofmordor': {'strongholds': ['Dol Guldur', 'Moria',
                                         'Mount Gundabad']},
        # T6 line 456: Riders muster is 1 ELITE + leader at Edoras
        # (engine default recruits a regular)
        'ridersoftheoden': {'unit_type': 'elite'},
    },
}

# ===================================================================
# Game 2: 2023-12-06_192618_FP_Rizino1_SP_Pineirin
# (report: Shadow Military, 'turns=1' is a data error — the log plays
#  10 turns)
# ===================================================================


CURATION['2023-12-06_192618_FP_Rizino1_SP_Pineirin.log'] = {
    # T10 line 1058: Pineirin declares 'attack dale' and says 'nc' —
    # then RETRACTS the decline ('a forgot about the king', 'ok from
    # the north then i willl play card') and reveals ORC PATROL at line
    # 1067/1069 for that same battle.  The stale cc_none record shifted
    # the whole SH combat-card queue one battle late (Orc Patrol landed
    # on the FP dale counter-attack, The Shadow Lengthens vanished, The
    # King is Revealed hit Dol Amroth round 1 instead of round 2 and
    # its Relentless self-hit choice missed the pending phase).
    'parser_gap_suppress': {(1058, 'combat_card_none')},
    # The human's actual in-game choices (each verified in the decoded
    # record stream / raw log):
    #   Shadows on the Misty Mountains T2 line 165/166: "muster Moria
    #       2E + 1 Nazgul" (engine default is Mount Gram / regulars)
    #   Rage of the Dunlendings T2 line 172/173: recruit region MORIA
    #       (the 2 Isengard regulars land there; the Dunland units'
    #       moves are card-window records the harness replays itself)
    #   Stormcrow T3 line 215/221: Elves retreated 2->1 (fellowship at
    #       Rivendell = inside Elven borders)
    #   Horde from the East T7 line 705/706: "5R East Rhun"
    #   Wisdom of Elrond T4 line 350/351: activated + advanced ROHAN
    'card_choices': {
        'shadowsonthemistymountains': {'region': 'Moria',
                                       'unit_combo': '2elite'},
        'rageofthedunlendings': {'recruit_region': 'Moria'},
        'stormcrow': {'nation': 'elves'},
        'hordefromtheeast': {'region': 'East Rhun'},
        'wisdomofelrond': {'nation': 'rohan'},
    },
}


# ===================================================================
# Game 3: 2023-03-15_030858_FP_sharpz_SP_complexflips
# (Free Peoples military victory at exactly 4 VP, T3 — complexflips
#  concedes 'welp I guess thats gg'; FP VP marker 422 walks 0->2 at
#  line 171 and 2->4 at line 238; SH marker 423 never moves)
# ===================================================================


CURATION['2023-03-15_030858_FP_sharpz_SP_complexflips.log'] = {
    # T2 line 136: the 2E+1N Moria muster is the line-137 card's effect,
    # dragged before the reveal (see _game3_schedule_fixups)
    # T3 line 214: duplicate combat_start synthesized from the Hollin ->
    # Moria battle_advance — the field battle already opened at line 213
    # and the REAL assault is the die-consuming attack at line 230
    # (Controls.840 + die 212).  Same pattern as game 1's line-632
    # duplicate.  Unsuppressed it stole the assault's roll-chunk span
    # (69,71) and forced a phantom extra siege ask (unmemorizable
    # conflicting sample).
    'parser_gap_suppress': {(136, 'muster'), (214, 'combat_start')},
    # T2 line 136/137: 'Shadows on the Misty Mountains' — the human chose
    # MORIA and 2 ELITES (+ the printed Nazgul); engine default is Mount
    # Gram / 2 regulars (same choice pattern as game 2 T2 line 165/166)
    'card_choices': {
        'shadowsonthemistymountains': {'region': 'Moria',
                                       'unit_combo': '2elite'},
    },
    # BID COMPENSATION (first game in the campaign with tokens in play):
    # the FP token pieces 426 (Hunt for the Ring Political) and 427 (Card
    # Draw) enter the FP dice pool at setup lines 45/48; 426 is SPENT at
    # T1 line 100 (Scribe drag pool->Used + Elves political 0->1 +
    # Controls.644 confirm).  The bid itself is chat-only — seed the
    # engine state with both tokens (attr names per
    # game_logic.has_action_token).
    'starting_resources': {'fp_political_token': True,
                           'fp_card_token': True},
    # T2 line 147: complexflips musters the WITCH-KING at Moria one
    # action BEFORE any FP nation is at war (Elves at box 2; his own
    # line-157 Dimrill Dale attack pushes them to At War at line 164,
    # 17 lines later).  USER RULING 2026-07-04: "Yes, they made a
    # mistake" — the engine's WK condition (Sauron at war + >=1 FP
    # nation At War) is CORRECT and KEPT; the early muster is a
    # documented human deviation (see data/rules_amendments.md).
    'minion_entry_waivers': {(147, 'witchking')},
}


# ===================================================================
# 2023-02-03_222800_FP_Wirbelwind_SP_GJB (Free Ring T5)
# ===================================================================


CURATION['2023-02-03_222800_FP_Wirbelwind_SP_GJB.log'] = {
}


# ===================================================================
# 2025-05-16_192114_FP_Adobim_SP_SlickRick441 (Free Ring T8)
# ===================================================================


CURATION['2025-05-16_192114_FP_Adobim_SP_SlickRick441.log'] = {
    # T4 line 291: SlickRick441 plays ORC PATROL while the Fellowship's
    # last known position (the pawn, hidden, progress 6) is RIVENDELL —
    # an FP settlement, so the printed requirement "the Fellowship is
    # not in a region containing a Free Peoples Settlement" FAILS by
    # the strict reading (the hidden Fellowship IS at the pawn's
    # region for all purposes).  Played anyway — either a loose common
    # interpretation ("it's really somewhere out in the wild by now")
    # or a misplay; FLAGGED for user batch review in
    # data/rules_amendments.md.  The log is ground truth.
    'play_requirement_waivers': {(291, 'orcpatrol')},
}


# ===================================================================
# 2023-08-13_201533_FP_HombreLobo_SP_GJB
# ===================================================================

CURATION['2023-08-13_201533_FP_HombreLobo_SP_GJB.log'] = {
    # T6 line 630: HombreLobo plays HOUSE OF THE STEWARDS while
    # Boromir stands SEPARATED at OSGILIATH — a FREE region belonging
    # to no nation (product FAQ), so the printed requirement "Boromir
    # is in a Gondor region" FAILS by the strict reading.  Same
    # Osgiliath-as-Gondor loose interpretation as Polijuna's Challenge
    # of the King (line 519 waiver) — or a misplay; FLAGGED for user
    # batch review in data/rules_amendments.md.  The log is ground
    # truth (the card's own two Strategy draws follow at L632-633,
    # confirming the effect resolved).
    'play_requirement_waivers': {(630, 'houseofthestewards')},
}


# ===================================================================
# 2023-11-09_131632_FP_Polijuna_SP_AngryBird (Free Military T8)
# ===================================================================

CURATION['2023-11-09_131632_FP_Polijuna_SP_AngryBird.log'] = {
    # T6 line 519: Polijuna plays CHALLENGE OF THE KING while Aragorn
    # stands at OSGILIATH — a FREE region, so the printed requirement
    # "Strider/Aragorn is with a Free Peoples Army in a Gondor or Rohan
    # region" FAILS by the strict reading (Osgiliath belongs to no
    # nation).  Played anyway — loose common interpretation (Osgiliath
    # as Gondor territory) or misplay; documented in
    # data/rules_amendments.md.  The log is ground truth.
    'play_requirement_waivers': {(519, 'challengeoftheking')},
}


# ===================================================================
# 2025-04-06_162825_FP_demanray_SP_Ilia Kungel (Shadow Military T26)
# ===================================================================

CURATION['2025-04-06_162825_FP_demanray_SP_Ilia Kungel.log'] = {
    # T4 line 201: Ilia_Kungel plays HORDE FROM THE EAST ("Play if the
    # Southrons & Easterlings are At War", card_list.csv verbatim)
    # while the S&E marker sits at Game.2001 — NOT at war (the marker
    # round-trip at lines 207-208 right after suggests they noticed;
    # S&E reach At War only at lines 955-962, T11, just before the
    # SECOND, legal Horde play at line 968).  Genuine misplay; the log
    # is ground truth.  Documented in data/rules_amendments.md.
    'play_requirement_waivers': {
        (201, 'hordefromtheeast'),
        # The three demanray dual-card plays ((2274,'theentsawakehuorns'),
        # (2318,'challengeoftheking'), (2339,'theentsawaketreebeard')) are
        # COMBAT-CARD commitments into the open Fords-of-Isen (T22) and
        # Orthanc-assault (T23) battles, revealed through the Scribe.141 +
        # Controls.45 play widget so the parser records them as
        # play_event; their event-side requirements are unmet in truth
        # (GtW at Umbar, all companions dead, Aragorn dead), which is
        # consistent with a combat-side play.  Reclassified by the general
        # effect-evidence classifier in build_schedules
        # (_reclassify_dual_cards; 1.217/1.218).  No per-game waiver.
    },
    # The reported "Shadow Military T26" is a CONCESSION (O.3): the
    # log's final turn is 23 (701 records) and the parser's own VP
    # tracking never exceeds SH 4 (Minas Tirith + Lorien; Erebor falls
    # and is retaken) -- nowhere near the 10 a military win needs.  At
    # the end the SH player presses the last siege ('press', rolls,
    # '3' hits) and FP resigns: DEMANRAY "gg" / "not funny" / "thats
    # not really fair hit hit hit hit hit)))" / "i did my best)";
    # Ilia_Kungel "I was afraid of lose)".  The 'T26' turn number is
    # report-metadata noise (same family as TBIS's misreported T25).
    'concession': 'Shadow',
}


# ===================================================================
# 2024-07-27_181335_FP_Columbo_SP_Adobim (Shadow Military T10)
# ===================================================================

CURATION['2024-07-27_181335_FP_Columbo_SP_Adobim.log'] = {
    # T5 line 451: Adobim plays MUSTERINGS OF LONG-PLANNED WAR ("Play
    # if all Shadow Nations are At War", card_list.csv verbatim) while
    # the S&E political marker sits at box 1 (Game.2001) — two boxes
    # short of At War per the marker walk ground truth.  Genuine
    # misplay, same family as demanray's Horde from the East.  The log
    # is ground truth.  Documented in data/rules_amendments.md (O.4).
    'play_requirement_waivers': {(451, 'musteringofalongplannedwar')},
}


# The reported "Free Ring T11" is a CONCESSION: the fellowship sits at
# Mordor step 4 (last track record L997, Game.532) when the Shadow
# player resigns — endgame chat L1199+ "its over" / "should have been
# last round xD" / "concede, you can say it was a ring win" / "ggwp".
# No step-5 record and no ring destruction exist in the log.  O.3
# ruling applies: certify on 0 mismatches + final board matching the
# replay at the concession point; outcome 'concession (reported
# winner: Free Peoples)'.
CURATION.setdefault(
    '2024-07-27_181335_FP_Columbo_SP_Adobim.log', {})[
    'concession'] = 'Free Peoples'

# T9 southern cascade (L939-966): West Harondor -> Pelargir ->
# Lamedon -> Dol Amroth; each hop's advance requires the logged
# defender wipe (Pelargir 2R at L939, Lamedon 1R at L957).  The
# engine's dice leave a survivor and the Lamedon -> Dol Amroth attack
# finds no army (dec199).  Ground-truth casualty specs from the log's
# owned marks, scoped to these battles only (Vayze precedent).
CURATION['2024-07-27_181335_FP_Columbo_SP_Adobim.log'][
    'casualty_spec_regions'] = {'pelargir', 'lamedon'}


# ===================================================================
# 2025-06-19_041657_FP_Stella Rossa_SP_GoSagan (CONCEDED, Shadow T7)
# ===================================================================

CURATION['2025-06-19_041657_FP_Stella Rossa_SP_GoSagan.log'] = {
    # The reported outcome ("Shadow Military T7") is a RESIGNATION: the
    # humans stopped mid-battle at T7 (Woodland Realm assault, cc's
    # still exchanged at L730; post-game chat "well, what could I
    # do????").  USER RULING (O.3, 2026-07-07): certify concession
    # games on 0 mismatches + final board state matching the replay at
    # the concession point; outcome recorded as 'concession (reported
    # winner: X)'.  The teacher/net stages STOP at the concession point
    # — no post-concession decision may become a training sample, and
    # NO trainable 'resign' decision exists.  (Diagnostic play-on:
    # CONCESSION_PLAY_ON=1.)  Corroboration: playing past the
    # concession ends in a Shadow Ring win T19 — same winner.
    'concession': 'Shadow',
}


# ===================================================================
# 2023-04-29_101111_FP_arlekj_SP_Fojtu (gate-1 sample)
# ===================================================================
CURATION['2023-04-29_101111_FP_arlekj_SP_Fojtu.log'] = {
    # T1 line 94: Fojtu plays OLOG-HAI (recruit one Sauron unit) with
    # Sauron NOT At War — only Isengard had walked to war (L80, marker
    # 263).  The at-war muster-event rule fails by the strict reading;
    # played anyway in VASSAL (no enforcement) — a logged human misplay;
    # the log is ground truth.  FLAGGED for user batch review in
    # data/rules_amendments.md (same class as the Orc Patrol waiver,
    # 2025-05-16 SlickRick441 L291).
    # T7 line 617: MORGUL WOUND ('the Fellowship is revealed') played
    # with the Fellowship hidden on the Mordor track (the T7 tile's
    # damage went to the Strider guide-sacrifice at L611; no reveal
    # record).  Same loose play as Ivan L701 — the humans commonly
    # ignore this card's revealed-condition.  FLAGGED.
    'play_requirement_waivers': {(94, 'ologhai'), (617, 'morgulwound')},
}


# ===================================================================
# 2024-06-21_002508_FP_jmelder95_SP_Adobim (gate-1 sample)
# ===================================================================
CURATION['2024-06-21_002508_FP_jmelder95_SP_Adobim.log'] = {
    # T6 line 456: Adobim plays THE DAY WITHOUT DAWN ('all Shadow
    # Nations are At War') with Southrons & Easterlings at political
    # box 1 (engine and parser agree — the resync found no lag).  A
    # logged human misplay; the log is ground truth.  FLAGGED for user
    # batch review in data/rules_amendments.md.
    'play_requirement_waivers': {(456, 'thedaywithoutdawn')},
}


# ===================================================================
# 2025-10-30_055513_FP_Ivan_SP_Stella Rossa (gate-1 sample)
# ===================================================================
CURATION['2025-10-30_055513_FP_Ivan_SP_Stella Rossa.log'] = {
    # T6 line 412: Ivan plays THE EAGLES ARE COMING with the only
    # companion-bearing FP army at Helm's Deep and the only Nazgul at
    # Minas Tirith — NOT adjacent, so the printed condition fails.
    # Truth then executes the effect anyway (L416: the MT Nazgul flees
    # to Minas Morgul, a Sauron stronghold, exactly the card's effect).
    # A logged loose play/misplay; the log is ground truth.  FLAGGED
    # for user batch review in data/rules_amendments.md.
    # T9 line 701: MORGUL WOUND ('the Fellowship is revealed') played
    # three lines after the Fellowship RE-HID (L698, after the Gimli
    # hunt sacrifice at L697).  Same loose play as arlekj L617.
    'play_requirement_waivers': {(412, 'theeaglesarecoming'),
                                 (701, 'morgulwound')},
}


# ===================================================================
# 2025-11-14_175207_FP_branstonpickle_SP_Compote_Andros (CONCEDED,
# Shadow T10)
# ===================================================================

# The reported "Shadow Military T10" is an AGREED result: the on-board
# Shadow VP marker ends at 9 (Game.391, L1137) and the endgame chat
# (L1160+) concedes the imminent Helm's Deep capture ("That I lack 1/2
# die to end that round" / "but that would actually stop the HD win" /
# "taking dale back wouldn't help here").  O.3 ruling applies: certify
# on 0 mismatches + final board matching the replay at the concession
# point; outcome 'concession (reported winner: Shadow)'.
CURATION.setdefault(
    '2025-11-14_175207_FP_branstonpickle_SP_Compote_Andros.log', {})[
    'concession'] = 'Shadow'

# T10 L1158: 'The Ents Awake: Entmoot' played with Gandalf the White at
# The Shire and NO Companion in Fangorn.  The card requires BOTH ('GtW
# in play AND a Companion in Fangorn' - DB text + almanac, QA
# 2026-07-11 cluster D fixed the engine's old OR reading).  The humans
# evidently used the OR reading; the log is ground truth for the
# replay - waived per-game and FLAGGED for user batch review in
# data/rules_amendments.md (1.126).
CURATION.setdefault(
    '2025-11-14_175207_FP_branstonpickle_SP_Compote_Andros.log', {})[
    'play_requirement_waivers'] = {(1158, 'theentsawakeentmoot')}

# ===================================================================
# 2025-05-18_170053_FP_PeterM_SP_jan (Free Ring T27) — PROBE
# ===================================================================

# T15 Pelargir is the game's pivot: the FP retake (a585-589, L1490-
# L1509 -- casualties both ways, Celeborn's Galadrim, SH retreat to
# West Harondor, FP advance 1R+5E from Lossarnach) must land, or a
# phantom Pelargir VP plus The Shire's fall (L1529) makes a phantom
# Shadow Military 10 at T15 (the humans see-sawed below 10 for twelve
# more turns and FP won the corruption race, Free Ring T27).  Handled
# by the 138-family span binding (default).


# 2025-05-18_170053_FP_PeterM_SP_jan T16 L1635 (WAVE 4C): PeterM reveals
# 'Paths of the Woses' through the CardBuffet widget INSIDE the open t16
# West Harondor battle = a COMBAT-CARD commitment of the card's combat
# side 'Sudden Strike'.  The EVENT side (move a Rohan army to Minas
# Tirith) never resolves in truth (no army_move to MT in the log/dumps).
# The general effect-evidence classifier (_reclassify_dual_cards in
# build_schedules) reclassifies the die-less play_event to a combat_card.


# ===================================================================
# 2024-02-05_013114_FP_Polijuna_SP_The board is set (Free Military T26)
# ===================================================================

# The human REPORT said "Free Military T25" — the LOG says T26 and the
# report is a misreport (curated, sharpz precedent): the T25 Umbar
# assault's VP entry was RETRACTED by both players (422 marker walked
# back 31->29 at L2269-2270), the raw turn marker Game.7=26 stands at
# L2275, and the decisive Gundabad battle + final VP move (L2326)
# happen inside turn 26 before 'gg'.  fidelity_campaign_games.txt
# updated 25 -> 26.  (Earlier "outcome-exact T25" runs were false
# positives: a teacher declaring victory at T25 can never have
# consumed the ~40 T26 records the log contains.)  No casualty-spec
# curation: the T25 Umbar garrison survivor was REAL — the humans'
# own correction proves the assault fell short.


# ===================================================================
# 2025-02-05_204637_FP_Vayze_SP_ProniX (Shadow Military T9)
# ===================================================================

# The T8/T9 endgame battles at Woodland Realm and Dale decide the 10th
# Shadow VP; the engine's casualty policies leave different survivors
# than the humans' hit allocations (v5 A/B proved forcing exactly these
# battles' logged outcomes flips the teacher to the EXACT human result,
# Shadow Military T9).  Ground-truth casualty specs from the log's
# owned marks, scoped to these regions only.
CURATION.setdefault(
    '2025-02-05_204637_FP_Vayze_SP_ProniX.log', {})[
    'casualty_spec_regions'] = {'woodland realm', 'dale'}


_EMPTY = {
    'parser_gap_suppress': set(),
    'card_choices': {},
    'casualty_policy_overrides': {},
    'curated_discards': (),
    'companion_queue_default': None,
    # O.3 (user ruling 2026-07-07): reported winner of a CONCEDED game
    # ('Free'/'Shadow') — the schedule simply ENDS at the concession
    # point; the closed loop stops there (no samples beyond it, no
    # 'resign' action).  None = game played to a real victory.
    'concession': None,
    # Regions whose battles' per-round casualty OUTCOMES are forced
    # from the log's owned casualty marks (the audited v5 machinery,
    # scoped per game per region — curations encode ground truth).
    # Used where a specific battle's engine casualty math diverges
    # from the humans' and decides the outcome (2025-02-05 Vayze T8/T9
    # Woodland Realm + Dale: survivor counts blocked the 10th SH VP).
    'casualty_spec_regions': set(),
    # (line, normalized_card_name) plays executed DESPITE a failed printed
    # requirement — logged human misplays; the log is ground truth
    'play_requirement_waivers': set(),
    # (line, normalized_minion_name) minion musters executed although the
    # engine's entry condition does NOT hold at that state — logged human
    # mistakes (user ruling 2026-07-04, see data/rules_amendments.md).
    # Closed loop force-applies the muster bookkeeping (ground truth);
    # diagnose counts the record as waived; engine legality is UNCHANGED.
    'minion_entry_waivers': set(),
    # (region, side_name, battle_seq|None) -> max applied hits for one
    # combat round — logged human deviations (e.g. a skipped Great Host
    # auto-hit); consumed via combat.FINAL_HITS_HOOK
    'combat_hit_caps': {},
    # Per-game bid-compensation counts (user ruling 2026-07-03): the
    # pre-game bid for Shadow grants the FP player tokens/dwarven rings.
    # The bid result is chat-only — set state attrs here when a game's
    # log shows token/ring usage, e.g. {'fp_dwarven_rings': 1,
    # 'fp_political_token': True, 'fp_card_token': True}.
    'starting_resources': {},
}


def get_curation(fname):
    """Curation entry for a log filename (FRESH empty defaults per
    call: sharing _EMPTY's mutable set()/dict instances would let one
    caller's mutation contaminate every later game's defaults)."""
    out = {k: (v.copy() if isinstance(v, (set, dict, list)) else v)
           for k, v in _EMPTY.items()}
    out.update(CURATION.get(os.path.basename(str(fname or '')), {}))
    return out
