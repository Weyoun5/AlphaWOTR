"""wotr_bridge.live — the THIN sided-play driver (REBUILD_PLAN.md, 2026-08-19).

One state authority (the engine), one wire (the AiFeed rendered feed),
client-owned randomness, fail-closed blinding.  The old monolith
(ai_move_server.py) was ARCHIVED 2026-08-22 (lab retired_2026-08-22):
this package IS the one live head (launch_head.py generates
Play_vs_AI.bat).  Lifted code was COPIED with its provenance noted;
nothing here ever imported from the donor.

Modules:
  feed_reader    tail the AiFeed file (truncation/re-emit tolerant)
  live_parser    rendered sentence -> record (rendered_replay core,
                 live-prefix normalization, loud unknown-shape alarm)
  randomness     NeedClientCall funnel protocol
  gesture_match  human gesture window -> engine menu match (delta-based)
  driver         gesture windows, pending buffer, adoption, undo
  chat           the sided chat vocabulary (policy exit)
  emission       engine events -> emitter L1 lines (staged, whitelisted)
  emission_card  card-effect board rendering
  adopt          CLIENT -> ENGINE board adoption (census-gated)
  client_board   census: engine-vs-dump divergence detector
  client_view    the client piece/zone view model
  undo_model     client-native undo bookkeeping + restore watch
  segmenter      feed lines -> gesture windows
  sidecar        AiFeed meta sidecar decode
  spectator      both-seats-external driver (corpus lane)
  scenario       scripted-prefix harness lane
  transport      file bridge + liveness lock
  card_identity  commitment-decode card identities
  live_server    file-bridge transport entry point

Design rules 1-12 are in WOTR_AI_training_lab/REBUILD_PLAN.md; each was
paid for by a live defect.  Client facts 1-22 live there too — verify
against the real client before relying on one.
"""
