"""core - the small live-play bridge (bridge2/DESIGN.md).

The engine is the only game state.  Per client poll the server ingests the
feed's events (dice, card and tile draws, chat words), reads the board dump
tied to that feed position, advances the engine while it does not need a
human, renders each committed step, and makes the client board equal the
engine (reconcile).

  server       the loop (one handler per poll), journal, status
  layout       the client's table as data: families of pieces, places that are one place
  boardmodel, match, menu   a person's answer read from the board (menu.PICKS: his picks)
  reconcile    engine -> client board: difference rows -> piece moves (THE comparison)
  transport, client_protocol, server_ownership   the file channel
  feed_reader, live_parser        feed tail + line grammar (events only)
  client_view  the client model the planners read
  randomness   dice / draws through the client (ask, feed, retry; who makes each: client or person)
  settings     what the person chose in the AI menu (three boxes, the strength slider; one file)
  strength     how hard the AI thinks at one decision (the slider's positions; small decisions cheaply)
  emission, emission_card, emitter   engine actions as client lines
  own_model, native_search, inference_cache, policy_thread_scope   AI decision
  turn_search  the rest-of-turn search on top of that decision (its own process; off unless turn_search.json says on)
  card_identity   drawn-card identity from the client's commitments
  words        the sentences the bridge says

Run: python -u -m core.server --client-dir <dir> --side AUTO  (cwd = wotr_bridge).
The engine (src/) is imported as bare modules; this file puts it on sys.path.
"""
import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
