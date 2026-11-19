# Our changes (werner77/AlphaWOTR, branch `ours`)

This fork carries AlphaWOTR (by Wey, MIT) for War of the Ring's bot protocol. **`ours` is the one branch every build
of ours follows** (since 2026-11-18): the bot-tournament image, the hosted protocol bots, the self-play bridge and the
browser port (`wasm/`). `main` stays their release, untouched (it mirrors `upstream/main`); the older branches
(`wotr-bot-protocol`, the head of PR #4 to Wey; `wasm-port`, the head of PR #5; `tournament-deadline`) stay as they
were and are all contained in `ours`.

Their v0.2.0 (`3dd6874`) plus, in order:

| Commit | Where | What |
|---|---|---|
| `041313c` | `protocol/` | The WotR Bot Protocol adapter on v0.2.0: the AI server's own strength-1 decision (league nets, native search, certain line, turn search) behind the protocol. |
| `9b1c304` | `protocol/` | The adapter runs with `PYTHONHASHSEED=0` as their launcher does (re-executes once when started with `-I`). |
| `f5cfc17` | native box (both) | The root free-force floor's "no child yet" held in a `uint64_t`: on 32-bit `size_t` (WebAssembly) `NO_ID` never matched. |
| `4fcc62d` | native box (both) | A pre-roll combat magnitude (Dread and Despair, Relentless Assault, Heroic Death) is searched natively as their Python engine offers it (was refused: "Unsupported pre-combat choice menu"). |
| `2cc5f6d` | `protocol/` | A battle's rearguard recorded as v0.2.0 records it (`attacker_rearguard`), not removed from the army ("Cannot form attacking army"). |
| `610380e` | `protocol/` | The seat's unit rule takes effect: an action `ai_units` named is matched by itself (was answered with the first matched option). |
| `1261011` | native box (both) | An army move's `unit_set` applied as Python's `_execute_action` applies it (was refused natively). |
| `20548e9` | native search, `wotr_bridge`, orchestrator | A deadline: the controller's stop flag (`wotr_search_controller_request_stop`, capability `stop`), checked between simulations (at least one runs); `native_search(deadline=)`, `OwnModelSearch.choose(deadline=)`; `certain_line.DEADLINE` bounds the certain line. Without a deadline nothing changes. |
| `b2445d1` | `protocol/` | A per-decision clock answered by a deadline (the search stops to answer `--margin-ms`, 200 ms, before the clock); the clock rule's budget (from 5 s a decision 80% of the clock less 0.3 s, under 5 s a quarter; the first search of a kind at most half the clock; a kind not timed yet a cautious 60 simulations a second, never a faster kind's rate); **the menu worked out once a decision** (`LeagueSearcher.choose(legal=)`: no second `get_legal_actions`). |
| `bb16d7f` … `46d2d23` | `wasm/` | The browser port (WebAssembly): their box compiled unchanged with patches on a copy, the network in C++, the search, the adapter and the root ported line for line; `wasm/README.md`. |
| (merge) | | `tournament-deadline` merged into `wasm-port`'s head: `ours`. |
| (this commit) | `wasm/` | The port's two speed fixes (2026-11-18, from profiling the tournament adapter): the search takes the decision's menu instead of working it out again (`awdec::Projection`), and the certain line is bounded by time as the Python is (`SECONDS` 15 s, the per-decision clock's deadline less 40 ms) plus the port's own cap of 1 s (`Bot::certain_seconds`); the port's clock rule (`Bot::budget`) as the adapter's `budget` of `b2445d1`. The C++ deadline stop compiles in the wasm build (single thread: never set there). |

The parity checks of each step (the same answers as their Python) are in `wasm/README.md` and in War of the Ring's
docs (docs/AlphaWOTR-Wasm-Plan.md, docs/Bot-Bridges.md).
