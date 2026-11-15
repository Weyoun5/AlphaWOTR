AlphaWOTR - War of the Ring (2nd ed) vs an AI

WHAT THIS IS
  An AI opponent for War of the Ring (2nd edition), played through the
  WOTR-Gandalf Java client: you play your side on the real board, the
  AI plays its side on the same board, the client rolls every die and
  draws every card.  A Python rules engine is the sole authority on
  state and legality; neural nets plus search pick the AI's moves.

  Play_vs_AI.bat asks which net the AI plays:
    Free Peoples  FP-WIN (wins by any route) or FP-MILITARY (aims for a
                  military victory)
    Shadow        SH-WIN (wins by any route) or SH-CORRUPTION (aims to
                  corrupt the Ring-bearer)
  Each net learned from human game logs, then from 2 million self-play
  games; the two WIN nets were then trained further on their own search.

  Please report strange moves and bugs.

INSTALL
  Windows.  Java 8 or newer (17 or 21 recommended: https://adoptium.net).
    The launcher checks  java -version  and says so if yours is too old;
    an old Java first on PATH is overridden by setting JAVA_HOME.
  Release zip (AlphaWOTR.zip, GitHub Releases): unzip anywhere, done -
    it carries its own python\ runtime with every package.
  Git clone instead: Python 3.10+, then  pip install -r requirements.txt
    (set ALPHAWOTR_PYTHON=<python.exe> to force one).  The repository
    holds everything but the python\ runtime.
  Mac: see MAC below.  New and not yet played on a real Mac.

MAC
  Apple silicon (M1 or newer), macOS 14 or newer, Java 8 or newer.
  AlphaWOTR-mac.zip (GitHub Releases): unzip anywhere, done - it carries
    its own Python with every package, and the search libraries.
  Play_vs_AI.command   as Play_vs_AI.bat:  Play_vs_AI.command SA COR
    The first time macOS refuses to open it (a download it does not
    know): allow it under System Settings > Privacy & Security, "Open
    Anyway".  It then takes the download mark off its own folder.
  The turn search runs on the processor (the Mac's GPU is not used), so
  the slider starts at 1; it needs about 5 GB of free memory, and with
  less the game plays at position 1.
  python/bin/python3 wotr_bridge/selftest.py   checks that the libraries
  pick the same moves as the Python engine, and that the turn search
  answers.  If it fails, please send its output.
  Messages inside the client still say "Play_vs_AI.bat".

PLAY
  Play_vs_AI.bat          pick your side and the AI's net; the game client opens
  Play_vs_AI.bat SA COR   the same without questions: the AI's side (SA or FP),
                          then its net (WIN or MIL for FP, WIN or COR for SA);
                          AUTO MIL COR watches AI vs AI with those two nets
  In the client: New Game -> menu AI -> "Play vs AI..." -> Start Game.
  Each turn the AI draws your cards and rolls your action dice for you
  (through the client, under your name), and it ends the turn.  You roll
  your own combat and Hunt dice when the AI asks for them in chat.
  Three check boxes in the menu AI change what the AI does for you.
  You answer each of your decisions in one of three ways:
    on the board   move your pieces as in a human game: the die into
                   Current Action, then the units, the card, the marker.
                   Killing the die after a move ends the action at once.
                   Kill it first and move afterwards and the moves still
                   count: the action then ends after six quiet seconds.
    in the panel   click the option (every legal option is listed)
    in chat        one of the words below, for what pieces cannot show
                   ("help" lists them in the game)
  When the pieces fit no legal move, or more than one, the AI says so in
  one sentence and waits; "back" puts the board back.

AI STRENGTH (menu AI: the slider)
  1  tree search only (fastest)
  2  + turn search: the best candidates are played out to the end of the
     next turn, and a move that proves better replaces the tree search's
  3  + the side's other net suggests candidates too
  4  + twice the playouts (strongest)
  Default: 3 when the turn search runs on an NVIDIA GPU (see below),
  else 1.
  Your own choice is remembered.  A change counts from the AI's next
  decision.  Small decisions (discards, casualties, retreats, two
  choices) are always searched cheaply.
  Seconds per AI action on an 8-core PC, typical:
    with the GPU     2: 8    3: 10   4: 17
    processor only   2: 27           4: 50
  The zip ships the CPU build of torch, so out of the box the turn
  search runs on the processor.  With an NVIDIA GPU, run Enable_GPU.bat
  once (game closed): it downloads the GPU build of the same torch
  (about 2 GB, 6 GB of free disk) and tries it; if it does not work on
  your PC the CPU build is put back.  "Enable_GPU.bat disable" undoes
  it, "Enable_GPU.bat status" says what is installed.
  The turn search needs about 2 GB of free memory per worker and uses
  fewer workers, or switches itself off, when there is less.  Each
  decision has a time limit; a slow PC gets fewer playouts.  What it
  decided is in turn_search\turn_search_service.log.

CHAT (only what the board cannot show)
  help                   the AI lists these words
  back                   put the board back as the game has it
  undo                   take back your last decision (until the AI has
                         answered it, or dice or cards went into it)
  pass                   pass your action
  fp ok                  go on without declaring the Fellowship
  done                   end the steps of a card
  <target> from <source> attack, for example: dale from erebor.  When
                         dice of several kinds could make the attack,
                         put the one you mean into Current Action first.
  nc                     no combat card
  cc <card name>         that combat card (or pick it in the client)
  field                  fight in the field; do not retreat
  stand                  do not advance after a won battle
  press / continue       go on with the battle or the siege
  stop                   end the attack
  0e                     no dice into the Hunt Box (Shadow)
  <gate>                 morannon or minas morgul, when the AI asks which
  The old words skip, reset, rewind, yes and no are gone: the AI answers
  each with one sentence that says what to do instead.

REGIONS (type the whole name; case, accents and punctuation do not matter)
  andrast                 fords of bruinen         northern rhovanion
  anfalas                  fords of isen            nurn
  angmar                   forlindon                old ford
  arnor                    gap of rohan             old forest
  ash mountains            gladden fields           old forest road
  barad dur                goblin's gate            orthanc
  bree                     gorgoroth                osgiliath
  buckland                 grey havens              parth celebrant
  cardolan                 harlindon                pelargir
  carrock                  helm's deep              rhosgobel
  dagorlad                 high pass                rivendell
  dale                     hollin                   south anduin vale
  dead marshes             iron hills               south downs
  dimrill dale             khand                    south dunland
  dol amroth               lamedon                  south ered luin
  dol guldur               lorien                   south ithilien
  druadan forest           lossarnach               south rhun
  druwaith iaur            minas morgul             southern dorwinion
  eagles eyrie             minas tirith             southern mirkwood
  east harondor            minhiriath               southern rhovanion
  east rhun                morannon                 tharbad
  eastemnet                moria                    the shire
  eastern brown lands      mount gram               tower hills
  eastern emyn muil        mount gundabad           trollshaws
  eastern mirkwood         narrows of the forest    umbar
  edoras                   near harad               vale of the carnen
  enedwaith                noman lands              vale of the celduin
  erebor                   north anduin vale        weather hills
  erech                    north downs              west harondor
  ered luin                north dunland            westemnet
  ettenmoors               north ered luin          western brown lands
  evendim                  north ithilien           western emyn muil
  fangorn                  north rhun               western mirkwood
  far harad                northern dorwinion       withered heath
  folde                    northern mirkwood        woodland realm

PANEL (inside the client; menu AI -> Show AI panel)
  Decide, Battle, Hunt, Card   every legal option of your open decision
  Fix: Put the board back      moves every piece to where the game has it
                               (the same as typing "back")
  Fix: Take back my last decision   the same as typing "undo"
  Fix: Settings, AI strength   tree-search simulations per move (default
                               300), from the AI's next decision
  Skip decision, Reset and resync and Check the board are greyed with
  the reason: this version has no skip and no reset.  The other Settings
  (search width, chance outcomes, legality check, whose turn) change
  nothing and say so under "Last result".

HOW THE AI THINKS
  Tree search: every AI decision is a Monte-Carlo tree search of 300
  simulations in a C++ copy of the engine.  The WIN nets judge positions
  with a separately trained value net; FP-MILITARY and SH-CORRUPTION use
  their own.
  Turn search (slider 2-4): the tree search's move and its best rivals
  are played out through the next turn on the same dice, many times.  A
  rival replaces the move only if it wins a second comparison on fresh
  dice by a clear margin.  Each seat is played and judged by its own net.

KNOWN ISSUES
  A card you draw by hand before your turn starts can be put back and
    cost you the action.
  An answer that could still grow (more pieces to move) is carried out
    only after about six quiet seconds.
  Menu AI -> "Show legacy AI window" does nothing.
  The turn search's gain is measured for the WIN nets only.
  PCs without an NVIDIA GPU, or with little memory, are covered by
    fallbacks that were tested on one PC only.

ENGINE-ONLY BOARD (no Java)
  Play_Engine_Only.bat [fp|sh|watch]   the engine's own board
  UNTESTED: only the real-client path above has been played and
  certified.

PLAY THROUGH THE WotR BOT PROTOCOL (other bots, hosted matches)
  protocol/wotr_protocol_bot.py lets the AI play through the WotR Bot
  Protocol (draft 0.1, github.com/behindmedia/wotr-protocol): a
  REFEREE (any rules engine that speaks the protocol) runs the game,
  rolls every die and draws every card; at each decision it sends the
  AI its view of the position and the list of legal options, and the
  AI answers with one of them.  No Java client and no .bat file are
  involved.  Same engine as Play_vs_AI.bat; the nets are the starting
  nets (wotr_*_corpus_live.pt) with the tree search in Python, not the
  league nets and turn search of Play_vs_AI.bat.
  Git clone + requirements.txt as above (Python 3.10+).  From the
  folder of this README:

    python -I protocol/wotr_protocol_bot.py
        the protocol on stdin/stdout: a referee starts it as its child
        process (stderr is the log)
    python -I protocol/wotr_protocol_bot.py --connect wss://wotr.behindmedia.com/referee --token TOKEN
        dial a referee's WebSocket with a seat token (the hosted
        referee: a match and its two tokens are made on that site)

  Options: --sims N (MCTS sims, default 300 as in the panel; a referee
  may also set it as the protocol option "sims"), --max-seconds S
  (cap per decision; the referee's clock is respected anyway),
  --verbose (each decision on stderr), --log FILE (one JSON line per
  decision).

  How it works: at each decision the engine's state is rebuilt from
  the referee's view (the opponent's hidden cards are sampled by the
  search as usual), the engine's pending decision is set up to match
  the question (a battle round, a Hunt, a discard ...), and MCTS picks
  one of the engine's own actions.  That action is matched to the
  referee's option; actions the referee does not offer (the two
  engines differ in a few map borders and rules) are kept out of the
  search.  The referee asks in small steps (the die, then the Army,
  then its units); the engine's action answers all of them.  Where the
  engine has no matching decision a plain rule answers, and the log
  says so ("fallbacks" in the end-of-game line).

CREDITS / LICENSE
  AlphaWOTR code (engine, bridge, panel, nets): MIT - see LICENSE.
  War of the Ring is a board game by Ares Games (Roberto Di Meglio,
  Marco Maggi, Francesco Nepitello); this project is an unofficial fan
  work and uses no rights to it.
  The game client is WOTR-Gandalf (a third-party program, not ours):
  the AI talks to it through small add-on jars; the client itself is
  NOT covered by this project's license.
  The nets were trained on logs of games played by the WOTR-Gandalf
  community - thank you to every player whose games are in there.
