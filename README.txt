AlphaWOTR - War of the Ring (2nd ed) vs an AI

WHAT THIS IS
  An AI opponent for War of the Ring (2nd edition), played through the
  WOTR-Gandalf Java client: you play your side on the real board, the
  AI plays its side on the same board, the client rolls every die and
  draws every card.  A Python rules engine is the sole authority on
  state and legality; two neural nets (one per side) plus Monte-Carlo
  tree search pick the AI's moves.
  Goal: a strong, rules-exact opponent for human play and, from there,
  a self-improving one.
  Release v0.1.6: v52 rules engine, native search, and self-play-trained
  models (Free Peoples gen5, Shadow league gen24).
  Model encoding: 4647 observation features and 157 action features.
  See RELEASE_MANIFEST.tsv for the packaged file hashes.

Please report strange moves or any bugs, I expect there are quite a few

INSTALL
  Java 8 or newer (17 or 21 recommended: https://adoptium.net).  The
    launcher checks  java -version  and says so if yours is too old; an
    old Java first on PATH is overridden by setting JAVA_HOME.
  Release zip (AlphaWOTR.zip, GitHub Releases): unzip anywhere,
    done - it carries its own python\ runtime with every package.
  Git clone instead: Python 3.10+, then  pip install -r requirements.txt
    (CPU torch is enough; set ALPHAWOTR_PYTHON=<python.exe> to force one).

PLAY
  Play_vs_AI.bat          pick your side, the game client opens
  In the client: New Game -> menu AI -> "Play vs AI..." -> Start Game.
  Play your side on the board as in a human game; after each action
  kill the spent die in Current Action.  The AI plays its side on the
  board.  The client rolls all dice and draws all cards.

CHAT (only what the board cannot show)
  help                   list every command the AI understands, and
                         what it is currently waiting on
  <region>               declare an attack on that region
  from <region>          answer the AI's "from?" when it asks
  nc                     no combat card (or choose one with the client)
  press / stop           continue / end a battle (attacker)
  siege / field battle   defend inside / outside a stronghold
  fp ok                  end your Fellowship phase (Free Peoples)
  0e / Ne                eyes into the Hunt Box (Shadow)
  pass                   pass your action
  done                   finished moving (nazgul/companion card chains)
  rolling for hunt       Nazgul Strike: take the hunt roll
  discarding card <name> Nazgul Strike: discard that table card
  none                   decline an either/or card choice
  resync                 reconcile the board (never triggers the AI)

REGIONS (type them as below; case, accents and punctuation do not matter)
  Short forms: any unique start of a name works ("osg" = osgiliath,
  "tirith" = minas tirith, "n ithilien" = north ithilien, "shire",
  "helms deep").  If it fits several regions the AI lists them
  Common aliases include dg, hd, mt, da, mg, mm, lor, wlr, ofr,
  pel, shire, riv, ereb and gh. "fords" means Fords of Isen.
  andrast                  fords of bruinen         northern rhovanion
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

PANEL
  RESET AI               anything stuck: clears decisions, re-reads the
                         board, the AI continues
  Enforce legal moves    beta; off = the AI takes your moves as given
  Sliders                see HOW THE AI THINKS below

HOW THE AI THINKS (panel sliders)
  Every AI decision is a Monte-Carlo tree search: "MCTS sims" (default
  300) is how many simulated lines it plays out before choosing - more
  = stronger and slower (roughly linear).  "Children / node" (default
  40) is how many candidate moves a search node expands, picked by the
  net's prior; "chance outcomes / node" (default 24) is how many
  distinct dice / tile results a random event keeps in the tree (combat
  rounds use exact probability kernels, so these are the most likely
  outcomes by mass).  Right end of a slider = no cap.  All three apply
  from the next AI decision.

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
  involved; the same engine, nets and search as Play_vs_AI.bat.
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

  Test (plain Python, a few seconds): python -I protocol/test_corsairs.py
  (Corsairs of Umbar's battle, which a referee reports as fought from
  Umbar, set up as the engine's co-located landing battle).

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
