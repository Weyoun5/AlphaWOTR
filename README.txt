AlphaWOTR - War of the Ring (2nd edition) against an AI

  You play one side on the WOTR-Gandalf game client, the AI plays the
  other.  Please report strange moves and bugs.

INSTALL
  Java 8 or newer is needed (https://adoptium.net).
  Windows   AlphaWOTR.zip: unzip, run Play_vs_AI.bat.
            With an NVIDIA graphics card, run Enable_GPU.bat once: the
            stronger AI settings then run much faster.
  Mac       AlphaWOTR-mac.zip (Apple silicon, macOS 14 or newer): unzip,
            open Play_vs_AI.command.  The first time, macOS refuses to
            open it: allow it under System Settings > Privacy & Security,
            "Open Anyway".

PLAY
  The launcher asks your side and which AI you play against.
  In the client: New Game, then menu AI > "Play vs AI..." > Start Game.
  Play on the board as against a person, or click your choice in the AI
  panel on the right.  In chat, "help" lists the words the AI
  understands; "back" puts the board back, "undo" takes back your last
  decision, "pass" passes.  Attacks are typed: "osg from si nc"
  (nc = no combat card).  Abbreviations: ofr mt da pel mm gorg mg dg
  lor riv wlr osg hd si west fords; any other region by its whole name.
  At the top of the AI panel: AI strength, 1 (fastest) to 4 (strongest),
  and the simulations of its search, from none to 1000.
  Without questions:  Play_vs_AI.bat SA COR   (the AI's side, SA or FP,
  then its net: WIN or MIL for FP, WIN or COR for SA).
  Play_vs_AI.bat AUTO   watches AI vs AI.

OTHER
  Play_Engine_Only.bat   the engine's own board, no Java (untested).
  protocol/wotr_protocol_bot.py   the AI through the WotR Bot Protocol
  (github.com/behindmedia/wotr-protocol):
    python -I protocol/wotr_protocol_bot.py
    python -I protocol/wotr_protocol_bot.py --connect wss://wotr.behindmedia.com/referee --token TOKEN

CREDITS / LICENSE
  AlphaWOTR code: MIT - see LICENSE.
  War of the Ring is a board game by Ares Games (Roberto Di Meglio,
  Marco Maggi, Francesco Nepitello); this is an unofficial fan work.
  The game client is WOTR-Gandalf, a third-party program not covered by
  this license.  The nets were trained on games of the WOTR-Gandalf
  community - thank you to every player whose games are in there.
