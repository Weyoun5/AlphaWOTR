"""What the person chose in the client's AI menu: the three check boxes (what the bridge does for
him) and the strength slider (how hard the AI thinks).

ONE source of truth: the file NAME in the client's folder.  The CLIENT writes it - whole, by a
rename - every time he ticks or unticks a box or moves the slider, and reads it when the menu is
built, so the menu shows these values at start-up and the choice is remembered between sessions.
The bridge only READS it, here and nowhere else: prefers[key] is the value as the file has it now
(read again when the file changed).  A missing or unreadable file, or a key it does not state
with a value of the default's own type, gives DEFAULT.

    'effects'   Automatically apply card effects for you
    'dice'      Automatically roll dice for me
    'turn'      Automatically roll action dice and draw cards for me
    'strength'  the slider's position, 1.. (what each position means: strength.py); the file states
                it only once he has moved the slider - 0 = not moved: the PC's default (strength.default)
    'sims'      the simulations knob: the simulations of the ordinary search, 0 = no search (the net's own
                choice); the file states it only once he has moved the knob - -1 = not moved: the server's --sims

WHEN a changed box counts is the asker's rule, not this file's: each thing a box governs asks once,
when it begins (randomness.ClientRandomness.directives_for: a request of the game; server.Server.
_decide: the effects of one card), and what has begun is finished as it began.
"""
import json
import os

NAME = 'play_settings.json'
DEFAULT = {'effects': True, 'dice': False, 'turn': True, 'strength': 0, 'sims': -1}


class Settings(object):
    def __init__(self, client_dir=None):
        self.path = os.path.join(client_dir, NAME) if client_dir else None      # (None: the defaults alone)
        self.stamp, self.values = None, dict(DEFAULT)

    def __getitem__(self, box):
        """Does the bridge do what this box governs for him?  (A box of None governs what is nobody's
        choice: the bridge does it.)"""
        if box is None:
            return True
        try:
            st = os.stat(self.path) if self.path else None
        except OSError:
            st = None
        stamp = (st.st_mtime_ns, st.st_size) if st is not None else None
        if stamp != self.stamp:
            try:        # (held or half replaced for the moment: the values read last stand, the next ask reads again)
                told = {}
                if stamp is not None:
                    with open(self.path, encoding='utf-8') as f:
                        told = json.load(f)
                self.values = {k: told[k] if type(told.get(k)) is type(v) else v for k, v in DEFAULT.items()}
                self.stamp = stamp
            except (OSError, ValueError, AttributeError):
                pass
        return self.values[box]
