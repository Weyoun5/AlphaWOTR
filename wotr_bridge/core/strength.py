"""How hard the AI thinks at ONE decision: the strength slider of the client's AI menu, and the rule
for small decisions.

The slider (settings 'strength', 1..len(LEVELS)) chooses what the TURN SEARCH does on top of the
ordinary search (turn_search.py): LEVELS[position - 1] is the NAME of a setting of the turn-search
kit (its search_settings.json says what the name means), None = the ordinary search alone.
The names and their measured worth (the cluster's test of the settings: gain per decision in
points of win probability, the narrow setting before them +0.34): wide +0.63, wide_advisers +0.67,
wide_advisers_double +0.92.

A SMALL decision - a choice among SMALL_KINDS only (a discard, the eyes, declaring the Fellowship,
a retreat, the casualties) - is searched cheaply whatever the slider says: at most SMALL_SIMS
simulations, and the turn search (where the slider has it on at all) with SMALL_SETTING.  How MANY
choices a decision has does not make it small: a menu of few choices is searched full width by
the engine (ai._SMALL_MENU: the same simulations for every choice; the net's own move plays unless
another choice's mean value leads it clearly).

A slider nobody has moved (settings 'strength' 0) stands at THIS PC's default: default(gpu) -
DEFAULT_GPU when the turn search runs its nets on the GPU (turn_search.gpu), DEFAULT_CPU when it
does not (there the turn search takes several times as long).

plan() is the one place that says this; server.Server._choose asks it once per decision, so a
slider moved during a game counts from the AI's next decision on.  The client's labels for the
positions are AiSettings.LEVELS (client_build/hook_out): the same number of positions, the same
two defaults.
"""
LEVELS = (None, 'wide', 'wide_advisers', 'wide_advisers_double')
DEFAULT_GPU, DEFAULT_CPU = 3, 1
SMALL_SIMS = 100
SMALL_SETTING = 'wide'
SMALL_KINDS = frozenset((
    'discard_card', 'discard_table_card', 'eye_allocate', 'fellowship_declare', 'declare_decline',
    'combat_retreat', 'combat_no_retreat', 'retreat_into_siege',
    'casualty_allocate', 'casualty_preserve_units', 'casualty_preserve_elites'))


def small(menu):
    return all(a.action_type in SMALL_KINDS for a in menu)


def default(gpu):
    """The position of a slider nobody has moved."""
    return DEFAULT_GPU if gpu else DEFAULT_CPU


def plan(position, menu, sims):
    """-> (simulations of the ordinary search, the turn search's setting or None) for the decision
    among `menu`.  position: the slider's (anything out of range is the nearest end); sims: the
    simulations of a full decision (the panel's simulations knob, else the server's --sims; 0: none)."""
    if not sims:        # no search at all: the net's own choice, and no candidates for a turn search
        return 0, None
    setting = LEVELS[min(max(int(position), 1), len(LEVELS)) - 1]
    if small(menu):
        return min(sims, SMALL_SIMS), setting and SMALL_SETTING
    return sims, setting
