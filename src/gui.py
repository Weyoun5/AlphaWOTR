# gui.py - Interactive Tkinter GUI for War of the Ring
"""Fully playable GUI for War of the Ring."""

from __future__ import annotations
from typing import Dict, List, Optional, TYPE_CHECKING
import os

# Set matplotlib backend BEFORE any imports to prevent Tkinter conflicts
import matplotlib
matplotlib.use('Agg')

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

if TYPE_CHECKING:
    from game_state import GameState
    from board import Board

from game_state import (
    Side, Nation, UnitKind, ActionDieFace, Nazgul, Companion,
    ActionDie
)
from board import sanitize
import board as board_module

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UNITS_DIR = os.path.join(BASE_DIR, "units")
DICE_DIR = os.path.join(BASE_DIR, "dice")
POLIT_DIR = os.path.join(BASE_DIR, "political_track")

# ---------------------------------------------------------------------------
# Colors - Dark theme with good contrast
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#1a1a2e",
    "panel_bg": "#16213e",
    "text": "#eaeaea",
    "text_dim": "#888888",
    "fp_primary": "#4a90d9",
    "fp_secondary": "#2d5a8a",
    "shadow_primary": "#e94560",
    "shadow_secondary": "#8a2d3d",
    "highlight": "#ffd700",
    "button_bg": "#0f3460",
    "button_fg": "#ffffff",
    "button_active": "#1a5490",
    "success": "#2ecc71",
    "warning": "#f39c12",
    "error": "#e74c3c",
}

# ---------------------------------------------------------------------------
# Unit icon mapping (from board_visual.py)
# ---------------------------------------------------------------------------
UNIT_ICON = {
    (Nation.DWARVES, UnitKind.ELITE): "DwarvenElite.png",
    (Nation.DWARVES, UnitKind.REGULAR): "DwarvenRegular.png",
    (Nation.ELVES, UnitKind.ELITE): "ElvenElite.png",
    (Nation.ELVES, UnitKind.REGULAR): "ElvenRegular.png",
    (Nation.GONDOR, UnitKind.ELITE): "GondorElite.png",
    (Nation.GONDOR, UnitKind.REGULAR): "GondorRegular.png",
    (Nation.THE_NORTH, UnitKind.ELITE): "NorthElite.png",
    (Nation.THE_NORTH, UnitKind.REGULAR): "NorthRegular.png",
    (Nation.ROHAN, UnitKind.ELITE): "RohanElite.png",
    (Nation.ROHAN, UnitKind.REGULAR): "RohanRegular.png",
    (Nation.ISENGARD, UnitKind.ELITE): "IsengardElite.png",
    (Nation.ISENGARD, UnitKind.REGULAR): "IsengardRegular.png",
    (Nation.SAURON, UnitKind.ELITE): "SauronElite.png",
    (Nation.SAURON, UnitKind.REGULAR): "SauronRegular.png",
    (Nation.SOUTHRON_EASTERLINGS, UnitKind.ELITE): "SouthronElite.png",
    (Nation.SOUTHRON_EASTERLINGS, UnitKind.REGULAR): "SouthronRegular.png",
}

LEADER_ICON = {
    Nation.DWARVES: "DwarvenLeader.png",
    Nation.ELVES: "ElvenLeader.png",
    Nation.GONDOR: "GondorLeader.png",
    Nation.THE_NORTH: "NorthLeader.png",
    Nation.ROHAN: "RohanLeader.png",
    Nation.ISENGARD: "Saruman.png",
    Nation.SAURON: "Nazgul.png",
    Nation.SOUTHRON_EASTERLINGS: "SouthronRegular.png",
}

POLIT_ICON = {
    Nation.ELVES: "elves",
    Nation.DWARVES: "dwarves",
    Nation.GONDOR: "gondor",
    Nation.ROHAN: "rohan",
    Nation.THE_NORTH: "north",
    Nation.ISENGARD: "isengard",
    Nation.SAURON: "sauron",
    Nation.SOUTHRON_EASTERLINGS: "southron",
}

# Mordor track step names (official board labels)
MORDOR_TRACK_NAMES = [
    "Cirith Ungol / The Black Gate",
    "Morgai / Udun",
    "Carach Angren",
    "Gorgoroth Plain",
    "On the Slopes of Mount Doom",
    "The Cracks of Doom",
]

# Side classification helpers for mixed siege armies
SHADOW_NATIONS_GUI = {Nation.ISENGARD, Nation.SAURON,
                      Nation.SOUTHRON_EASTERLINGS}
# Both spellings occur: game_logic muster code uses 'WitchKing'/'Mouth',
# _add_minion_to_region uses the ShadowMinion display names.
SHADOW_MINION_NAMES = {'Witch King', 'WitchKing', 'Saruman',
                       'Mouth of Sauron', 'Mouth'}

# Special-character map markers: initials + leadership values
COMPANION_INITIALS = {
    'Gandalf the White': 'GW', 'Gandalf the Grey': 'GG',
    'Strider': 'ST', 'Aragorn - Heir to Isildur': 'AR', 'Aragorn': 'AR',
    'Legolas': 'LE', 'Gimli': 'GI', 'Boromir': 'BO',
    'Meriadoc': 'ME', 'Peregrin': 'PE', 'Gollum': 'GO',
}
MINION_INITIALS = {
    'Witch King': 'WK', 'WitchKing': 'WK', 'Saruman': 'SA',
    'Mouth of Sauron': 'MO', 'Mouth': 'MO',
}
MINION_LEADERSHIP = {
    'Witch King': 2, 'WitchKing': 2, 'Saruman': 1,
    'Mouth of Sauron': 2, 'Mouth': 2,
}


def _norm_icon_key(s: str) -> str:
    """Case/space/punctuation-insensitive key for icon-file matching."""
    return ''.join(c for c in str(s).lower() if c.isalpha())


def _build_character_icon_map() -> dict:
    """Map companion/minion names -> PNG filenames actually present in
    src/units/ (matched case/space-insensitively). Characters without a
    file fall back to the initials markers."""
    files = {}
    try:
        for fn in os.listdir(UNITS_DIR):
            if fn.lower().endswith('.png'):
                files.setdefault(_norm_icon_key(fn[:-4]), fn)
    except OSError:
        return {}
    wanted = {
        'Gandalf the Grey': ['gandalfthegrey', 'gandalf'],
        'Gandalf the White': ['gandalfthewhite'],
        'Strider': ['strider'],
        'Aragorn': ['aragorn'],
        'Aragorn - Heir to Isildur': ['aragorn'],
        'Legolas': ['legolas'],
        'Gimli': ['gimli'],
        'Boromir': ['boromir'],
        'Meriadoc': ['meriadoc'],
        'Peregrin': ['peregrin'],
        'Gollum': ['gollum', 'smeagol'],
        'Witch King': ['witchking'],
        'WitchKing': ['witchking'],
        'Saruman': ['saruman'],
        'Mouth of Sauron': ['mouthofsauron', 'mouth'],
        'Mouth': ['mouthofsauron', 'mouth'],
    }
    out = {}
    for name, keys in wanted.items():
        for k in keys:
            if k in files:
                out[name] = files[k]
                break
    return out

# Die-face -> short letter (hunt-box glyphs)
FACE_LETTER = {
    ActionDieFace.CHARACTER: 'C', ActionDieFace.ARMY: 'A',
    ActionDieFace.MUSTER: 'M', ActionDieFace.MUSTER_ARMY: 'H',
    ActionDieFace.EVENT: 'P', ActionDieFace.EYE: 'E',
    ActionDieFace.WILL_OF_WEST: 'W',
}

# Dice face to image mapping
DICE_IMAGES = {
    # Free Peoples
    (Side.FREE_PEOPLES, ActionDieFace.CHARACTER): "ADFPcharacter.png",
    (Side.FREE_PEOPLES, ActionDieFace.ARMY): "ADFParmymuster.png",
    (Side.FREE_PEOPLES, ActionDieFace.MUSTER): "ADFPmuster.png",
    (Side.FREE_PEOPLES, ActionDieFace.MUSTER_ARMY): "ADFParmymuster.png",
    (Side.FREE_PEOPLES, ActionDieFace.EVENT): "ADFPevent.png",
    (Side.FREE_PEOPLES, ActionDieFace.WILL_OF_WEST): "ADFPwill.png",
    # Shadow
    (Side.SHADOW, ActionDieFace.CHARACTER): "ADSAcharacter.png",
    (Side.SHADOW, ActionDieFace.ARMY): "ADSAarmy.png",
    (Side.SHADOW, ActionDieFace.MUSTER): "ADSAmuster.png",
    (Side.SHADOW, ActionDieFace.MUSTER_ARMY): "ADSAarmymuster.png",
    (Side.SHADOW, ActionDieFace.EVENT): "ADSAevent.png",
    (Side.SHADOW, ActionDieFace.EYE): "ADSAeye.png",
}


def normalize_region(name: str) -> str:
    """Normalize region name for consistent lookup."""
    # Remove special chars, lowercase, strip
    return name.lower().replace("'", "").replace("-", " ").replace("û", "u").replace("ó", "o").replace("í", "i").strip()


class WotrGUI:
    """Main GUI class for War of the Ring."""

    def __init__(self, state: "GameState", board: "Board"):
        self.state = state
        self.board = board
        self.root = tk.Tk()
        self.root.title("War of the Ring")
        self.root.configure(bg=COLORS["bg"])
        # R3-LEGACY-GUI-TURN (R2-RS-P1-09 FU2): an exception escaping any
        # Tk callback (button binding, after() timer, key binding) is
        # named in the GUI log, not only printed to stderr
        self.root.report_callback_exception = self._report_callback_exception

        # State tracking
        self.current_side = Side.FREE_PEOPLES
        self.selected_region: Optional[str] = None
        self.selected_die_idx: Optional[int] = None
        self.pending_action: Optional[str] = None  # "move", "attack", etc.
        self.action_source: Optional[str] = None  # Source region for move/attack
        self.zoom_level = 0.55

        # AI integration
        self.ai_side: Optional[Side] = None    # Which side is AI-controlled (None = no AI)
        self.ai_net = None                      # Loaded WOTRValueNet
        self.ai_playing = False                 # Guard against re-entry during AI turn

        # Self-play (watch mode)
        self.selfplay_running = False
        self.selfplay_active_side = None  # Who acts next
        # C02 (round 18): the side whose die action is being resolved
        # (ai._turn_owner_after); None before the turn's first action
        self.selfplay_turn_owner = None
        self.selfplay_turn = 0
        self.selfplay_speed = 800  # ms between actions
        self.selfplay_paused = False

        # Trace replay mode (step through a recorded self-play game)
        self.replay_steps = None
        self.replay_meta = {}
        self.replay_idx = 0

        # Read-only viewer label (P1-04): live_view's ENGINE MIRROR sets
        # it; while set, manual state-editing controls are refused (see
        # _manual_controls_refusal).
        self.read_only_view: Optional[str] = None

        # Card decks live on self.state (fp_char_deck etc.); no GUI mirrors.
        self._init_card_decks()

        # Player hands - actual card IDs
        self.hands: Dict[Side, List[str]] = {
            Side.FREE_PEOPLES: [],
            Side.SHADOW: [],
        }

        # Image caches
        self._unit_images: Dict[str, ImageTk.PhotoImage] = {}
        self._dice_images: Dict[str, ImageTk.PhotoImage] = {}
        self._polit_images: Dict[str, ImageTk.PhotoImage] = {}
        self._board_image: Optional[ImageTk.PhotoImage] = None

        # Build the GUI
        self._setup_layout()
        self._load_images()
        self.refresh()
        # Players start with 0 cards and draw during play; no setup draw.

    def _init_card_decks(self):
        """Initialize card decks with actual card IDs."""
        from game_logic import initialize_decks

        # Call game_logic to handle deck initialization (shuffling happens there)
        initialize_decks(self.state)

    def _setup_layout(self):
        """Create the main layout."""
        # Main container
        self.main_frame = tk.Frame(self.root, bg=COLORS["bg"])
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Left: Siege status panel
        self.siege_panel = tk.Frame(self.main_frame, bg=COLORS["panel_bg"], width=170)
        self.siege_panel.pack(side=tk.LEFT, fill=tk.Y, padx=2)
        self.siege_panel.pack_propagate(False)

        siege_title = tk.Label(self.siege_panel, text="Strongholds",
                               font=("Arial", 10, "bold"),
                               bg=COLORS["panel_bg"], fg=COLORS["text"])
        siege_title.pack(pady=(5, 2))

        # Dynamic siege boxes: one per besieged stronghold, showing the
        # GARRISON (defender units). The map region then renders only the
        # besieger units. Rebuilt on every refresh (works in live play and
        # trace replay alike).
        self.siege_boxes_frame = tk.Frame(self.siege_panel,
                                          bg=COLORS["panel_bg"])
        self.siege_boxes_frame.pack(fill=tk.X, padx=3, pady=(0, 4))

        _ALL_STRONGHOLDS = [
            ("--- Free Peoples ---", None),
            ("Rivendell", "rivendell"), ("Lorien", "lorien"),
            ("Grey Havens", "grey havens"), ("Woodland Realm", "woodland realm"),
            ("Edoras", "edoras"), ("Helm's Deep", "helm s deep"),
            ("Minas Tirith", "minas tirith"), ("Dol Amroth", "dol amroth"),
            ("Erebor", "erebor"), ("Dale", "dale"),
            ("--- Shadow ---", None),
            ("Barad-dur", "barad dur"), ("Dol Guldur", "dol guldur"),
            ("Minas Morgul", "minas morgul"), ("Morannon", "morannon"),
            ("Mount Gundabad", "mount gundabad"), ("Orthanc", "orthanc"),
        ]
        self.siege_labels = {}  # sanitized_key -> (label_widget, orig_name)
        for name, key in _ALL_STRONGHOLDS:
            if key is None:
                # Section header
                tk.Label(self.siege_panel, text=name,
                         font=("Arial", 8, "italic"),
                         bg=COLORS["panel_bg"], fg=COLORS["text_dim"]
                ).pack(fill=tk.X, padx=3, pady=(4, 0))
                continue
            lbl = tk.Label(self.siege_panel, text=f"  {name}",
                           font=("Arial", 8),
                           bg=COLORS["panel_bg"], fg=COLORS["text_dim"],
                           anchor=tk.W, justify=tk.LEFT)
            lbl.pack(fill=tk.X, padx=3, pady=0)
            self.siege_labels[key] = (lbl, name)

        # Center: Board canvas
        self.canvas_frame = tk.Frame(self.main_frame, bg=COLORS["bg"])
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Canvas with scrollbars
        self.canvas = tk.Canvas(
            self.canvas_frame,
            bg="#2a2a3e",
            highlightthickness=0
        )
        self.h_scroll = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.v_scroll = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)

        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # Right: Side panel with scrollbar
        self.side_panel_outer = tk.Frame(self.main_frame, bg=COLORS["panel_bg"], width=400)
        self.side_panel_outer.pack(side=tk.RIGHT, fill=tk.Y)
        self.side_panel_outer.pack_propagate(False)

        # Create canvas for scrolling
        self.side_canvas = tk.Canvas(self.side_panel_outer, bg=COLORS["panel_bg"], highlightthickness=0)
        self.side_scrollbar = tk.Scrollbar(self.side_panel_outer, orient=tk.VERTICAL, command=self.side_canvas.yview)
        self.side_canvas.configure(yscrollcommand=self.side_scrollbar.set)

        self.side_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.side_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create frame inside canvas
        self.side_panel = tk.Frame(self.side_canvas, bg=COLORS["panel_bg"])
        self.side_canvas_window = self.side_canvas.create_window((0, 0), window=self.side_panel, anchor=tk.NW)

        # Bind resize events
        self.side_panel.bind("<Configure>", self._on_side_panel_configure)
        self.side_canvas.bind("<Configure>", self._on_side_canvas_configure)

        # Enable mousewheel scrolling
        self.side_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self._build_side_panel()

    def _on_side_panel_configure(self, event):
        """Update scroll region when side panel changes size."""
        self.side_canvas.configure(scrollregion=self.side_canvas.bbox("all"))

    def _on_side_canvas_configure(self, event):
        """Update window width when canvas changes size."""
        self.side_canvas.itemconfig(self.side_canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling for side panel."""
        self.side_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _build_side_panel(self):
        """Build the side panel with all controls."""
        # --- Header ---
        header = tk.Frame(self.side_panel, bg=COLORS["panel_bg"])
        header.pack(fill=tk.X, padx=10, pady=10)

        self.turn_label = tk.Label(
            header, text="Turn 1", font=("Arial", 14, "bold"),
            bg=COLORS["panel_bg"], fg=COLORS["text"]
        )
        self.turn_label.pack(side=tk.LEFT)

        self.side_label = tk.Label(
            header, text="Free Peoples", font=("Arial", 12),
            bg=COLORS["panel_bg"], fg=COLORS["fp_primary"]
        )
        self.side_label.pack(side=tk.RIGHT)

        # --- Zoom controls ---
        zoom_frame = tk.Frame(self.side_panel, bg=COLORS["panel_bg"])
        zoom_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(zoom_frame, text="Zoom:", bg=COLORS["panel_bg"], fg=COLORS["text"]).pack(side=tk.LEFT)
        self._make_button(zoom_frame, "-", self._zoom_out, width=3).pack(side=tk.LEFT, padx=2)
        self.zoom_label = tk.Label(zoom_frame, text="55%", bg=COLORS["panel_bg"], fg=COLORS["text"], width=5)
        self.zoom_label.pack(side=tk.LEFT)
        self._make_button(zoom_frame, "+", self._zoom_in, width=3).pack(side=tk.LEFT, padx=2)
        self._make_button(zoom_frame, "Fit", self._zoom_fit, width=4).pack(side=tk.LEFT, padx=2)

        # --- VP Display ---
        vp_frame = tk.Frame(self.side_panel, bg=COLORS["panel_bg"])
        vp_frame.pack(fill=tk.X, padx=10, pady=5)

        self.fp_vp_label = tk.Label(vp_frame, text="FP VP: 0", bg=COLORS["panel_bg"], fg=COLORS["fp_primary"])
        self.fp_vp_label.pack(side=tk.LEFT, padx=10)
        self.sh_vp_label = tk.Label(vp_frame, text="Shadow VP: 0", bg=COLORS["panel_bg"], fg=COLORS["shadow_primary"])
        self.sh_vp_label.pack(side=tk.RIGHT, padx=10)

        # --- Log (at top for watch mode visibility) ---
        log_frame = tk.LabelFrame(self.side_panel, text="Log", bg=COLORS["panel_bg"], fg=COLORS["text"])
        log_frame.pack(fill=tk.X, padx=10, pady=5)

        self.log_text = tk.Text(
            log_frame, bg=COLORS["bg"], fg=COLORS["text_dim"],
            font=("Consolas", 9), height=12, state=tk.DISABLED
        )
        self.log_text.pack(fill=tk.X, padx=5, pady=5)

        # Color tags for log
        self.log_text.tag_configure("combat", foreground="#ffd700")
        self.log_text.tag_configure("casualty", foreground="#e74c3c")
        self.log_text.tag_configure("turn", foreground="#3498db")
        self.log_text.tag_configure("die", foreground="#95a5a6")

        # --- Fellowship Info ---
        fel_frame = tk.LabelFrame(self.side_panel, text="Fellowship", bg=COLORS["panel_bg"], fg=COLORS["text"])
        fel_frame.pack(fill=tk.X, padx=10, pady=5)

        self.fel_info = tk.Label(fel_frame, text="", bg=COLORS["panel_bg"], fg=COLORS["text"], justify=tk.LEFT)
        self.fel_info.pack(anchor=tk.W, padx=5, pady=2)

        # Companions list
        self.companions_frame = tk.Frame(fel_frame, bg=COLORS["panel_bg"])
        self.companions_frame.pack(fill=tk.X, padx=5, pady=2)

        tk.Label(self.companions_frame, text="Companions:", bg=COLORS["panel_bg"],
                fg=COLORS["text"], font=("Arial", 9, "bold")).pack(anchor=tk.W)

        self.companions_listbox = tk.Listbox(
            self.companions_frame, bg=COLORS["bg"], fg=COLORS["text"],
            selectbackground=COLORS["highlight"], selectforeground="black",
            font=("Arial", 9), height=4, selectmode=tk.MULTIPLE
        )
        self.companions_listbox.pack(fill=tk.X, pady=2)

        # Companion buttons
        comp_btn_frame = tk.Frame(self.companions_frame, bg=COLORS["panel_bg"])
        comp_btn_frame.pack(fill=tk.X, pady=2)

        self._make_button(comp_btn_frame, "Separate", self._on_separate_companions,
                         width=10).pack(side=tk.LEFT, padx=2)
        self._make_button(comp_btn_frame, "Move on Board", self._on_move_companions,
                         width=12).pack(side=tk.LEFT, padx=2)

        # --- Hunt Box Display ---
        hunt_frame = tk.LabelFrame(self.side_panel, text="Hunt Box", bg=COLORS["panel_bg"], fg=COLORS["text"])
        hunt_frame.pack(fill=tk.X, padx=10, pady=5)

        self.hunt_box_label = tk.Label(hunt_frame, text="Eyes: 0 | Character dice: 0",
                                       bg=COLORS["panel_bg"], fg=COLORS["shadow_primary"])
        self.hunt_box_label.pack(anchor=tk.W, padx=5, pady=2)

        self.hunt_pool_label = tk.Label(hunt_frame, text="Hunt Pool: 16 tiles remaining",
                                        bg=COLORS["panel_bg"], fg=COLORS["text_dim"])
        self.hunt_pool_label.pack(anchor=tk.W, padx=5, pady=2)

        # --- Casualty & Reinforcement Pools (USER directive 2026-07-20:
        # per-nation deployed / casualties / computed reinforcements so
        # piece conservation is visible in any trace step) ---
        pools_frame = tk.LabelFrame(self.side_panel, text="Pools (deployed | casualties | reinforcements)",
                                    bg=COLORS["panel_bg"], fg=COLORS["text"])
        pools_frame.pack(fill=tk.X, padx=10, pady=5)
        self.pools_info = tk.Label(pools_frame, text="", justify=tk.LEFT,
                                   font=("Consolas", 8),
                                   bg=COLORS["panel_bg"], fg=COLORS["text_dim"])
        self.pools_info.pack(anchor=tk.W, padx=5, pady=2)

        # --- Elven Rings & Cards In Play ---
        rings_frame = tk.LabelFrame(self.side_panel, text="Elven Rings / In Play",
                                    bg=COLORS["panel_bg"], fg=COLORS["text"])
        rings_frame.pack(fill=tk.X, padx=10, pady=5)

        rings_row = tk.Frame(rings_frame, bg=COLORS["panel_bg"])
        rings_row.pack(fill=tk.X, padx=5, pady=2)

        self.fp_rings_label = tk.Label(
            rings_row, text="FP Rings: 3", bg=COLORS["panel_bg"],
            fg=COLORS["fp_primary"], font=("Arial", 9, "bold"),
            bd=1, relief=tk.RIDGE, padx=6, pady=2)
        self.fp_rings_label.pack(side=tk.LEFT, padx=(0, 8))

        self.sh_rings_label = tk.Label(
            rings_row, text="SH Rings: 0", bg=COLORS["panel_bg"],
            fg=COLORS["shadow_primary"], font=("Arial", 9, "bold"),
            bd=1, relief=tk.RIDGE, padx=6, pady=2)
        self.sh_rings_label.pack(side=tk.LEFT)

        self.inplay_label = tk.Label(
            rings_frame, text="In Play: —", bg=COLORS["panel_bg"],
            fg=COLORS["text"], justify=tk.LEFT, anchor=tk.W,
            wraplength=350, font=("Arial", 9))
        self.inplay_label.pack(fill=tk.X, padx=5, pady=(2, 4))

        # --- Special Characters Display ---
        special_frame = tk.LabelFrame(self.side_panel, text="Special Characters",
                                      bg=COLORS["panel_bg"], fg=COLORS["text"])
        special_frame.pack(fill=tk.X, padx=10, pady=5)

        # FP special characters
        self.fp_special_label = tk.Label(special_frame, text="FP: —",
                                         bg=COLORS["panel_bg"], fg=COLORS["fp_primary"],
                                         font=("Arial", 9))
        self.fp_special_label.pack(anchor=tk.W, padx=5)

        # Shadow special characters
        self.sh_special_label = tk.Label(special_frame, text="Shadow: —",
                                         bg=COLORS["panel_bg"], fg=COLORS["shadow_primary"],
                                         font=("Arial", 9))
        self.sh_special_label.pack(anchor=tk.W, padx=5)

        # Special action buttons
        special_btn_frame = tk.Frame(special_frame, bg=COLORS["panel_bg"])
        special_btn_frame.pack(fill=tk.X, padx=5, pady=5)

        self._make_button(special_btn_frame, "Muster Minion", self._on_muster_minion,
                         width=12).pack(side=tk.LEFT, padx=2)
        self._make_button(special_btn_frame, "Gandalf White", self._on_transform_gandalf,
                         width=12).pack(side=tk.LEFT, padx=2)
        self._make_button(special_btn_frame, "Crown Aragorn", self._on_crown_aragorn,
                         width=12).pack(side=tk.LEFT, padx=2)

        # --- Political Track ---
        polit_frame = tk.LabelFrame(self.side_panel, text="Political Track", bg=COLORS["panel_bg"], fg=COLORS["text"])
        polit_frame.pack(fill=tk.X, padx=10, pady=5)

        self.polit_canvas = tk.Canvas(polit_frame, bg=COLORS["panel_bg"], height=220, highlightthickness=0)
        self.polit_canvas.pack(fill=tk.X, padx=5, pady=5)

        # --- Dice Display ---
        dice_frame = tk.LabelFrame(self.side_panel, text="Action Dice", bg=COLORS["panel_bg"], fg=COLORS["text"])
        dice_frame.pack(fill=tk.X, padx=10, pady=5)

        # FP dice row
        fp_dice_label = tk.Label(dice_frame, text="Free Peoples:", bg=COLORS["panel_bg"], fg=COLORS["fp_primary"])
        fp_dice_label.pack(anchor=tk.W, padx=5)
        self.fp_dice_frame = tk.Frame(dice_frame, bg=COLORS["panel_bg"])
        self.fp_dice_frame.pack(fill=tk.X, padx=5, pady=2)

        # Shadow dice rows (wraps)
        sh_dice_label = tk.Label(dice_frame, text="Shadow:", bg=COLORS["panel_bg"], fg=COLORS["shadow_primary"])
        sh_dice_label.pack(anchor=tk.W, padx=5)
        self.sh_dice_frame = tk.Frame(dice_frame, bg=COLORS["panel_bg"])
        self.sh_dice_frame.pack(fill=tk.X, padx=5, pady=2)

        # Dice control buttons
        dice_btn_frame = tk.Frame(dice_frame, bg=COLORS["panel_bg"])
        dice_btn_frame.pack(pady=5)

        self._make_button(dice_btn_frame, "ROLL DICE", self._on_roll_dice,
                         bg=COLORS["success"], width=12).pack(side=tk.LEFT, padx=2)
        self._make_button(dice_btn_frame, "Alloc Eyes", self._on_allocate_eyes,
                         width=10).pack(side=tk.LEFT, padx=2)

        dice_btn_frame2 = tk.Frame(dice_frame, bg=COLORS["panel_bg"])
        dice_btn_frame2.pack(pady=2)

        self._make_button(dice_btn_frame2, "Use Ring", self._on_use_ring,
                         bg=COLORS["fp_secondary"], width=10).pack(side=tk.LEFT, padx=2)
        self._make_button(dice_btn_frame2, "Recover", self._on_recover_dice,
                         width=10).pack(side=tk.LEFT, padx=2)
        # R4-GUI-SKIP-DIE-BUTTON (USER A2-06a/b, ruling 1.276): the
        # engine's die-discard rows; End Turn's rule-1.67 refusal names it
        self._make_button(dice_btn_frame2, "Skip Die", self._on_skip_die,
                         width=10).pack(side=tk.LEFT, padx=2)

        # --- Action Buttons ---
        action_frame = tk.LabelFrame(self.side_panel, text="Actions", bg=COLORS["panel_bg"], fg=COLORS["text"])
        action_frame.pack(fill=tk.X, padx=10, pady=5)

        # Grid of action buttons
        btn_grid = tk.Frame(action_frame, bg=COLORS["panel_bg"])
        btn_grid.pack(padx=5, pady=5)

        actions = [
            ("Move Army", self._on_move_army),
            ("Attack", self._on_attack),
            ("Muster", self._on_muster),
            ("Political", self._on_political),
            ("Play Card", self._on_play_card),
            ("FSP Move", self._on_fsp_move),
            ("FSP Hide", self._on_fsp_hide),
            ("FSP Declare", self._on_fsp_declare),
            ("Move Nazgûl", self._on_move_nazgul),
            ("Change Guide", self._on_change_guide),
        ]

        for i, (text, cmd) in enumerate(actions):
            row, col = divmod(i, 2)
            btn = self._make_button(btn_grid, text, cmd, width=12)
            btn.grid(row=row, column=col, padx=3, pady=3)

        # Card buttons
        card_frame = tk.Frame(action_frame, bg=COLORS["panel_bg"])
        card_frame.pack(fill=tk.X, padx=5, pady=5)

        self._make_button(card_frame, "Draw Cards", self._on_draw_cards, width=12).pack(side=tk.LEFT, padx=3)
        self._make_button(card_frame, "Discard", self._on_discard_card, width=12).pack(side=tk.LEFT, padx=3)

        # End turn / Switch side
        ctrl_frame = tk.Frame(action_frame, bg=COLORS["panel_bg"])
        ctrl_frame.pack(fill=tk.X, padx=5, pady=5)

        self._make_button(ctrl_frame, "End Turn", self._on_end_turn,
                         bg=COLORS["warning"], width=12).pack(side=tk.LEFT, padx=3)
        self.switch_btn = self._make_button(ctrl_frame, "Switch Side", self._on_switch_side, width=12)
        self.switch_btn.pack(side=tk.LEFT, padx=3)
        self.pass_btn = self._make_button(ctrl_frame, "Pass", self._on_pass_turn, width=12)
        self.pass_btn.pack(side=tk.LEFT, padx=3)

        # --- Selected Region ---
        sel_frame = tk.LabelFrame(self.side_panel, text="Selected Region", bg=COLORS["panel_bg"], fg=COLORS["text"])
        sel_frame.pack(fill=tk.X, padx=10, pady=5)

        self.sel_region_label = tk.Label(sel_frame, text="None", bg=COLORS["panel_bg"],
                                         fg=COLORS["highlight"], font=("Arial", 11, "bold"))
        self.sel_region_label.pack(anchor=tk.W, padx=5)
        self.sel_army_label = tk.Label(sel_frame, text="", bg=COLORS["panel_bg"],
                                       fg=COLORS["text"], justify=tk.LEFT, wraplength=350)
        self.sel_army_label.pack(anchor=tk.W, padx=5, pady=5)

        # --- Hand Display ---
        hand_frame = tk.LabelFrame(self.side_panel, text="Hand", bg=COLORS["panel_bg"], fg=COLORS["text"])
        hand_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.hand_listbox = tk.Listbox(
            hand_frame, bg=COLORS["bg"], fg=COLORS["text"],
            selectbackground=COLORS["highlight"], selectforeground="black",
            font=("Arial", 10), height=6
        )
        self.hand_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.hand_listbox.bind("<<ListboxSelect>>", self._on_card_select)

        # Card details
        self.card_detail = tk.Text(
            hand_frame, bg=COLORS["bg"], fg=COLORS["text"],
            font=("Arial", 9), height=4, wrap=tk.WORD, state=tk.DISABLED
        )
        self.card_detail.pack(fill=tk.X, padx=5, pady=5)

        # Status bar
        self.status_label = tk.Label(
            self.side_panel, text="Click a region to select",
            bg=COLORS["panel_bg"], fg=COLORS["text_dim"]
        )
        self.status_label.pack(fill=tk.X, padx=10, pady=5)

    def _make_button(self, parent, text, command, width=10, bg=None, style=None):
        """Create a styled button using Label (works reliably on macOS)."""
        # Determine colors
        if bg == COLORS["success"]:
            bg_color = "#2ecc71"
            hover_color = "#27ae60"
        elif bg == COLORS["warning"]:
            bg_color = "#e67e22"
            hover_color = "#d35400"
        else:
            bg_color = "#0f3460"
            hover_color = "#1a5490"

        fg_color = "white"

        # Create label that looks like a button
        btn = tk.Label(
            parent, text=text,
            bg=bg_color, fg=fg_color,
            font=("Arial", 10, "bold"),
            padx=10, pady=5,
            relief=tk.RAISED, bd=2,
            cursor="hand2"
        )

        # Bind click and hover events.  Read-only guard: manual
        # state-editing controls are refused in LIVE mode, trace replay,
        # the ENGINE MIRROR and watch mode (_manual_controls_refusal);
        # zoom controls stay usable (view-only).
        def _dispatch(_e=None, _cmd=command, _txt=text):
            if (_txt not in self._VIEW_ONLY_CONTROLS
                    and not self._replay_view_control(_txt)
                    and not self._watch_view_control(_txt)):
                why = self._manual_controls_refusal()
                if why is not None:
                    self.log("REFUSED: " + why)
                    return
            _cmd()
        btn.bind("<Button-1>", _dispatch)
        btn.bind("<Enter>", lambda e: btn.configure(bg=hover_color))
        btn.bind("<Leave>", lambda e: btn.configure(bg=bg_color))

        return btn

    # Controls that only change the view; usable in every mode.
    _VIEW_ONLY_CONTROLS = ('-', '+', 'Fit')

    # R2-S4-ENGINE-AI-TOOLS (rescan r022, P1-04 regression): view controls
    # usable in TRACE REPLAY only.  A recorded trace hides nothing and
    # _replay_goto fills self.hands for both sides; 'Switch Side' only
    # flips current_side (the Hand listbox's source) and mutates no
    # snapshot.  Still refused in LIVE mode and the ENGINE MIRROR, where
    # the other side's hand is hidden information (1.261).
    _REPLAY_VIEW_CONTROLS = ('Switch Side',)

    def _replay_view_control(self, text) -> bool:
        return (text in self._REPLAY_VIEW_CONTROLS
                and getattr(self, 'replay_steps', None) is not None
                and getattr(self, 'live_session', None) is None
                and not getattr(self, 'read_only_view', None))

    # R2-RS-v068-P1-04: view controls usable in WATCH MODE (AI vs AI).
    # Both seats are AI, so no hand is hidden; 'Switch Side' only flips
    # current_side (the Hand listbox's source) and mutates nothing in
    # self.state - the watch loop never reads current_side.  Never lifts
    # the LIVE / ENGINE MIRROR refusals (hidden information, 1.261); trace
    # replay has its own allowlist above.
    _WATCH_VIEW_CONTROLS = ('Switch Side',)

    def _watch_view_control(self, text) -> bool:
        return (text in self._WATCH_VIEW_CONTROLS
                and getattr(self, 'selfplay_running', False)
                and getattr(self, 'live_session', None) is None
                and getattr(self, 'replay_steps', None) is None
                and not getattr(self, 'read_only_view', None))

    def _manual_controls_refusal(self) -> Optional[str]:
        """Why manual state-editing controls are refused right now, or
        None when they may run (the hot-seat / legacy AI GUI).

        P1-04: refusing only in LIVE mode left two read-only viewers
        open.  The ENGINE MIRROR renders the FULL engine state, so 'Draw
        Cards' popped the AI's deck, logged its draws and filled the Hand
        listbox with the AI's real hand (1.261).  Trace replay swaps in
        step['state'] without a copy, so every button rewrote the stored
        snapshot.

        R2-RS-v068-P1-04: watch mode (AI vs AI) set none of those flags,
        only selfplay_running, so every button ran on the game the watch
        loop drives ('End Turn' -> next_turn rolled the turn over an open
        engine pending and both sides drew).  The watch loop opens no
        _make_button dialog and no required _show_choice_dialog, so this
        blocks nothing the loop needs; it clears with selfplay_running
        (game end or halt) and 'Switch Side' stays usable
        (_watch_view_control)."""
        if getattr(self, 'live_session', None) is not None:
            return ("manual controls are disabled in LIVE mode - use the "
                    "decision panel (engine-legal actions only)")
        if getattr(self, 'replay_steps', None) is not None:
            return ("manual controls are disabled in trace replay - the "
                    "recorded snapshots are read-only")
        view = getattr(self, 'read_only_view', None)
        if view:
            return ("manual controls are disabled in the %s - it is a "
                    "read-only view" % view)
        if getattr(self, 'selfplay_running', False):
            return ("manual controls are disabled in watch mode - the "
                    "AI-vs-AI loop drives the game")
        return None

    def _load_images(self):
        """Pre-load commonly used images."""
        # Load board background
        try:
            img = Image.open(self.board.background_path)
            w = int(img.width * self.zoom_level)
            h = int(img.height * self.zoom_level)
            img = img.resize((w, h), Image.Resampling.LANCZOS)
            self._board_image = ImageTk.PhotoImage(img)
            # Stamp the zoom key so _draw_board's per-zoom cache reuses this
            # copy instead of re-resizing the full PNG on the first draw.
            self._board_image_zoom = round(self.zoom_level, 4)
        except Exception as e:
            print(f"Error loading board image: {e}")

        # Load dice images
        for (side, face), filename in DICE_IMAGES.items():
            path = os.path.join(DICE_DIR, filename)
            if os.path.exists(path):
                try:
                    img = Image.open(path).resize((36, 36), Image.Resampling.LANCZOS)
                    self._dice_images[(side, face)] = ImageTk.PhotoImage(img)
                except:
                    pass

        # Load FP action token images
        IMAGES_DIR = os.path.join(BASE_DIR, "images")
        for token_file in ["freeaction1.png", "freeaction2.png"]:
            path = os.path.join(IMAGES_DIR, token_file)
            if os.path.exists(path):
                try:
                    img = Image.open(path).resize((36, 36), Image.Resampling.LANCZOS)
                    self._dice_images[token_file] = ImageTk.PhotoImage(img)
                except:
                    pass

        # Load political track images
        for nation, stub in POLIT_ICON.items():
            for prefix in ["n", "p"]:  # active, passive
                fname = f"{prefix}{stub}.png"
                path = os.path.join(POLIT_DIR, fname)
                if os.path.exists(path):
                    try:
                        img = Image.open(path).resize((28, 28), Image.Resampling.LANCZOS)
                        self._polit_images[fname] = ImageTk.PhotoImage(img)
                    except:
                        pass

    def _load_unit_image(self, filename: str, size: int = 24,
                         directory: str = None) -> Optional[ImageTk.PhotoImage]:
        """Load a counter image, caching the PhotoImage.

        ASPECT RATIO IS PRESERVED (no squash): the image is scaled so its
        LARGER dimension equals *size*. This mirrors the reference Java
        app (decompiled-src h_0.java paint()): counters are drawn with
        their natural width/height (drawImage centred at the anchor via
        -getWidth()/2, -getHeight()/2) and only uniformly scaled by the
        zoom factor — never stretched to a square."""
        directory = directory or UNITS_DIR
        cache_key = f"{directory}|{filename}_{size}"
        if cache_key not in self._unit_images:
            path = os.path.join(directory, filename)
            if not os.path.exists(path):
                return None
            try:
                img = Image.open(path)
                w, h = img.size
                scale = size / float(max(1, max(w, h)))
                nw = max(1, int(round(w * scale)))
                nh = max(1, int(round(h * scale)))
                img = img.resize((nw, nh), Image.Resampling.LANCZOS)
                self._unit_images[cache_key] = ImageTk.PhotoImage(img)
            except Exception:
                return None
        return self._unit_images.get(cache_key)

    def refresh(self):
        """Refresh the entire display."""
        self._draw_board()
        # P1-15: re-read the 'Selected Region' inspector from the snapshot
        # _draw_board just recorded (_display_state).  Replay steps
        # (_replay_goto) and live engine decisions (_live_refresh) swap or
        # mutate self.state and only call refresh(); without this the panel
        # kept the forces of the step at which the region was clicked.
        if self.selected_region:
            self._update_region_info(self.selected_region)
        self._draw_political_track()
        self._update_dice_display()
        self._update_hand_display()
        self._update_info_labels()
        self._update_companions_display()
        self._update_hunt_box_display()
        self._update_rings_display()
        self._update_special_characters_display()
        self._update_siege_display()
        self._check_victory_conditions()
        self._draw_mirror_marks()

    def _draw_mirror_marks(self):
        """ENGINE-MIRROR divergence paint (USER 2026-08-16: 'open the
        engine GUI as a mirror ... so I can directly see whether there
        are desyncs').  live_view sets self.mirror_marks to a list of
        region keys where the client board and the engine view
        disagree; each gets a red ring.  Empty/absent = no paint."""
        marks = getattr(self, 'mirror_marks', None) or []
        for key in marks:
            region = self._find_region(key)
            if not region:
                continue
            cx, cy = region.screen_xy
            cx = int(cx * self.zoom_level)
            cy = int(cy * self.zoom_level)
            r = int(46 * self.zoom_level)
            self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                outline='#ff3030', width=4, tags=('mirror_mark',))

    def _draw_board(self):
        """Draw the board with all units."""
        self.canvas.delete("all")

        # Remember exactly which state snapshot this render shows — the
        # region-click inspector must read THIS state, not a stale one
        # (replay mode swaps self.state per step).
        self._render_state = self.state

        # Reload board image at current zoom (cached per zoom level --
        # re-opening + LANCZOS-resizing the full PNG every refresh is
        # far too slow for replay stepping)
        try:
            zkey = round(self.zoom_level, 4)
            if (self._board_image is None
                    or getattr(self, '_board_image_zoom', None) != zkey):
                img = Image.open(self.board.background_path)
                w = int(img.width * self.zoom_level)
                h = int(img.height * self.zoom_level)
                img = img.resize((w, h), Image.Resampling.LANCZOS)
                self._board_image = ImageTk.PhotoImage(img)
                self._board_image_zoom = zkey
            w = self._board_image.width()
            h = self._board_image.height()
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self._board_image, tags="background")
            self.canvas.configure(scrollregion=(0, 0, w, h))
        except Exception as e:
            print(f"Error drawing board: {e}")
            return

        # Control tokens on captured EMPTY settlements (under the figures)
        self._draw_control_tokens()

        # Draw armies with actual unit icons
        self._draw_armies()

        # Draw fellowship
        self._draw_fellowship()

        # Draw the Mordor track strip (right edge of the board)
        self._draw_mordor_track()

        # Draw the hunt box (below the Mordor strip)
        self._draw_hunt_box()

        # Draw selection highlight
        if self.selected_region:
            self._draw_selection()

    def _draw_armies(self):
        """Draw compact army markers: at most one figure per present unit
        type per side plus a R/E/L count label (regulars / elites / total
        leadership) and distinct markers for special characters."""
        # +60% larger icons (31 -> 50 base, min 21 -> 34)
        icon_size = max(34, int(50 * self.zoom_level))
        besieged = {sanitize(str(k)) for k in
                    getattr(self.state, 'besieged_strongholds', set())}

        for region_key, army in self.state.armies.items():
            if not army.units and not army.leaders:
                continue

            # Find region coordinates
            region = self._find_region(region_key)
            if not region:
                continue

            # Besieged stronghold: the garrison renders in its siege box
            # (left panel); the map region shows only the BESIEGERS.
            units_to_draw = army.units
            leaders_to_draw = army.leaders
            san_key = sanitize(str(region_key))
            if san_key in besieged:
                def_side = self._stronghold_defender_side(san_key)
                units_to_draw = [u for u in army.units
                                 if self._unit_side(u) != def_side]
                leaders_to_draw = [l for l in army.leaders
                                   if self._leader_side(l) != def_side]
                if not units_to_draw and not leaders_to_draw:
                    continue

            cx, cy = region.screen_xy
            cx = int(cx * self.zoom_level)
            cy = int(cy * self.zoom_level)

            # Radius budget: every marker of this region must keep its
            # centre within 0.6 x the distance from this anchor to the
            # nearest OTHER region anchor, so a stack can never look as
            # if it belongs to a neighbouring region.
            max_r = self._region_radius_budget(region.key)

            # One sub-row per side present; the group of rows is centred
            # VERTICALLY on the anchor so the whole region renders as
            # ONE tight cluster (d.java n() centres the stack offsets
            # on the region point).
            present = []
            for side in (Side.FREE_PEOPLES, Side.SHADOW):
                s_units = [u for u in units_to_draw
                           if self._unit_side(u) == side]
                s_leaders = [l for l in leaders_to_draw
                             if self._leader_side(l) == side]
                if s_units or s_leaders:
                    present.append((side, s_units, s_leaders))
            row_pitch = int(icon_size * 0.9)
            for i, (side, s_units, s_leaders) in enumerate(present):
                base_dy = (i - (len(present) - 1) / 2.0) * row_pitch
                self._draw_side_stack(
                    region_key, region.key, side, s_units, s_leaders,
                    cx, cy, base_dy, icon_size, max_r)

    def _region_radius_budget(self, anchor_key: str) -> float:
        """Maximum distance (canvas px) any marker centre of a region
        may sit from its anchor: 0.6 x the distance to the nearest
        other region anchor. Nearest-anchor distances (board coords)
        are computed once and cached."""
        m = getattr(self, '_nearest_anchor_dist', None)
        if m is None:
            pts = {r.key: r.screen_xy for r in self.board.regions}
            m = {}
            for k, (x, y) in pts.items():
                m[k] = min(
                    ((x - x2) ** 2 + (y - y2) ** 2) ** 0.5
                    for k2, (x2, y2) in pts.items() if k2 != k)
            self._nearest_anchor_dist = m
        d = m.get(anchor_key)
        if d is None:
            d = min(m.values()) if m else 100.0
        return 0.6 * d * self.zoom_level

    # -- Special-character / leadership helpers -------------------------

    def _display_state(self):
        """The state snapshot the renderer last drew. In replay mode the
        click handler must inspect exactly what is on screen, never a
        stale self.state."""
        return getattr(self, '_render_state', None) or self.state

    def _leadership_value(self, leader) -> int:
        """Leadership of one figure: plain Leader=1, Nazgul=1, Witch
        King=2, Mouth of Sauron=2, Saruman=1, Companion=.leadership."""
        # The figure's own .leadership attribute is authoritative
        # (game_state defines minion leadership); the name table is only
        # a fallback for attribute-less figures.
        if hasattr(leader, 'leadership'):
            try:
                return int(leader.leadership or 0)
            except (TypeError, ValueError):
                return 1
        name = getattr(leader, 'name', None)
        if name in MINION_LEADERSHIP:
            return MINION_LEADERSHIP[name]
        return 1  # plain Leader / Nazgul

    def _saruman_on_board(self) -> bool:
        """Mirror combat.setup_combat's Saruman-in-play test exactly:
        a Minion figure named Saruman in ANY army's leaders."""
        for a in self._display_state().armies.values():
            for l in a.leaders:
                if (type(l).__name__ == 'Minion'
                        and 'saruman' in str(getattr(l, 'name', '')).lower()):
                    return True
        return False

    def _saruman_elite_bonus(self, units, side: Side) -> int:
        """Saruman's ability (combat.py setup_combat): while Saruman is
        in play, EVERY Isengard Elite in a Shadow army counts as a Leader
        (leadership 1 each) — applies to all Isengard elites on the
        board, not only the army Saruman is with."""
        if side != Side.SHADOW or not self._saruman_on_board():
            return 0
        return sum(1 for u in units
                   if u.nation == Nation.ISENGARD
                   and u.kind == UnitKind.ELITE)

    def _leader_display_name(self, leader) -> str:
        """Proper display name for one leader figure: Companions and
        Minions by NAME (Saruman / Witch King / Mouth of Sauron), plain
        leaders by nation."""
        name = getattr(leader, 'name', None)
        if isinstance(leader, Companion):
            return str(name) if name else "Companion"
        if name in SHADOW_MINION_NAMES:
            canon = {'WitchKing': 'Witch King', 'Mouth': 'Mouth of Sauron'}
            return canon.get(name, str(name))
        nat = getattr(leader, 'nation', None)
        nat_txt = getattr(nat, 'value', None) or "?"
        return f"Leader ({nat_txt})"

    def _draw_control_tokens(self):
        """Control markers on captured EMPTY settlements (issue #7):
        a City/Stronghold captured by a side (in
        state.enemy_controlled[side]) that contains NO units of either
        side shows that side's control token — ShadowControl.png for
        Shadow-captured, FreeControl.png for FP-captured."""
        st = self._display_state()
        ec = getattr(st, 'enemy_controlled', {}) or {}
        # USER (2026-07-08): 30% smaller tokens (38 -> 27, min 22 -> 15)
        size = max(15, int(27 * self.zoom_level))
        img_dir = os.path.join(BASE_DIR, "images")
        for side, fname in ((Side.SHADOW, "ShadowControl.png"),
                            (Side.FREE_PEOPLES, "FreeControl.png")):
            for rkey in (ec.get(side) or set()):
                san = sanitize(str(rkey))
                army = self._find_army_sanitized(san)
                if army is not None and army.units:
                    continue  # units present (either side) — no token
                region = self._find_region(san)
                if region is None:
                    continue
                cx, cy = region.screen_xy
                cx = int(cx * self.zoom_level)
                cy = int(cy * self.zoom_level)
                tok_tags = ("control_token",
                            "rg_" + str(region.key).replace(" ", "_"))
                img = self._load_unit_image(fname, size, directory=img_dir)
                if img:
                    self.canvas.create_image(
                        cx, cy, image=img, anchor=tk.CENTER,
                        tags=tok_tags)
                else:  # fallback disc in the capturing side's colour
                    col = (COLORS["shadow_primary"] if side == Side.SHADOW
                           else COLORS["fp_primary"])
                    r = size // 2
                    self.canvas.create_oval(
                        cx - r, cy - r, cx + r, cy + r, fill=col,
                        outline="#ffffff", width=2, tags=tok_tags)

    def _character_image(self, name, size):
        """Portrait PNG for a companion/minion (or None -> initials
        fallback). PhotoImage objects are CACHED on the GUI object via
        _load_unit_image — Tkinter garbage-collects images referenced only
        by the canvas and they silently vanish otherwise."""
        if not name:
            return None
        m = getattr(self, '_char_icon_map', None)
        if m is None:
            m = _build_character_icon_map()
            self._char_icon_map = m
        fn = m.get(str(name)) or m.get(
            next((k for k in m if _norm_icon_key(k) ==
                  _norm_icon_key(name)), ''))
        if not fn:
            return None
        return self._load_unit_image(fn, size)

    def _special_initials(self, leader) -> Optional[str]:
        """Short initials for a special character, None for generic
        leaders/nazgul."""
        name = getattr(leader, 'name', None)
        if not name:
            return None
        if isinstance(leader, Companion):
            return COMPANION_INITIALS.get(name, name[:2].upper())
        if name in MINION_INITIALS:
            return MINION_INITIALS[name]
        return None

    def _draw_side_stack(self, region_key, anchor_key, side, units, leaders,
                         cx, cy, base_dy, icon_size, max_r):
        """One side's forces as ONE tight cluster CENTRED on the region
        anchor: an overlapping figure row (reg/elite/leader/nazgul),
        the R/E/L count label adjacent, and character portraits in
        row(s) directly beneath, overlapping Java-style.

        Rendering approach follows the reference Java app
        ("WOTR-GANDALF" decompiled-src): d.java n() builds a per-stack
        offset table CENTRED on the region point -- 1 figure at (0,0),
        2 at (-15,0)/(15,0), 3 at (-28,0)/(0,0)/(28,0) with ~40px
        counters, i.e. neighbouring figures advance only ~0.7 of a
        counter width; CHARACTER counters join the SAME cluster at
        (14,20)/(5,20), half a counter below and overlapping the figure
        row. h_0.java paint() draws every image centred on
        anchor+offset (drawImage at -w/2,-h/2) and draws the count
        string two-pass (black at +1/+1 under white/red, lines
        108-132). Here all offsets are additionally shrunk uniformly
        (scale-to-fit) so no marker centre ever leaves the region's
        radius budget: big stacks overlap TIGHTER instead of spreading
        toward a neighbouring region."""
        n_reg = sum(1 for u in units if u.kind == UnitKind.REGULAR)
        n_eli = sum(1 for u in units if u.kind == UnitKind.ELITE)
        leadership = sum(self._leadership_value(l) for l in leaders)
        # Saruman's ability: Isengard Elites count as leadership 1 each
        # while Saruman is in play (mirrors combat.setup_combat)
        leadership += self._saruman_elite_bonus(units, side)

        companions = [l for l in leaders if isinstance(l, Companion)]
        minions = [l for l in leaders
                   if not isinstance(l, (Companion, Nazgul))
                   and getattr(l, 'name', None) in MINION_INITIALS]
        plain_leaders = [l for l in leaders
                         if not isinstance(l, (Companion, Nazgul))
                         and getattr(l, 'name', None) not in MINION_INITIALS]
        has_nazgul = any(isinstance(l, Nazgul) for l in leaders)

        rg_tag = "rg_" + str(anchor_key).replace(" ", "_")
        tags = ("unit", f"army_{region_key}", rg_tag)
        ldr_tags = ("leader", f"army_{region_key}", rg_tag)

        # -- figure markers: at most ONE per present type ---------------
        figures = []            # ('img', PhotoImage) | ('nazgul', None)
        first_reg = next((u for u in units
                          if u.kind == UnitKind.REGULAR), None)
        first_eli = next((u for u in units if u.kind == UnitKind.ELITE), None)
        for unit in (first_reg, first_eli):
            if unit is None:
                continue
            icon_file = UNIT_ICON.get((unit.nation, unit.kind))
            img = self._load_unit_image(icon_file, icon_size) \
                if icon_file else None
            if img:
                figures.append(('img', img, tags))
        if plain_leaders:
            nation = getattr(plain_leaders[0], 'nation', None)
            img = self._load_unit_image(
                LEADER_ICON.get(nation, "GondorLeader.png"), icon_size)
            if img:
                figures.append(('img', img, tags))
        if has_nazgul:
            img = self._load_unit_image("Nazgul.png", icon_size)
            figures.append(('img', img, tags) if img
                           else ('nazgul', None, tags))

        # -- character portraits: second row(s) of the same cluster -----
        chars = []              # ('img', PhotoImage) | ('comp'|'min', ini)
        seen = set()
        for group, kind in ((companions, 'comp'), (minions, 'min')):
            for ch in group:
                ini = self._special_initials(ch) or '??'
                if ini in seen:
                    continue
                seen.add(ini)
                img = self._character_image(getattr(ch, 'name', None),
                                            icon_size)
                chars.append(('img', img, ldr_tags) if img
                             else (kind, ini, ldr_tags))

        # -- nominal offsets relative to the anchor (Java-style) --------
        items = []                          # (dx, dy, kind, payload, tags)
        fig_step = max(6, int(icon_size * 0.55))   # d.java: ~0.7 counter w
        n_fig = len(figures)
        for i, (kind, payload, tg) in enumerate(figures):
            dx = (i - (n_fig - 1) / 2.0) * fig_step
            items.append((dx, base_dy, kind, payload, tg))
        fig_half = (((n_fig - 1) / 2.0) * fig_step + icon_size * 0.5
                    if n_fig else 0.0)

        # R/E/L count label (L = total leadership value), adjacent
        label = f"{n_reg}/{n_eli}/{leadership}"
        font_sz = max(7, int(12 * self.zoom_level))
        lbl_w = int(font_sz * 0.75 * len(label)) + 6
        lbl_h = font_sz + 6
        items.append((fig_half + lbl_w / 2.0 + 2, base_dy, 'label',
                      (label, lbl_w, lbl_h, font_sz), tags))

        char_step = max(5, int(icon_size * 0.5))
        char_row_dy = max(6, int(icon_size * 0.45))  # d.java: chars at +20
        per_row = 4
        for r0 in range(0, len(chars), per_row):
            row = chars[r0:r0 + per_row]
            dy = base_dy + (r0 // per_row + 1) * char_row_dy
            for j, (kind, payload, tg) in enumerate(row):
                dx = (j - (len(row) - 1) / 2.0) * char_step
                items.append((dx, dy, kind, payload, tg))

        # -- scale-to-fit: overlap tighter, never spread out ------------
        max_off = max((dx * dx + dy * dy) ** 0.5
                      for dx, dy, _k, _p, _t in items)
        s = 1.0
        if max_r and max_off > 0:
            s = min(1.0, (0.85 * max_r) / max_off)

        # -- draw --------------------------------------------------------
        side_col = (COLORS["fp_primary"] if side == Side.FREE_PEOPLES
                    else COLORS["shadow_primary"])
        r = max(6, icon_size // 2)
        for dx, dy, kind, payload, tg in items:
            x = int(round(cx + s * dx))
            y = int(round(cy + s * dy))
            if kind == 'img':
                self.canvas.create_image(x, y, image=payload,
                                         anchor=tk.CENTER, tags=tg)
            elif kind == 'nazgul':      # fallback: dark winged triangle
                h = icon_size // 2
                self.canvas.create_polygon(
                    x - h, y + h, x, y - h, x + h, y + h,
                    fill="#1b1b2b", outline="#e94560", width=2, tags=tg)
            elif kind == 'comp':        # fallback: initials disc
                self.canvas.create_oval(
                    x - r, y - r, x + r, y + r,
                    fill="#2d5a8a", outline="#8fd0ff", width=2, tags=tg)
                self.canvas.create_text(
                    x, y, text=payload, fill="#ffffff",
                    font=("Arial", max(6, int(r * 0.9)), "bold"), tags=tg)
            elif kind == 'min':         # fallback: initials diamond
                self.canvas.create_polygon(
                    x - r, y, x, y - r, x + r, y, x, y + r,
                    fill="#8a2d3d", outline="#ff8080", width=2, tags=tg)
                self.canvas.create_text(
                    x, y, text=payload, fill="#ffffff",
                    font=("Arial", max(6, int(r * 0.9)), "bold"), tags=tg)
            elif kind == 'label':
                text, lw, lh, fsz = payload
                self.canvas.create_rectangle(
                    x - lw // 2, y - lh // 2, x + lw // 2, y + lh // 2,
                    fill="#101020", outline=side_col, width=1, tags=tg)
                # Two-pass label draw (shadow then colour) -- h_0.java
                # lines 108-132: black at +1/+1 under white/red
                self.canvas.create_text(
                    x + 1, y + 1, text=text, fill="#000000",
                    font=("Arial", fsz, "bold"), tags=tg)
                self.canvas.create_text(
                    x, y, text=text, fill=side_col,
                    font=("Arial", fsz, "bold"), tags=tg)

    def _draw_fellowship(self):
        """Draw the Fellowship token."""
        if self.state.fellowship.mordor_entered:
            return
        # LIVE mode hidden-info rule: a human playing Shadow never sees
        # the hidden Fellowship's true position on the board.
        if (getattr(self, 'live_hide_fellowship', False)
                and self.state.fellowship.hidden):
            return

        # Get fellowship region
        try:
            from game_logic import fellowship_region
            reg_name = fellowship_region(self.state)
            region = self._find_region(reg_name)
            if not region:
                return

            cx, cy = region.screen_xy
            cx = int(cx * self.zoom_level)
            cy = int(cy * self.zoom_level)

            # Load fellowship token (~30% larger: 32 -> 42 base)
            token_file = "Fellowship.png" if self.state.fellowship.hidden else "Fellowship_revealed.png"
            icon_size = max(26, int(42 * self.zoom_level))
            img = self._load_unit_image(token_file, icon_size)
            if img:
                # Offset slightly from armies
                self.canvas.create_image(
                    cx, cy - icon_size,
                    image=img, anchor=tk.CENTER,
                    tags=("fellowship",)
                )
        except Exception as e:
            print(f"Fellowship draw error: {e}")

    # ------------------------------------------------------------------
    # Siege / side helpers
    # ------------------------------------------------------------------

    def _unit_side(self, unit) -> Side:
        """Which side a unit belongs to."""
        return (Side.SHADOW if unit.nation in SHADOW_NATIONS_GUI
                else Side.FREE_PEOPLES)

    def _leader_side(self, leader) -> Side:
        """Which side a leader/companion/nazgul/minion belongs to."""
        if isinstance(leader, Nazgul):
            return Side.SHADOW
        if getattr(leader, 'name', '') in SHADOW_MINION_NAMES:
            return Side.SHADOW
        if getattr(leader, 'nation', None) in SHADOW_NATIONS_GUI:
            return Side.SHADOW
        return Side.FREE_PEOPLES

    def _stronghold_defender_side(self, san_key: str) -> Side:
        """Defender of a besieged stronghold: its native side unless the
        settlement was previously captured by the enemy."""
        try:
            from game_logic import is_shadow_region
            native = (Side.SHADOW if is_shadow_region(san_key)
                      else Side.FREE_PEOPLES)
        except Exception:
            native = Side.FREE_PEOPLES
        ec = getattr(self.state, 'enemy_controlled', {}) or {}
        if native == Side.FREE_PEOPLES and \
                san_key in ec.get(Side.SHADOW, set()):
            return Side.SHADOW
        if native == Side.SHADOW and \
                san_key in ec.get(Side.FREE_PEOPLES, set()):
            return Side.FREE_PEOPLES
        return native

    def _format_side_forces(self, army, side: Side) -> str:
        """Compact composition of one side's forces as 'R/E/L' (regulars /
        elites / total leadership) plus special-character initials."""
        if army is None:
            return "none"
        units = [u for u in army.units if self._unit_side(u) == side]
        leaders = [l for l in army.leaders if self._leader_side(l) == side]
        if not units and not leaders:
            return "none"
        n_r = sum(1 for u in units if u.kind == UnitKind.REGULAR)
        n_e = sum(1 for u in units if u.kind == UnitKind.ELITE)
        leadership = sum(self._leadership_value(l) for l in leaders)
        # Saruman's ability: Isengard Elites add +1 leadership each
        leadership += self._saruman_elite_bonus(units, side)
        text = f"{n_r}/{n_e}/{leadership}"
        specials = []
        for l in leaders:
            ini = self._special_initials(l)
            if ini and ini not in specials:
                specials.append(ini)
        if specials:
            text += " +" + "+".join(specials)
        return text

    def _draw_mordor_track(self):
        """Compact vertical Mordor-track strip on the right edge of the
        board canvas. Highlights the fellowship's step once in Mordor
        (the map pawn is not drawn then); dim otherwise."""
        if self._board_image is None:
            return
        try:
            board_w = self._board_image.width()
        except Exception:
            return

        z = self.zoom_level
        box_w = max(120, int(240 * z))
        box_h = max(26, int(48 * z))
        gap = max(3, int(6 * z))
        font_sz = max(7, int(13 * z))
        x1 = board_w - box_w - max(6, int(12 * z))
        y = max(6, int(12 * z))

        in_mordor = bool(self.state.fellowship.mordor_entered)
        step = self.state.fellowship.mordor_step if in_mordor else -1

        title_col = COLORS["highlight"] if in_mordor else COLORS["text_dim"]
        self.canvas.create_text(
            x1 + box_w // 2, y, text="MORDOR TRACK", anchor=tk.N,
            fill=title_col, font=("Arial", font_sz + 1, "bold"),
            tags=("mordor_track",))
        y += font_sz + 10

        for i, name in enumerate(MORDOR_TRACK_NAMES):
            active = (i == step)
            fill = "#5a1010" if active else "#241722"
            outline = COLORS["highlight"] if active else "#4a3a4a"
            txt_col = "#ffffff" if active else (
                COLORS["text"] if in_mordor else COLORS["text_dim"])
            self.canvas.create_rectangle(
                x1, y, x1 + box_w, y + box_h,
                fill=fill, outline=outline, width=3 if active else 1,
                tags=("mordor_track",))
            self.canvas.create_text(
                x1 + 6, y + box_h // 2, text=f"{i}  {name}",
                anchor=tk.W, fill=txt_col, font=("Arial", font_sz),
                tags=("mordor_track",))
            if active:
                # Fellowship marker on the active slot
                token_file = ("Fellowship.png" if self.state.fellowship.hidden
                              else "Fellowship_revealed.png")
                icon = max(16, int(box_h * 0.8))
                img = self._load_unit_image(token_file, icon)
                mx = x1 + box_w - icon // 2 - 4
                my = y + box_h // 2
                if img:
                    self.canvas.create_image(mx, my, image=img,
                                             anchor=tk.CENTER,
                                             tags=("mordor_track",))
                else:
                    r = icon // 2
                    self.canvas.create_oval(
                        mx - r, my - r, mx + r, my + r,
                        fill=COLORS["highlight"], outline="#ffffff",
                        tags=("mordor_track",))
            y += box_h + gap

    def _draw_hunt_box(self):
        """Visible hunt box under the Mordor strip: one die glyph per
        ActionDie in state.hunt_box (red = Shadow, blue = Free Peoples),
        with the face letter (E, C, W, ...). Empty box when no dice."""
        if self._board_image is None:
            return
        try:
            board_w = self._board_image.width()
        except Exception:
            return

        z = self.zoom_level
        # Same geometry as the Mordor strip, positioned just below it
        box_w = max(120, int(240 * z))
        box_h = max(26, int(48 * z))
        gap = max(3, int(6 * z))
        font_sz = max(7, int(13 * z))
        x1 = board_w - box_w - max(6, int(12 * z))
        y = (max(6, int(12 * z)) + font_sz + 10
             + len(MORDOR_TRACK_NAMES) * (box_h + gap)
             + max(8, int(14 * z)))

        dice = list(getattr(self.state, 'hunt_box', None) or [])
        title_col = COLORS["shadow_primary"] if dice else COLORS["text_dim"]
        self.canvas.create_text(
            x1 + box_w // 2, y, text="HUNT BOX", anchor=tk.N,
            fill=title_col, font=("Arial", font_sz + 1, "bold"),
            tags=("hunt_box",))
        y += font_sz + 8

        die_sz = max(18, int(34 * z))
        pad = max(4, int(8 * z))
        per_row = max(1, (box_w - pad) // (die_sz + pad))
        n_rows = (len(dice) + per_row - 1) // per_row if dice else 1
        frame_h = pad + n_rows * (die_sz + pad)
        self.canvas.create_rectangle(
            x1, y, x1 + box_w, y + frame_h,
            fill="#241722", outline="#4a3a4a", width=1, tags=("hunt_box",))

        for i, die in enumerate(dice):
            row, col = divmod(i, per_row)
            dx = x1 + pad + col * (die_sz + pad)
            dy = y + pad + row * (die_sz + pad)
            is_sh = getattr(die, 'side', None) == Side.SHADOW
            fill = "#7a1f1f" if is_sh else "#2d5a8a"
            outline = (COLORS["shadow_primary"] if is_sh
                       else COLORS["fp_primary"])
            face = getattr(die, 'current', None)
            letter = FACE_LETTER.get(face, '-')
            self.canvas.create_rectangle(
                dx, dy, dx + die_sz, dy + die_sz,
                fill=fill, outline=outline, width=2, tags=("hunt_box",))
            self.canvas.create_text(
                dx + die_sz // 2, dy + die_sz // 2, text=letter,
                fill="#ffffff",
                font=("Arial", max(8, int(die_sz * 0.45)), "bold"),
                tags=("hunt_box",))

    def _draw_selection(self):
        """Draw selection highlight on selected region."""
        region = self._find_region(self.selected_region)
        if not region:
            return

        cx, cy = region.screen_xy
        cx = int(cx * self.zoom_level)
        cy = int(cy * self.zoom_level)
        r = int(40 * self.zoom_level)

        self.canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            outline=COLORS["highlight"], width=3,
            tags=("selection",)
        )

    def _draw_political_track(self):
        """Draw the political track display."""
        self.polit_canvas.delete("all")

        # Nations in order: FP on left, Shadow on right
        fp_nations = [Nation.ELVES, Nation.DWARVES, Nation.GONDOR, Nation.ROHAN, Nation.THE_NORTH]
        sh_nations = [Nation.ISENGARD, Nation.SAURON, Nation.SOUTHRON_EASTERLINGS]
        all_nations = fp_nations + sh_nations

        cell_w = 40
        cell_h = 25
        start_x = 10
        start_y = 5

        # Draw header
        self.polit_canvas.create_text(
            start_x, start_y, text="Nation", anchor=tk.NW,
            fill=COLORS["text"], font=("Arial", 8)
        )
        for i in range(4):
            x = start_x + 60 + i * cell_w
            label = str(i) if i < 3 else "WAR"
            self.polit_canvas.create_text(
                x + cell_w//2, start_y, text=label, anchor=tk.N,
                fill=COLORS["text"], font=("Arial", 8)
            )

        # Draw each nation
        for idx, nation in enumerate(all_nations):
            y = start_y + 15 + idx * cell_h

            # Nation name (abbreviated)
            name = nation.value[:6]
            color = COLORS["fp_primary"] if nation in fp_nations else COLORS["shadow_primary"]
            self.polit_canvas.create_text(
                start_x, y + cell_h//2, text=name, anchor=tk.W,
                fill=color, font=("Arial", 8)
            )

            # Draw boxes
            box = self.state.political.boxes.get(nation, 0)
            active = self.state.political.active.get(nation, False)

            for i in range(4):
                x = start_x + 60 + i * cell_w
                fill = COLORS["bg"]
                if i == box:
                    # Draw marker
                    prefix = "n" if active else "p"
                    stub = POLIT_ICON.get(nation)
                    if stub:
                        fname = f"{prefix}{stub}.png"
                        img = self._polit_images.get(fname)
                        if img:
                            self.polit_canvas.create_image(
                                x + cell_w//2, y + cell_h//2,
                                image=img
                            )
                        else:
                            # Fallback: colored rectangle
                            fill = color
                            self.polit_canvas.create_rectangle(
                                x + 5, y + 2, x + cell_w - 5, y + cell_h - 2,
                                fill=fill, outline=COLORS["text"]
                            )
                else:
                    # Empty box
                    self.polit_canvas.create_rectangle(
                        x + 5, y + 2, x + cell_w - 5, y + cell_h - 2,
                        fill=fill, outline=COLORS["text_dim"]
                    )

    def _update_dice_display(self):
        """Update the dice display."""
        # Clear existing
        for widget in self.fp_dice_frame.winfo_children():
            widget.destroy()
        for widget in self.sh_dice_frame.winfo_children():
            widget.destroy()

        # Wrap dice into rows of 8: the 400px side panel fits 8 of the
        # 40px die cells — Shadow's pool (7 base + minion dice) ran
        # past the edge in a single LEFT-packed row and got clipped
        cur_row = {}

        def _row(frame, cell):
            if cell % 8 == 0 or frame not in cur_row:
                rf = tk.Frame(frame, bg=COLORS["panel_bg"])
                rf.pack(fill=tk.X, anchor=tk.W)
                cur_row[frame] = rf
            return cur_row[frame]

        # FP dice
        fp_dice = self.state.dice[Side.FREE_PEOPLES]
        cell = 0
        for i, die in enumerate(fp_dice):
            self._create_die_widget(_row(self.fp_dice_frame, cell),
                                    die, Side.FREE_PEOPLES, i)
            cell += 1

        # FP action tokens (displayed alongside dice)
        for token_available, img_key, label_text in [
            (self.state.fp_political_token, "freeaction1.png", "Pol"),
            (self.state.fp_card_token, "freeaction2.png", "Card"),
        ]:
            if token_available:
                frame = tk.Frame(_row(self.fp_dice_frame, cell),
                                 bg=COLORS["panel_bg"])
                frame.pack(side=tk.LEFT, padx=2, pady=2)
                cell += 1
                img = self._dice_images.get(img_key)
                if img:
                    lbl = tk.Label(frame, image=img, bg=COLORS["panel_bg"])
                    lbl.image = img
                else:
                    lbl = tk.Label(frame, text=label_text,
                                   bg=COLORS["button_bg"], fg=COLORS["button_fg"],
                                   width=4, height=2)
                lbl.pack()

        # Shadow dice
        sh_dice = self.state.dice[Side.SHADOW]
        for i, die in enumerate(sh_dice):
            self._create_die_widget(_row(self.sh_dice_frame, i),
                                    die, Side.SHADOW, i)

    def _create_die_widget(self, parent, die: ActionDie, side: Side, index: int):
        """Create a clickable die widget."""
        frame = tk.Frame(parent, bg=COLORS["panel_bg"])
        frame.pack(side=tk.LEFT, padx=2, pady=2)

        # Check if this is the selected die
        is_selected = (side == self.current_side and index == self.selected_die_idx)

        # Handle unrolled dice
        if die.current is None:
            label = tk.Label(
                frame, text="?",
                bg=COLORS["button_bg"], fg=COLORS["button_fg"],
                width=4, height=2, font=("Arial", 12, "bold")
            )
            label.pack()
            return

        # Get image
        img_key = (side, die.current)
        img = self._dice_images.get(img_key)

        if img:
            label = tk.Label(frame, image=img, bg=COLORS["panel_bg"])
            label.image = img  # Keep reference
        else:
            # Fallback text
            label = tk.Label(
                frame, text=die.current.value[:3],
                bg=COLORS["button_bg"], fg=COLORS["button_fg"],
                width=4, height=2
            )

        # Highlight if selected
        if is_selected:
            label.configure(highlightbackground=COLORS["highlight"], highlightthickness=3)

        # Mark used dice
        if die.used:
            label.configure(bg=COLORS["text_dim"])

        label.pack()

        # Click handler
        label.bind("<Button-1>", lambda e, s=side, i=index: self._on_die_click(s, i))

    def _on_die_click(self, side: Side, index: int):
        """Handle die click."""
        if side != self.current_side:
            self.log(f"Select {self.current_side.value} dice only")
            return

        die = self.state.dice[side][index]
        if die.current is None:
            self.log("Roll dice first!")
            return
        if die.used:
            self.log("Die already used")
            return

        self.selected_die_idx = index
        self.log(f"Selected {die.current.value} die")
        self._update_dice_display()

    def _update_hand_display(self):
        """Update the hand listbox."""
        self.hand_listbox.delete(0, tk.END)

        from cards import get_card
        hand = self.hands[self.current_side]

        for card_id in hand:
            card = get_card(card_id)
            if card:
                display = f"{card.card_name} ({card.deck})"
            else:
                display = card_id
            self.hand_listbox.insert(tk.END, display)

    def _on_card_select(self, event):
        """Show card details when selected."""
        selection = self.hand_listbox.curselection()
        if not selection:
            return

        idx = selection[0]
        hand = self.hands[self.current_side]
        if idx >= len(hand):
            return

        card_id = hand[idx]
        from cards import get_card
        card = get_card(card_id)

        self.card_detail.configure(state=tk.NORMAL)
        self.card_detail.delete(1.0, tk.END)

        if card:
            details = f"Name: {card.card_name}\n"
            details += f"Type: {card.card_type} | Deck: {card.deck}\n"
            if card.event_req:
                details += f"Requirement: {card.event_req}\n"
            if card.event_effect:
                details += f"Effect: {card.event_effect}\n"
            if card.combat_name:
                details += f"\nCombat: {card.combat_name} (Value: {card.combat_value})\n"
                if card.combat_req:
                    details += f"Combat Req: {card.combat_req}\n"
                if card.combat_effect:
                    details += f"Combat Effect: {card.combat_effect}"
        else:
            details = f"Card not found: {card_id}"

        self.card_detail.insert(tk.END, details)
        self.card_detail.configure(state=tk.DISABLED)

    def _update_info_labels(self):
        """Update info labels."""
        # Turn and side
        self.turn_label.configure(text=f"Turn {self.state.turn}")
        side_text = "Free Peoples" if self.current_side == Side.FREE_PEOPLES else "Shadow"
        side_color = COLORS["fp_primary"] if self.current_side == Side.FREE_PEOPLES else COLORS["shadow_primary"]
        self.side_label.configure(text=side_text, fg=side_color)

        # VP - calculate from settlements
        from game_logic import calculate_victory_points, FP_VP_VICTORY, SHADOW_VP_VICTORY
        vp = calculate_victory_points(self.state)
        fp_vp = vp["free_peoples"]
        sh_vp = vp["shadow"]
        # Display-only: calculate_victory_points is authoritative; a refresh
        # must never write back into state (replay snapshots stay untouched).

        # Display with threshold
        fp_text = f"FP VP: {fp_vp}/{FP_VP_VICTORY}"
        sh_text = f"Shadow VP: {sh_vp}/{SHADOW_VP_VICTORY}"

        self.fp_vp_label.configure(text=fp_text)
        self.sh_vp_label.configure(text=sh_text)

        # Fellowship
        from game_logic import fellowship_region
        try:
            reg = fellowship_region(self.state)
        except Exception as e:
            reg = "Unknown"
            self.log(f"fellowship_region failed: {e}")
        hidden = "Hidden" if self.state.fellowship.hidden else "Revealed"
        # LIVE mode hidden-info rule (matches _draw_fellowship)
        if (getattr(self, 'live_hide_fellowship', False)
                and self.state.fellowship.hidden):
            reg = "(hidden from Shadow)"
        fel_text = f"Location: {reg} ({hidden})\n"
        fel_text += f"Progress: {self.state.fellowship.progress} | Corruption: {self.state.fellowship.corruption}/12\n"
        fel_text += f"Companions: {len(self.state.fellowship.companions)}"
        self.fel_info.configure(text=fel_text)

        # Pools table (R=regular E=elite L=leader). Reinforcements are
        # COMPUTED: max - deployed (- casualties for FP, whose losses
        # are permanent; SH casualties return to the pool).
        try:
            import ai as _ai_p
            dep = _ai_p._count_deployed_all(self.state)
            fcas = getattr(self.state, 'fp_casualties', None) or {}
            rows = []
            for nat in _ai_p.ALL_NATIONS:
                mx = _ai_p.NATION_MAX_FIGURES.get(nat, {})
                d = dep.get(nat, {})
                is_fp = _ai_p._side_of_nation(nat) == Side.FREE_PEOPLES
                c = fcas.get(nat, {}) if is_fp else {}
                # ONE ANSWERER (USER ruling 2026-08-18): the viewer must
                # show the SAME availability the engine enforces, so it
                # asks rather than re-deriving.
                pool = {t: _ai_p.figures_available(self.state, nat, t,
                                                   dep)
                        for t in ('regular', 'elite', 'leader')}
                cs = ("%d/%d/%d" % (c.get('regular', 0), c.get('elite', 0),
                                    c.get('leader', 0)) if is_fp else " - ")
                rows.append("%-11s %d/%d/%d | %s | %d/%d/%d" % (
                    nat.value[:11],
                    d.get('regular', 0), d.get('elite', 0), d.get('leader', 0),
                    cs,
                    pool['regular'], pool['elite'], pool['leader']))
            self.pools_info.configure(text="\n".join(rows))
        except Exception as e:
            # The pools panel exists to surface conservation breaks; a
            # failure here must be visible, never a stale/blank panel.
            self.pools_info.configure(text=f"POOLS ERROR: {e}")

        # Update status based on pending action
        if self.pending_action == "move":
            self.status_label.configure(text="Select destination region", fg=COLORS["warning"])
        elif self.pending_action == "attack":
            self.status_label.configure(text="Select target region", fg=COLORS["error"])
        else:
            self.status_label.configure(text="Click a region to select", fg=COLORS["text_dim"])

    def _update_companions_display(self):
        """Update the companions listbox."""
        self.companions_listbox.delete(0, tk.END)

        # Add guide indicator
        guide = self.state.fellowship.guide

        for comp in self.state.fellowship.companions:
            marker = "★ " if comp == guide else "  "
            self.companions_listbox.insert(tk.END, f"{marker}{comp.name} (Lv.{comp.level})")

    def _update_hunt_box_display(self):
        """Update the hunt box display."""
        # Count eyes and character dice in hunt box
        eyes = sum(1 for d in self.state.hunt_box if d.current == ActionDieFace.EYE)
        char_dice = sum(1 for d in self.state.hunt_box if d.current == ActionDieFace.CHARACTER)

        self.hunt_box_label.configure(text=f"Eyes: {eyes} | Character dice: {char_dice}")

        # Hunt pool composition — THREE visible pools with exact tiles
        # (user directive 2026-07-07: current pool / used (eyes + specials
        # return to the pool on Mordor entry) / removed from game)
        def _tile_txt(t):
            # Named specials keep their identity even when is_eye is
            # set (mirrors HuntTile.__repr__ ordering).
            nm = getattr(t, 'special_name', None)
            if nm:
                return str(nm)
            if getattr(t, 'is_eye', False):
                return 'Eye'
            base = str(getattr(t, 'damage', 0))
            if getattr(t, 'is_special', False):
                base += ('b' if getattr(t, 'is_fellowship_special', False)
                         else 'x')
            return base + ('r' if getattr(t, 'reveal', False) else '')

        def _pool_line(title, tiles):
            if not tiles:
                return f"{title}: -"
            from collections import Counter
            cnt = Counter(_tile_txt(t) for t in tiles)
            body = ' '.join(f"{k}x{v}" if v > 1 else k
                            for k, v in sorted(cnt.items()))
            return f"{title} ({len(tiles)}): {body}"

        lines = [
            _pool_line('Pool', self.state.hunt_pool),
            _pool_line('Used', getattr(self.state, 'hunt_discard', [])),
            _pool_line('Removed', getattr(self.state, 'hunt_removed', [])),
        ]
        self.hunt_pool_label.configure(text="\n".join(lines),
                                       justify=tk.LEFT)

    def _update_rings_display(self):
        """Update the Elven Rings boxes and the persistent 'In Play' list."""
        fp = getattr(self.state, 'fp_elven_rings', 0)
        sh = getattr(self.state, 'sh_elven_rings', 0)
        dot = "●"  # filled circle per ring
        self.fp_rings_label.configure(
            text=f"FP Rings: {fp} {dot * fp}".rstrip())
        self.sh_rings_label.configure(
            text=f"SH Rings: {sh} {dot * sh}".rstrip())

        pc = getattr(self.state, 'persistent_cards', None)
        entries = []
        if pc is not None:
            entries += [f"FP: {c.card_name}"
                        for c in (getattr(pc, 'fp_cards', None) or [])]
            entries += [f"SH: {c.card_name}"
                        for c in (getattr(pc, 'sh_cards', None) or [])]
        text = "In Play: " + ("; ".join(entries) if entries else "—")
        self.inplay_label.configure(text=text)

    def _update_special_characters_display(self):
        """Update the special characters display."""
        # Free Peoples special characters
        fp_chars = []
        if self.state.gandalf_the_white and self.state.gandalf_the_white.in_play:
            fp_chars.append("Gandalf the White")
        if self.state.aragorn_crowned and self.state.aragorn_crowned.crowned:
            fp_chars.append("Aragorn")

        if fp_chars:
            self.fp_special_label.configure(text=f"FP: {', '.join(fp_chars)}")
        else:
            self.fp_special_label.configure(text="FP: —")

        # Shadow minions
        sh_chars = []
        if self.state.witch_king and self.state.witch_king.in_play:
            sh_chars.append("Witch King")
        if self.state.saruman and self.state.saruman.in_play:
            sh_chars.append("Saruman")
        if self.state.mouth_of_sauron and self.state.mouth_of_sauron.in_play:
            sh_chars.append("Mouth of Sauron")

        if sh_chars:
            self.sh_special_label.configure(text=f"Shadow: {', '.join(sh_chars)}")
        else:
            self.sh_special_label.configure(text="Shadow: —")

    def _find_army_sanitized(self, san_key: str):
        """Find an army whose key sanitizes to san_key (keys vary in case)."""
        for k, a in self._display_state().armies.items():
            if sanitize(str(k)) == san_key:
                return a
        return None

    def _update_siege_display(self):
        """Update siege panel: garrison boxes for besieged strongholds +
        the stronghold status list."""
        besieged = {sanitize(str(k)) for k in
                    getattr(self.state, 'besieged_strongholds', set())}

        # --- Garrison boxes (one per active siege) ---
        for w in self.siege_boxes_frame.winfo_children():
            w.destroy()
        for san in sorted(besieged):
            region = self._find_region(san)
            pretty = getattr(region, 'orig_name', None) or san.title()
            def_side = self._stronghold_defender_side(san)
            atk_side = (Side.SHADOW if def_side == Side.FREE_PEOPLES
                        else Side.FREE_PEOPLES)
            army = self._find_army_sanitized(san)
            garrison = self._format_side_forces(army, def_side)
            besieger = self._format_side_forces(army, atk_side)
            def_tag = "FP" if def_side == Side.FREE_PEOPLES else "SH"
            def_col = (COLORS["fp_primary"] if def_side == Side.FREE_PEOPLES
                       else COLORS["shadow_primary"])
            atk_col = (COLORS["fp_primary"] if atk_side == Side.FREE_PEOPLES
                       else COLORS["shadow_primary"])
            box = tk.LabelFrame(
                self.siege_boxes_frame, text=f"SIEGE: {pretty}",
                bg=COLORS["panel_bg"], fg=COLORS["warning"],
                font=("Arial", 8, "bold"), bd=2, relief=tk.RIDGE)
            box.pack(fill=tk.X, pady=2)
            tk.Label(box, text=f"Garrison ({def_tag}): {garrison}",
                     bg=COLORS["panel_bg"], fg=def_col,
                     font=("Arial", 8), anchor=tk.W, justify=tk.LEFT,
                     wraplength=150).pack(fill=tk.X, padx=3)
            tk.Label(box, text=f"Besieger: {besieger}",
                     bg=COLORS["panel_bg"], fg=atk_col,
                     font=("Arial", 8), anchor=tk.W, justify=tk.LEFT,
                     wraplength=150).pack(fill=tk.X, padx=3, pady=(0, 2))

        # --- Stronghold status list ---
        for key, (lbl, orig_name) in self.siege_labels.items():
            if key in besieged:
                lbl.configure(fg=COLORS["warning"],
                              font=("Arial", 8, "bold"),
                              text=f">> {orig_name} << (siege)")
            else:
                lbl.configure(fg=COLORS["text_dim"],
                              font=("Arial", 8),
                              text=f"  {orig_name}")

    def _check_victory_conditions(self):
        """Check all victory conditions and display victory message if met."""
        from game_logic import check_all_victory_conditions, should_game_end

        if self.replay_steps is not None:
            return  # Replay mode: outcome shown in the replay bar, no popups

        if getattr(self, 'live_session', None) is not None:
            # LIVE mode: the ENGINE decides victory with the proper
            # rulebook p.17 timing (military = end-of-turn Victory Check
            # only); this naive per-refresh check would announce early.
            # _live_finish announces the authoritative result once.
            return

        # Announce-once latch: victory state is monotone, so without this
        # every refresh after game end would re-open the modal.
        if getattr(self, '_victory_announced', False):
            return

        # P1-08 TIMING (rulebook p.17, ruling 1.212a): ring endings are
        # immediate, mid-action; a MILITARY victory exists only once the
        # engine's end-of-turn Victory Check (game_logic._settle_end_of_turn,
        # run by next_turn / ai._safe_next_turn / the live driver's
        # begin_turn_reset) has declared it and stored
        # _search_endturn_victory.  A military threshold reached mid-turn
        # can still be undone (recapture) before that check.
        result = getattr(self.state, '_search_endturn_victory', None)
        if result is None:
            result = check_all_victory_conditions(self.state)
            if not should_game_end(result, end_of_turn=False):
                return
        self._victory_announced = True

        winner = result["victory"]  # "free_peoples" or "shadow"
        victory_type = result["victory_type"]  # "ring_corruption", "ring_destroyed", or "military"

        if winner == "free_peoples":
            if victory_type == "ring_destroyed":
                title = "Free Peoples Victory!"
                msg = "The Ring has been destroyed in Mount Doom!\nThe Free Peoples win!"
            else:  # military
                vp = result["free_peoples_vp"]
                title = "Free Peoples Victory!"
                msg = f"The Free Peoples have captured enough Shadow strongholds!\nVictory Points: {vp}\nThe Free Peoples win!"
        else:  # shadow
            if victory_type == "ring_corruption":
                corruption = self.state.fellowship.corruption
                title = "Shadow Victory!"
                msg = f"The Ring-bearer has been corrupted!\nCorruption: {corruption}\nSauron wins!"
            else:  # military
                vp = result["shadow_vp"]
                title = "Shadow Victory!"
                msg = f"The Shadow armies have conquered the Free Peoples!\nVictory Points: {vp}\nSauron wins!"

        # Show victory dialog
        messagebox.showinfo(title, msg)
        self.log(f"*** {title} ***")
        self.log(msg)

    def _on_muster_minion(self):
        """Handle mustering a shadow minion."""
        if self.ai_playing:
            self.log("Wait for AI to finish its turn")
            return
        import game_logic as gl

        if self.current_side != Side.SHADOW:
            self.log("Only Shadow can muster minions")
            return

        # Check for muster die
        valid = (ActionDieFace.MUSTER, ActionDieFace.MUSTER_ARMY,
                 ActionDieFace.WILL_OF_WEST)
        if not self._check_die_selected(valid):
            return

        # Check which minions can be mustered
        available_minions = []

        can_wk, reason_wk, regions_wk = gl.can_muster_witch_king(self.state)
        if can_wk:
            available_minions.append(("Witch King", regions_wk))

        can_sar, reason_sar, regions_sar = gl.can_muster_saruman(self.state)
        if can_sar:
            available_minions.append(("Saruman", regions_sar))

        can_mouth, reason_mouth, regions_mouth = gl.can_muster_mouth_of_sauron(self.state)
        if can_mouth:
            available_minions.append(("Mouth of Sauron", regions_mouth))

        if not available_minions:
            reasons = []
            if not can_wk:
                reasons.append(f"Witch King: {reason_wk}")
            if not can_sar:
                reasons.append(f"Saruman: {reason_sar}")
            if not can_mouth:
                reasons.append(f"Mouth: {reason_mouth}")
            self.log("No minions available. " + "; ".join(reasons))
            return

        self._show_muster_minion_dialog(available_minions)

    def _show_muster_minion_dialog(self, available_minions):
        """Show dialog to select and muster a minion."""
        import game_logic as gl

        dialog = tk.Toplevel(self.root)
        dialog.title("Muster Minion")
        dialog.configure(bg=COLORS["panel_bg"])
        dialog.geometry("400x400")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(
            dialog, text="Muster Shadow Minion",
            bg=COLORS["panel_bg"], fg=COLORS["highlight"],
            font=("Arial", 12, "bold")
        ).pack(pady=10)

        # Minion selection
        tk.Label(dialog, text="Select Minion:", bg=COLORS["panel_bg"], fg=COLORS["text"]).pack(pady=5)

        minion_var = tk.StringVar(value="")

        regions_listbox = tk.Listbox(dialog, bg=COLORS["bg"], fg=COLORS["text"], height=8)

        def update_regions(*args):
            regions_listbox.delete(0, tk.END)
            selected = minion_var.get()
            for name, regions in available_minions:
                if name == selected:
                    for r in sorted(regions):
                        regions_listbox.insert(tk.END, r)
                    break

        for name, regions in available_minions:
            rb = tk.Radiobutton(
                dialog, text=name, variable=minion_var, value=name,
                bg=COLORS["panel_bg"], fg=COLORS["text"],
                selectcolor=COLORS["bg"], activebackground=COLORS["panel_bg"],
                command=update_regions
            )
            rb.pack(anchor=tk.W, padx=20)

        tk.Label(dialog, text="Select Region:", bg=COLORS["panel_bg"], fg=COLORS["text"]).pack(pady=5)
        regions_listbox.pack(fill=tk.X, padx=10, pady=5)

        result = {"done": False, "success": False}

        def muster():
            minion = minion_var.get()
            sel = regions_listbox.curselection()
            if not minion:
                messagebox.showwarning("Select Minion", "Please select a minion")
                return
            if not sel:
                messagebox.showwarning("Select Region", "Please select a region")
                return

            region = regions_listbox.get(sel[0])

            from ai import Action
            ok = self._run_engine_action(
                Action(die_idx=self.selected_die_idx,
                       action_type='muster_minion',
                       params={'minion': minion, 'region': region}),
                log_msg=f"Mustered {minion} in {region}")
            result["success"] = ok
            result["done"] = True
            dialog.destroy()

        def cancel():
            result["done"] = True
            dialog.destroy()

        btn_frame = tk.Frame(dialog, bg=COLORS["panel_bg"])
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        self._make_button(btn_frame, "Muster", muster, bg=COLORS["success"]).pack(side=tk.LEFT, padx=5)
        self._make_button(btn_frame, "Cancel", cancel).pack(side=tk.LEFT, padx=5)

        dialog.wait_window()
        if result["success"]:
            self.refresh()

    def _on_transform_gandalf(self):
        """Handle transforming Gandalf the Grey to Gandalf the White."""
        if self.ai_playing:
            self.log('Wait for AI to finish its turn')
            return
        import game_logic as gl

        if self.current_side != Side.FREE_PEOPLES:
            self.log("Only Free Peoples can transform Gandalf")
            return

        # Check for Will of the West die
        valid = (ActionDieFace.WILL_OF_WEST,)
        if not self._check_die_selected(valid):
            return

        can, reason, valid_regions = gl.can_transform_gandalf_white(self.state)
        if not can:
            self.log(f"Cannot transform Gandalf: {reason}")
            return

        self._show_transform_gandalf_dialog(valid_regions)

    def _show_transform_gandalf_dialog(self, valid_regions):
        """Show dialog to select region for Gandalf the White."""
        import game_logic as gl

        dialog = tk.Toplevel(self.root)
        dialog.title("Transform Gandalf")
        dialog.configure(bg=COLORS["panel_bg"])
        dialog.geometry("350x400")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(
            dialog, text="Gandalf the Grey → Gandalf the White",
            bg=COLORS["panel_bg"], fg=COLORS["highlight"],
            font=("Arial", 12, "bold")
        ).pack(pady=10)

        tk.Label(
            dialog, text="Gandalf the White:\n• Level 3, Leadership 1\n• Movement 4 when alone/with hobbit\n• Can cancel all Nazgûl leadership",
            bg=COLORS["panel_bg"], fg=COLORS["text"], justify=tk.LEFT
        ).pack(pady=5)

        tk.Label(dialog, text="Select Region:", bg=COLORS["panel_bg"], fg=COLORS["text"]).pack(pady=5)

        listbox = tk.Listbox(dialog, bg=COLORS["bg"], fg=COLORS["text"], height=8)
        for r in sorted(valid_regions):
            listbox.insert(tk.END, r)
        listbox.pack(fill=tk.X, padx=10, pady=5)

        result = {"done": False, "success": False}

        def transform():
            sel = listbox.curselection()
            if not sel:
                messagebox.showwarning("Select Region", "Please select a region")
                return

            region = listbox.get(sel[0])

            from ai import Action
            ok = self._run_engine_action(
                Action(die_idx=self.selected_die_idx,
                       action_type='transform_gandalf',
                       params={'region': region}),
                log_msg=f"Gandalf the White appeared in {region}!")
            result["success"] = ok
            result["done"] = True
            dialog.destroy()

        def cancel():
            result["done"] = True
            dialog.destroy()

        btn_frame = tk.Frame(dialog, bg=COLORS["panel_bg"])
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        self._make_button(btn_frame, "Transform", transform, bg=COLORS["success"]).pack(side=tk.LEFT, padx=5)
        self._make_button(btn_frame, "Cancel", cancel).pack(side=tk.LEFT, padx=5)

        dialog.wait_window()
        if result["success"]:
            self.refresh()

    def _on_crown_aragorn(self):
        """Handle crowning Strider as Aragorn."""
        if self.ai_playing:
            self.log('Wait for AI to finish its turn')
            return
        import game_logic as gl

        if self.current_side != Side.FREE_PEOPLES:
            self.log("Only Free Peoples can crown Aragorn")
            return

        # Check for Will of the West die
        valid = (ActionDieFace.WILL_OF_WEST,)
        if not self._check_die_selected(valid):
            return

        can, reason, valid_regions = gl.can_crown_aragorn(self.state)
        if not can:
            self.log(f"Cannot crown Aragorn: {reason}")
            return

        # Show confirmation
        region = valid_regions[0] if valid_regions else "unknown"
        if messagebox.askyesno(
            "Crown Aragorn",
            f"Crown Strider as King Aragorn in {region}?\n\n"
            f"Aragorn: Level 3, Leadership 2, Captain of the West\n"
            f"(Grants +1 die starting next turn)"
        ):
            from ai import Action
            self._run_engine_action(
                Action(die_idx=self.selected_die_idx,
                       action_type='crown_aragorn', params={}),
                log_msg="Aragorn has been crowned King!")

    def _on_separate_companions(self):
        """Separate the companions selected in the Fellowship list (one
        Character / Will of the West die moves one companion or one GROUP;
        range = Fellowship progress + the group's highest level).

        P1-10 (I1): the destinations are the ENGINE's companion_separate
        rows for exactly the selected group and die, and the move executes
        through ai._execute_action - the engine owns the Mordor bar, the
        siege/stronghold rules, nation activation and the guide change
        (a decision driven by _drive_engine_decisions).  The GUI no longer
        computes reachability or edits the Fellowship / state.armies."""
        if self.ai_playing:
            self.log('Wait for AI to finish its turn')
            return
        if self.current_side != Side.FREE_PEOPLES:
            self.log("Only Free Peoples can separate companions")
            return

        # Check for character die
        valid = (ActionDieFace.CHARACTER, ActionDieFace.WILL_OF_WEST)
        if not self._check_die_selected(valid):
            return

        # Get selected companions
        selection = self.companions_listbox.curselection()
        if not selection:
            self.log("Select companion(s) to separate")
            return
        companions = self.state.fellowship.companions
        names = sorted(companions[i].name for i in selection
                       if i < len(companions))
        if not names:
            self.log("No valid companions selected")
            return

        acts = self._engine_die_menu(Side.FREE_PEOPLES,
                                     self.selected_die_idx,
                                     ('companion_separate',))
        if acts is None:
            return
        acts = [a for a in acts
                if sorted(a.params.get('companions')
                          or [a.params.get('companion')]) == names]
        if not acts:
            self.log(f"The engine offers no separation of "
                     f"{', '.join(names)} with this die")
            return
        groups, _done = self._figure_move_groups(acts)
        pick = self._pick_figure_move("Separate Companions", groups)
        if pick is None:
            self.log("Separation cancelled")
            return
        self._run_figure_moves(pick, "Separated")

    def _on_move_companions(self):
        """Move companions already on the board (not in fellowship) with
        one Character / Will of the West die, each group within its own
        allowance.

        P1-10 (I1): the menu is the ENGINE's companion_move rows for the
        die, each move executes through ai._execute_action, and further
        groups on the SAME die are the engine's '__companion_moves__'
        chain."""
        if self.ai_playing:
            self.log('Wait for AI to finish its turn')
            return
        if self.current_side != Side.FREE_PEOPLES:
            self.log("Only Free Peoples can move companions")
            return

        # Check for character die
        valid = (ActionDieFace.CHARACTER, ActionDieFace.WILL_OF_WEST)
        if not self._check_die_selected(valid):
            return

        acts = self._engine_die_menu(Side.FREE_PEOPLES,
                                     self.selected_die_idx,
                                     ('companion_move',))
        if acts is None:
            return
        if not acts:
            self.log("No companion on the board can move with this die")
            return
        groups, _done = self._figure_move_groups(acts)
        pick = self._pick_figure_move("Move Companions", groups)
        if pick is None:
            self.log("Companion move cancelled")
            return
        self._run_figure_moves(pick, "Moved companions")

    # --- Character-die figure moves through the ENGINE (P1-10, I1) ---

    def _engine_die_menu(self, side, die_idx, action_types):
        """The engine's legal `action_types` rows for the selected die.

        get_legal_actions lists a die FACE once (on the first unused die
        showing it), so a second die with the same face is served that
        face's rows re-stamped with its own index (the executors check
        only the face of die_idx).  Returns None (logged) when no menu may
        be served now (an open engine decision)."""
        from ai import Action, get_legal_actions
        if self._ai_mode_decision_open():       # P1-07: never mid-chain
            return None
        if self._engine_pending_side() is not None:
            # manual mode: the engine serves only the open decision's menu
            self.log("An open decision must be resolved first - "
                     "action not taken")
            return None
        acts = get_legal_actions(self.state, side)
        rows = [a for a in acts if a.action_type in action_types]
        mine = [a for a in rows if a.die_idx == die_idx]
        if mine:
            return mine
        dice = self.state.dice[side]
        face = dice[die_idx].current
        rep = next((a.die_idx for a in rows
                    if isinstance(a.die_idx, int)
                    and 0 <= a.die_idx < len(dice)
                    and dice[a.die_idx].current == face), None)
        if rep is None:
            return []
        return [Action(die_idx=die_idx, action_type=a.action_type,
                       params=dict(a.params))
                for a in rows if a.die_idx == rep]

    def _region_display(self, region) -> str:
        """Board display name of a (sanitized) region key."""
        return board_module.regions.get(sanitize(str(region)), {}).get(
            'orig', str(region))

    def _figure_move_label(self, action):
        """(figures, from, to) of an engine figure move - a Character-die
        nazgul_move / mouth_move / companion_move / companion_separate row
        or a same-die chain option - or None for the chain's Done
        terminator."""
        p = action.params or {}
        at = action.action_type

        def _army_at(region):
            return next((a for k, a in self.state.armies.items()
                         if sanitize(str(k)) == sanitize(str(region))), None)

        if at == 'card_option':
            c = p.get('choices') or {}
            if c.get('_naz_done') or c.get('_comp_done'):
                return None
            src, dst = c.get('from_region'), c.get('to_region')
            if c.get('include_mouth'):
                who = "Mouth of Sauron"
            elif c.get('include_wk'):
                who = "Witch-king"
            elif c.get('companions'):
                who = ", ".join(c['companions'])
            else:
                who = f"{int(c.get('count', 1))} Nazgul"
        elif at == 'nazgul_move':
            src, dst = p.get('from'), p.get('to')
            army = _army_at(src)
            n = sum(1 for l in (army.leaders if army else ())
                    if isinstance(l, Nazgul))
            k = n if p.get('count') is None else int(p['count'])
            if p.get('include_wk'):
                who = "Witch-king" + (f" + {k} Nazgul" if k else "")
            else:
                who = f"{k} Nazgul"
        elif at == 'mouth_move':
            src, dst = p.get('from'), p.get('to')
            who = "Mouth of Sauron"
        elif at == 'companion_move':
            src, dst = p.get('from'), p.get('to')
            names = p.get('companions')
            if not names:
                army = _army_at(src)
                names = [l.name for l in (army.leaders if army else ())
                         if isinstance(l, Companion)]
            who = ", ".join(names)
        elif at == 'companion_separate':
            from game_logic import fellowship_region
            src, dst = fellowship_region(self.state), p.get('to_region')
            names = p.get('companions') or [p.get('companion')]
            who = "Separate " + ", ".join(names)
        else:
            raise ValueError(f"not a figure move: {at}")
        return who, src, dst

    def _figure_move_text(self, action) -> str:
        lab = self._figure_move_label(action)
        if lab is None:
            return "no further moves"
        who, src, dst = lab
        return (f"{who}: {self._region_display(src)} -> "
                f"{self._region_display(dst)}")

    def _figure_move_groups(self, acts):
        """Group engine figure-move rows for the two-list picker:
        ([(figures label, [(destination label, action), ...]), ...],
        done_action or None).  No state change."""
        groups = {}
        done = None
        for a in acts:
            lab = self._figure_move_label(a)
            if lab is None:
                done = a
                continue
            who, src, dst = lab
            key = f"{who} from {self._region_display(src)}"
            groups.setdefault(key, []).append((self._region_display(dst), a))
        return ([(k, sorted(rows, key=lambda r: r[0]))
                 for k, rows in sorted(groups.items())], done)

    def _pick_figure_move(self, title, groups, done_action=None):
        """Two-list picker over the ENGINE's figure-move rows: the figures
        (left), then a destination (right).  Returns the chosen engine
        Action, `done_action` for Done, or None when cancelled/closed."""
        result = {"value": None}

        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.configure(bg=COLORS["panel_bg"])
        dialog.geometry("640x480")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(
            dialog,
            text=("Legal moves for this die: pick the figures, then the "
                  "destination." + ("\nDone ends this die's moves."
                                    if done_action is not None else "")),
            bg=COLORS["panel_bg"], fg=COLORS["text"], justify=tk.LEFT,
            wraplength=600
        ).pack(pady=8, padx=10)

        lists = tk.Frame(dialog, bg=COLORS["panel_bg"])
        lists.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        left = tk.Listbox(lists, bg=COLORS["bg"], fg=COLORS["text"],
                          exportselection=False, width=40)
        lsb = tk.Scrollbar(lists, orient="vertical", command=left.yview)
        left.configure(yscrollcommand=lsb.set)
        right = tk.Listbox(lists, bg=COLORS["bg"], fg=COLORS["text"],
                           exportselection=False, width=30)
        rsb = tk.Scrollbar(lists, orient="vertical", command=right.yview)
        right.configure(yscrollcommand=rsb.set)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        lsb.pack(side=tk.LEFT, fill=tk.Y)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        rsb.pack(side=tk.LEFT, fill=tk.Y)

        for label, _rows in groups:
            left.insert(tk.END, label)

        def show_dests(_e=None):
            right.delete(0, tk.END)
            sel = left.curselection()
            if not sel:
                return
            for dest, _a in groups[sel[0]][1]:
                right.insert(tk.END, dest)

        left.bind("<<ListboxSelect>>", show_dests)
        if len(groups) == 1:
            left.selection_set(0)
            show_dests()

        def move(_e=None):
            ls, rs = left.curselection(), right.curselection()
            if not ls or not rs:
                messagebox.showwarning(
                    "Select", "Pick the figures and a destination",
                    parent=dialog)
                return
            result["value"] = groups[ls[0]][1][rs[0]][1]
            dialog.destroy()

        def done():
            result["value"] = done_action
            dialog.destroy()

        def cancel():
            dialog.destroy()

        right.bind("<Double-Button-1>", move)
        btn_frame = tk.Frame(dialog, bg=COLORS["panel_bg"])
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        self._make_button(btn_frame, "Move", move,
                          bg=COLORS["success"]).pack(side=tk.LEFT, padx=5)
        if done_action is not None:
            self._make_button(btn_frame, "Done", done).pack(side=tk.LEFT,
                                                            padx=5)
        else:
            self._make_button(btn_frame, "Cancel", cancel).pack(
                side=tk.LEFT, padx=5)
        dialog.protocol("WM_DELETE_WINDOW", cancel)

        dialog.wait_window()
        return result["value"]

    def _run_figure_moves(self, first, title) -> bool:
        """Execute a Character-die figure move through ai._execute_action
        (the one execution authority), then let THIS seat play the
        engine's same-die continuation ('__nazgul_moves__' /
        '__companion_moves__' pending) on the same picker; any other
        decision the moves open goes to _drive_engine_decisions.
        Returns True when the first move executed."""
        from ai import _execute_action, get_legal_actions
        if self._ai_mode_decision_open():       # P1-07: never mid-chain
            return False
        side = self.current_side
        try:
            _execute_action(self.state, first)
        except Exception as e:
            self.log(f"Move failed: {e}")
            import traceback
            traceback.print_exc()
            self.refresh()
            return False
        self.selected_die_idx = None
        self.log(f"{title}: {self._figure_move_text(first)}")
        while side != self.ai_side:
            pco = getattr(self.state, 'pending_card_option', None) or {}
            if (pco.get('card_name') not in ('__nazgul_moves__',
                                             '__companion_moves__')
                    or self._engine_pending_side() != side):
                break
            chain = [a for a in get_legal_actions(self.state, side)
                     if a.action_type == 'card_option']
            groups, done = self._figure_move_groups(chain)
            if done is None:
                raise RuntimeError(f"{pco.get('card_name')} chain has no "
                                   f"Done option")
            pick = self._pick_figure_move(f"{title} - same die", groups,
                                          done)
            if pick is None:
                # Closing the picker ends this die's moves - visibly.
                self.log("Picker closed - no further moves on this die")
                pick = done
            try:
                _execute_action(self.state, pick)
            except Exception as e:
                self.log(f"Move failed: {e}")
                import traceback
                traceback.print_exc()
                self.refresh()
                return True
            if pick is not done:
                self.log(f"{title}: {self._figure_move_text(pick)}")
        if self._engine_pending_side() is not None:
            self._drive_engine_decisions()
        if (self.ai_side and self.current_side != self.ai_side
                and not self.ai_playing):
            if hasattr(self, '_ai_after_id'):
                self.root.after_cancel(self._ai_after_id)
            self._ai_after_id = self.root.after(600, self._after_human_action)
        self.refresh()
        return True

    def _find_region(self, region_key: str) -> Optional[object]:
        """Find a region by name, handling various formats."""
        if not region_key:
            return None

        # Try direct lookup
        region = self.board._regions.get(region_key)
        if region:
            return region

        # Try sanitized
        san_key = sanitize(region_key)
        region = self.board._regions.get(san_key)
        if region:
            return region

        # Try normalized
        norm_key = normalize_region(region_key)
        for key, reg in self.board._regions.items():
            if normalize_region(key) == norm_key or normalize_region(reg.orig_name) == norm_key:
                return reg

        return None

    def _find_region_key(self, click_x: int, click_y: int) -> Optional[str]:
        """Find region key from click coordinates."""
        best_dist = float('inf')
        best_key = None

        for key, region in self.board._regions.items():
            cx, cy = region.screen_xy
            cx = int(cx * self.zoom_level)
            cy = int(cy * self.zoom_level)

            dist = ((click_x - cx) ** 2 + (click_y - cy) ** 2) ** 0.5
            if dist < best_dist and dist < 60 * self.zoom_level:
                best_dist = dist
                best_key = region.orig_name

        return best_key

    def _find_army_key(self, region_name: str) -> Optional[str]:
        """Find the army key for a region using SANITIZED matching like
        the rest of the codebase (placement executors create keys such
        as 'bree'; the board uses 'Bree'; normalize_region() disagrees
        with sanitize() on apostrophes — 'helms deep' vs 'helm s deep')."""
        if not region_name:
            return None
        st = self._display_state()

        # Direct lookup
        if region_name in st.armies:
            return region_name

        # Sanitized lookup (canonical)
        san = sanitize(region_name)
        for key in st.armies.keys():
            if sanitize(str(key)) == san:
                return key

        # Legacy normalized fallback
        norm = normalize_region(region_name)
        for key in st.armies.keys():
            if normalize_region(str(key)) == norm:
                return key

        return None

    def _on_canvas_click(self, event):
        """Handle canvas click."""
        # Get canvas coordinates (accounting for scroll)
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        region_key = self._find_region_key(int(x), int(y))

        if not region_key:
            return

        # Check if we're in a pending action mode
        if self.pending_action == "move" and self.action_source:
            self._complete_move(region_key)
            return
        elif self.pending_action == "attack" and self.action_source:
            self._complete_attack(region_key)
            return
        elif self.pending_action == "muster":
            self._complete_muster(region_key)
            return

        # Normal selection
        self.selected_region = region_key
        self._update_region_info(region_key)
        self.refresh()

    def _update_region_info(self, region_key: str):
        """Update the selected region info panel: per-side R/E/L counts,
        unit composition, and every leader by PROPER NAME (Companions:
        Meriadoc, ...; Minions: Saruman / Witch King / Mouth of Sauron).

        Reads the state the renderer last drew (_display_state) and uses
        sanitized army-key matching — the previous version crashed on
        Companions (no .nation attribute), silently leaving stale panel
        text, and showed Minions as 'Leader (sauron)'."""
        self.sel_region_label.configure(text=str(region_key))
        st = self._display_state()

        army_key = self._find_army_key(region_key)
        army = st.armies.get(army_key) if army_key else None

        if not army or (not army.units and not army.leaders):
            self.sel_army_label.configure(text="No army present")
            return

        lines = []
        for side, side_name in ((Side.FREE_PEOPLES, "Free Peoples"),
                                (Side.SHADOW, "Shadow")):
            units = [u for u in army.units if self._unit_side(u) == side]
            leaders = [l for l in army.leaders
                       if self._leader_side(l) == side]
            if not units and not leaders:
                continue
            n_r = sum(1 for u in units if u.kind == UnitKind.REGULAR)
            n_e = sum(1 for u in units if u.kind == UnitKind.ELITE)
            lead = sum(self._leadership_value(l) for l in leaders)
            lead += self._saruman_elite_bonus(units, side)
            lines.append(f"{side_name}: {n_r}R / {n_e}E / {lead}L")

            unit_counts = {}
            for u in units:
                key = f"{u.nation.value} {u.kind.value}"
                unit_counts[key] = unit_counts.get(key, 0) + 1
            for key in sorted(unit_counts):
                lines.append(f"  {unit_counts[key]}x {key}")

            n_naz = sum(1 for l in leaders if isinstance(l, Nazgul))
            if n_naz:
                lines.append(f"  {n_naz}x Nazgûl")
            for leader in leaders:
                if isinstance(leader, Nazgul):
                    continue
                lines.append(f"  {self._leader_display_name(leader)}")

        self.sel_army_label.configure(
            text="\n".join(lines) if lines else "No army present")

    # --- Action Handlers ---

    def _check_die_selected(self, valid_faces: tuple) -> bool:
        """Check if a valid die is selected."""
        if self.selected_die_idx is None:
            self.log("Select a die first")
            return False

        die = self.state.dice[self.current_side][self.selected_die_idx]
        if die.current is None:
            self.log("Roll dice first!")
            return False
        if die.used:
            self.log("Die already used")
            return False

        if die.current not in valid_faces:
            self.log(f"Need {[f.value for f in valid_faces]} die")
            return False

        return True

    def _count_unused_dice(self, side: Side) -> int:
        """Count unused dice for the given side."""
        return sum(1 for d in self.state.dice.get(side, [])
                   if not d.used and d.current is not None)

    def _run_engine_action(self, action, log_msg=None):
        """Execute a human action through ai._execute_action — the SAME
        engine path self-play and the teacher use (2026-07-08 consistency
        refactor).  _execute_action consumes the die itself, so we only
        reset the selection + drive any resulting pending decisions +
        schedule the AI reply.  Returns True on success."""
        from ai import _execute_action
        # R2-SL-34: the second recruit of the human's own open Muster die
        # is that window's continuation; any other top-level action is
        # refused while a multi-step window is open
        _cont = ('muster' if action.action_type == 'muster_recruit'
                 and self.state.pending_muster_die is not None
                 and action.die_idx == self.state.pending_muster_die
                 else None)
        if self._ai_mode_decision_open(continues=_cont):  # P1-07
            return False
        try:
            _execute_action(self.state, action)
        except Exception as e:
            self.log(f"Action failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        # the engine owns the hands (a card play removes the card even
        # when it opens no decision for _drive_engine_decisions to sync)
        self._sync_hands()
        if log_msg:
            self.log(log_msg)
        self.selected_die_idx = None
        # resolve any pending decisions the action created (combat/hunt/
        # card-option/muster-double/etc.) through the same engine driver
        if self._engine_pending_side() is not None:
            self._drive_engine_decisions()
        # R2-SL-34: no AI reply while the human's own multi-step window
        # (the first recruit of a Muster die just opened it) is open - the
        # second recruit, or the 'No' / Pass that declines it, arms it
        if (self.ai_side and self.current_side != self.ai_side
                and not self.ai_playing
                and self._human_multistep_window() is None):
            if hasattr(self, '_ai_after_id'):
                self.root.after_cancel(self._ai_after_id)
            self._ai_after_id = self.root.after(600, self._after_human_action)
        self.refresh()
        return True

    def _use_selected_die(self):
        """Mark the selected die as used.  It no longer arms the AI's
        reply (R2-SL-32): the callers still open modals after it (the
        card's engine decisions, the hunt prompts) and Tk services after()
        timers inside a modal's wait_window, so the AI ran in the middle
        of the human's own decision.  Each caller arms the reply with
        _arm_ai_reply at its very end."""
        if self.selected_die_idx is not None:
            self.state.dice[self.current_side][self.selected_die_idx].used = True
            self.selected_die_idx = None

    def _arm_ai_reply(self):
        """R2-SL-32: arm the AI's reply to a finished human action, once,
        after every modal the action opened has returned (AI mode only)."""
        if self.ai_side and self.current_side != self.ai_side and not self.ai_playing:
            if hasattr(self, '_ai_after_id'):
                self.root.after_cancel(self._ai_after_id)
            self._ai_after_id = self.root.after(600, self._after_human_action)

    def _on_roll_dice(self):
        """Roll all dice for both sides. In AI mode, Shadow AI allocates eyes first."""
        if self.ai_playing:
            self.log("Wait for AI to finish its turn")
            return
        from game_logic import roll_action_dice

        # One roll per turn: roll_action_dice re-randomizes every unused die
        # and re-appends Shadow Eyes to the hunt box on each call.
        if getattr(self.state, 'dice_rolled', False):
            self.log("Dice already rolled this turn")
            return

        # R2-SL-33: the turn-start Fellowship phase (discards, declaration,
        # tail with Mordor entry, Guide) precedes the Hunt allocation
        if not self._legacy_fellowship_phase():
            return

        # R2-RS-P1-09: the Shadow's Hunt allocation on the engine eye menu
        # (AI search, or a human Shadow's 1.271 minimum); False = refused
        # and logged, nothing rolled, the next ROLL DICE retries
        if not self._pre_roll_hunt_allocation():
            return

        # Roll dice for both sides
        roll_action_dice(self.state, pool=Side.FREE_PEOPLES)
        roll_action_dice(self.state, pool=Side.SHADOW)
        self.selected_die_idx = None

        # Log raw dice faces
        fp_faces = [d.current.value for d in self.state.dice[Side.FREE_PEOPLES]
                     if d.current and not d.used]
        sh_faces = [d.current.value for d in self.state.dice[Side.SHADOW]
                     if d.current and not d.used]
        eyes_in_box = len(self.state.hunt_box)
        self.log(f"FP dice: {', '.join(fp_faces)}")
        self.log(f"Shadow dice: {', '.join(sh_faces)} ({eyes_in_box} eye(s) in hunt box)")
        self.refresh()

    def _on_move_army(self):
        """Start army movement - with Army die can move 2 armies OR attack with 1."""
        if self.ai_playing:
            self.log("Wait for AI to finish its turn")
            return
        valid = (ActionDieFace.ARMY, ActionDieFace.MUSTER_ARMY, ActionDieFace.CHARACTER)
        if not self._check_die_selected(valid):
            return

        if not self.selected_region:
            self.log("Select a region with your army first")
            return

        # Check for army
        army_key = self._find_army_key(self.selected_region)
        army = self.state.armies.get(army_key) if army_key else None

        if not army or not army.units:
            self.log("No army in selected region")
            return

        # Check if army belongs to current side.  Besieged regions hold one
        # Army object with BOTH sides' units, so test for own units rather
        # than classifying the whole stack by its first unit.
        own_units = [u for u in army.units
                     if self._unit_side(u) == self.current_side]
        if not own_units:
            self.log("Not your army!")
            return

        # The engine's pending_army_move_die machinery owns the double-move
        # count; the GUI only derives the status suffix from the die face.
        die = self.state.dice[self.current_side][self.selected_die_idx]
        two_moves = die.current in (ActionDieFace.ARMY, ActionDieFace.MUSTER_ARMY)

        # Character die moves require a leader figure (engine menu gate:
        # require_leader=True; army_has_leader_figure is the authority).
        if die.current is ActionDieFace.CHARACTER:
            import game_logic as _gl_lead
            army_obj = self.state.armies.get(army_key)
            if not _gl_lead.army_has_leader_figure(self.state, army_obj,
                                                   side=self.current_side):
                self.log("Character die needs a leader in the army")
                return

        self.pending_action = "move"
        self.action_source = army_key
        moves_text = " (2 moves)" if two_moves else ""
        self.status_label.configure(
            text=f"Click destination for army from {self.selected_region}{moves_text}",
            fg=COLORS["warning"]
        )

    def _blocking_card_notice(self, card_name: str) -> None:
        """A persistent FP card blocks the move/attack; the blocking card
        is only removed via the engine's table-card-kill action."""
        self.log(f"Move blocked by {card_name} (discard it via the "
                 f"table-card kill action)")

    def _complete_move(self, dest_region: str):
        """Complete the movement through the ENGINE (2026-07-08): builds an
        army_move Action and runs it via ai._execute_action — the SAME
        path self-play/teacher use, including the pending_army_move_die
        double-move machinery.  The GUI no longer manages its own
        move-counter or calls game_logic.move_army directly."""
        from ai import Action, _execute_action
        # R2-SL-34: the human's own open Army-die window is continued here
        if self._ai_mode_decision_open(continues='army_move'):  # P1-07
            self.pending_action = None
            self.action_source = None
            return
        src_key = self.action_source
        dest_key = dest_region
        try:
            if not self._find_region(src_key) or not self._find_region(dest_key):
                raise ValueError("Invalid regions")
            # Shadow attack-block by a persistent FP card
            if self.current_side == Side.SHADOW:
                blocked, blocker = self._check_shadow_movement_blocked(dest_key)
                if blocked:
                    self._blocking_card_notice(blocker)
                    return
            # Enemy at destination = attack, not move
            dest_army = self.state.armies.get(self._find_army_key(dest_key))
            if dest_army and dest_army.units:
                from combat import side_of_army
                if side_of_army(dest_army) != self.current_side:
                    self.log(f"Enemy units in {dest_key}! Use Attack instead.")
                    return
            side_tag = 'fp' if self.current_side == Side.FREE_PEOPLES else 'sh'
            # whole-stack move (no composition params); engine sets
            # pending_army_move_die if this is an Army/Muster-Army die
            _execute_action(self.state, Action(
                die_idx=self.selected_die_idx, action_type='army_move',
                params={'from': sanitize(src_key), 'to': sanitize(dest_key),
                        'side': side_tag}))
            self.log(f"Moved army: {src_key} -> {dest_key}")
        except Exception as e:
            self.log(f"Move failed: {e}")
            import traceback
            traceback.print_exc()
            self.pending_action = None
            self.action_source = None
            self.refresh()
            return
        self.pending_action = None
        self.action_source = None
        # P1-07: resolve any engine decision the move opened through the
        # same driver _run_engine_action uses
        if self._engine_pending_side() is not None:
            self._drive_engine_decisions()
        # A second move is available iff the engine left a pending die
        if self.state.pending_army_move_die is not None:
            self.log("  (1 move left - select another army, or click Done)")
            self._offer_second_move()
        else:
            self.selected_die_idx = None
            # R2-SL-34: cancel-then-arm - a reply armed earlier never
            # fires beside this one (two AI actions in a row)
            self._arm_ai_reply()
            self.refresh()

    def _offer_second_move(self):
        """Offer the engine's pending second army move (Army/H die)."""
        from ai import Action, _execute_action
        if self.state.pending_army_move_die is None:
            self.refresh()
            return
        if messagebox.askyesno(
                "Second Movement",
                "You have 1 more army movement.\n\nMove another army?",
                parent=self.root):
            self.log("Select another army, then Move Army (2nd move) - "
                     "or press Pass to decline it")
            self.pending_action = "move"
            # the next _complete_move sees pending_army_move_die and the
            # engine applies it as the second move + consumes the die
        else:
            # Done: pass consumes the pending die
            _execute_action(self.state, Action(
                die_idx=self.state.pending_army_move_die,
                action_type='pass', params={}))
            self.selected_die_idx = None
            self._arm_ai_reply()        # R2-SL-34: cancel-then-arm
            self.refresh()

    def _on_attack(self):
        """Start attack action."""
        if self.ai_playing:
            self.log("Wait for AI to finish its turn")
            return
        # Character die: leader-attack (engine gate require_leader=True)
        valid = (ActionDieFace.ARMY, ActionDieFace.MUSTER_ARMY,
                 ActionDieFace.CHARACTER)
        if not self._check_die_selected(valid):
            return

        if not self.selected_region:
            self.log("Select a region with your army first")
            return

        army_key = self._find_army_key(self.selected_region)
        army = self.state.armies.get(army_key) if army_key else None

        if not army or not army.units:
            self.log("No army in selected region")
            return

        die = self.state.dice[self.current_side][self.selected_die_idx]
        if die.current is ActionDieFace.CHARACTER:
            import game_logic as _gl_lead
            if not _gl_lead.army_has_leader_figure(self.state, army,
                                                   side=self.current_side):
                self.log("Character die needs a leader in the army")
                return

        # Check if army can attack (all units must be from nations at
        # war) - QUEUE_4: side-filtered, so a shared co-located siege
        # entry judges only the acting side's nations
        from game_logic import can_attack_with_army
        can_attack, reason = can_attack_with_army(
            self.state, army, side=self.current_side)
        if not can_attack:
            self.log(f"Cannot attack: {reason}")
            return

        self.pending_action = "attack"
        self.action_source = army_key
        self.status_label.configure(text=f"Click target region to attack from {self.selected_region}", fg=COLORS["error"])

    def _complete_attack(self, target_region: str):
        """Complete the attack via the ENGINE (2026-07-08 consistency
        refactor): the GUI no longer has a parallel combat implementation
        — it builds an army_attack Action, executes it through
        ai._execute_action (the SAME path self-play and the teacher use),
        then drives every resulting combat decision through
        get_legal_actions + _execute_action so the rules are identical."""
        from ai import Action, _execute_action
        if self._ai_mode_decision_open():       # P1-07: never mid-chain
            self.pending_action = None
            self.action_source = None
            return
        try:
            # Shadow attack-block by a persistent FP card (A Power too
            # Great / Tom Bombadil) — offer the discard, as before.
            if self.current_side == Side.SHADOW:
                blocked, blocker = self._check_shadow_movement_blocked(target_region)
                if blocked:
                    self._blocking_card_notice(blocker)
                    return
            side_tag = 'fp' if self.current_side == Side.FREE_PEOPLES else 'sh'
            action = Action(die_idx=self.selected_die_idx,
                            action_type='army_attack',
                            params={'from': sanitize(self.action_source),
                                    'to': sanitize(target_region),
                                    'side': side_tag})
            _execute_action(self.state, action)  # consumes the die + sets pending_combat
            self.log(f"Attack: {self.action_source} -> {target_region}")
            self.selected_die_idx = None
            # Drive card selection / retreat / press / casualties / siege
            # through the engine's pending-combat state machine.
            self._drive_engine_decisions()
            if self.ai_side and self.current_side != self.ai_side and not self.ai_playing:
                self._ai_after_id = self.root.after(600, self._after_human_action)
        except Exception as e:
            self.log(f"Attack failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.pending_action = None
            self.action_source = None
            self.refresh()

    def _engine_pending_side(self):
        """Side that must decide on any engine pending state, else None.
        ai._pending_decision_side is the single source of truth for the
        cascade order; the GUI must never inline its own copy."""
        from ai import _pending_decision_side
        return _pending_decision_side(self.state, fallback=self.current_side)

    def _drive_engine_decisions(self):
        """Resolve every engine pending decision (combat/hunt/card-option/
        casualty/siege) through get_legal_actions + _execute_action — the
        one execution authority.  AI-controlled sides pick via the net;
        human sides are prompted with the legal actions."""
        from ai import (get_legal_actions, _execute_action)
        guard = 0
        while guard < 400:
            guard += 1
            side = self._engine_pending_side()
            if side is None:
                break
            acts = get_legal_actions(self.state, side)
            if not acts:
                break
            _execute_action(self.state,
                            self._choose_engine_decision(side, acts))
        self._sync_hands()
        self.refresh()

    def _choose_engine_decision(self, side, acts):
        """The pick for ONE open engine decision (`acts` = its legal menu):
        a one-row menu is that row, an AI side searches, a human side is
        asked in the required dialog.  Shared by _drive_engine_decisions
        and the turn-start driver's open decisions / Fellowship effects
        (_legacy_fellowship_phase, R3-LEGACY-GUI-TURN).  Raises, with
        nothing chosen or run, when the search or the dialog gives no
        answer."""
        from ai import mcts_action
        if len(acts) == 1:
            return acts[0]
        net = None
        if self.ai_side is not None and side == self.ai_side:
            net = self.ai_net
        if net is not None:
            # MCTS needs the true per-side pair: opponent nodes are
            # evaluated with the opponent's own net.
            fp_n = getattr(self, 'fp_net', None) or net
            sh_n = getattr(self, 'sh_net', None) or net
            a = mcts_action(self.state, side, fp_n, sh_n,
                            num_simulations=100, temperature=0.1)
            if a is None:
                # P1-12: never substitute acts[0] - the search did not
                # choose it (I1).  Stop loudly; the pending stays open.
                raise RuntimeError(
                    f"{side.value} AI search returned no action for "
                    f"its open engine decision - no action taken, the "
                    f"decision is still open")
            return a
        # human: present the legal decisions
        sidename = ('Free Peoples' if side == Side.FREE_PEOPLES
                    else 'Shadow')
        opts = [(self._describe_action(a), i) for i, a in enumerate(acts)]
        pick = self._show_choice_dialog(
            f"{sidename} decision", "Choose an action:", opts)
        if pick is None:
            # P1-13: never substitute acts[0] - the human did not
            # choose it (I1).  The dialog (required by default) has no
            # Cancel and refuses the close, so None means it was
            # destroyed with no choice (main window closing): stop
            # loudly, run nothing; the pending stays open.
            raise RuntimeError(
                f"{sidename} decision dialog closed without a choice "
                f"- no action taken, the decision is still open")
        return acts[pick]

    def _on_muster(self):
        """Start muster action - click on region."""
        if self.ai_playing:
            self.log("Wait for AI to finish its turn")
            return
        valid = (ActionDieFace.MUSTER, ActionDieFace.MUSTER_ARMY,
                 ActionDieFace.WILL_OF_WEST)
        if not self._check_die_selected(valid):
            return

        self.pending_action = "muster"
        self.status_label.configure(text="Click a settlement to muster in", fg=COLORS["warning"])

    def _complete_muster(self, region_key: str):
        """Complete muster in selected region."""
        try:
            # Check if region is a settlement
            san_key = sanitize(region_key)
            meta = board_module.regions.get(san_key)

            if not meta:
                raise ValueError(f"Region not found: {region_key}")

            if not (meta.get("is_city") or meta.get("is_stronghold") or meta.get("is_town")):
                raise ValueError(f"{region_key} is not a settlement")

            orig_name = meta.get("orig", region_key)

            # Second recruit of a Muster die: only regular/leader, in a
            # DIFFERENT settlement, paid with the engine's pending die.
            second = self.state.pending_muster_die is not None
            if second and san_key == getattr(self.state,
                                             'pending_muster_region', None):
                self.log("Second recruit must be in a different settlement")
                return

            # Ask what to muster via simple popup
            popup = tk.Toplevel(self.root)
            popup.title("Muster")
            popup.geometry("200x150")
            popup.configure(bg=COLORS["panel_bg"])

            tk.Label(popup, text=f"Muster in {orig_name}:",
                    bg=COLORS["panel_bg"], fg=COLORS["text"]).pack(pady=10)

            result = {"type": None}

            def muster_type(t):
                result["type"] = t
                popup.destroy()

            self._make_button(popup, "Regular", lambda: muster_type("regular"), width=10).pack(pady=5)
            if not second:
                self._make_button(popup, "Elite", lambda: muster_type("elite"), width=10).pack(pady=5)
            self._make_button(popup, "Leader", lambda: muster_type("leader"), width=10).pack(pady=5)

            popup.wait_window()

            if result["type"]:
                from ai import Action, _region_nation
                nation = _region_nation(san_key)
                atype = ('muster_elite' if result["type"] == "elite"
                         else 'muster_recruit')
                die_idx = (self.state.pending_muster_die if second
                           else self.selected_die_idx)
                ok = self._run_engine_action(
                    Action(die_idx=die_idx,
                           action_type=atype,
                           params={'nation': nation, 'region': san_key,
                                   'type': result["type"]}),
                    log_msg=f"Mustered {result['type']} in {orig_name}")
                if ok:
                    self.pending_action = None
                # engine sets pending_muster_die for the 2nd recruit
                # (Muster die = 2); offer it or finalize with pass
                if not second and self.state.pending_muster_die is not None:
                    self._offer_second_muster()
            else:
                self.pending_action = None
                self.refresh()

        except Exception as e:
            self.log(f"Muster failed: {e}")
            self.pending_action = None
            self.refresh()

    def _offer_second_muster(self):
        """Offer the engine's pending second recruit (Muster die = 2)."""
        from ai import Action, _execute_action
        if self.state.pending_muster_die is None:
            self.refresh()
            return
        if messagebox.askyesno(
                "Second Muster",
                "You have 1 more recruit.\n\nMuster in another settlement?",
                parent=self.root):
            self.log("Click another settlement to muster (2nd recruit) - "
                     "or press Pass to decline it")
            self.pending_action = "muster"
            # R2-SL-34: the AI reply is armed when the second recruit
            # lands (_run_engine_action) or the window is declined (Pass)
        else:
            _execute_action(self.state, Action(
                die_idx=self.state.pending_muster_die,
                action_type='pass', params={}))
            self.selected_die_idx = None
            # R2-SL-34: the human's Muster die is now spent - the AI
            # replies (as _offer_second_move's 'No' does)
            self._arm_ai_reply()
            self.refresh()

    def _on_political(self):
        """Political action - advance a nation."""
        if self.ai_playing:
            self.log('Wait for AI to finish its turn')
            return
        valid = (ActionDieFace.MUSTER, ActionDieFace.MUSTER_ARMY,
                 ActionDieFace.WILL_OF_WEST)
        if not self._check_die_selected(valid):
            return

        # Show nation selection popup
        popup = tk.Toplevel(self.root)
        popup.title("Advance Nation")
        popup.geometry("250x350")
        popup.configure(bg=COLORS["panel_bg"])

        tk.Label(popup, text="Select nation to advance:",
                bg=COLORS["panel_bg"], fg=COLORS["text"]).pack(pady=10)

        # Available nations based on side
        if self.current_side == Side.FREE_PEOPLES:
            nations = [Nation.ELVES, Nation.DWARVES, Nation.GONDOR, Nation.ROHAN, Nation.THE_NORTH]
        else:
            nations = [Nation.ISENGARD, Nation.SAURON, Nation.SOUTHRON_EASTERLINGS]

        for nation in nations:
            box = self.state.political.boxes.get(nation, 0)
            active = self.state.political.active.get(nation, False)
            status = "Active" if active else "Passive"
            at_war = " [AT WAR]" if box >= 3 else ""

            btn_text = f"{nation.value}: {box} ({status}){at_war}"
            btn = self._make_button(popup, btn_text,
                                   lambda n=nation: self._do_political(n, popup), width=25)
            btn.pack(pady=3)

    def _do_political(self, nation: Nation, popup):
        """Execute political advancement through the engine."""
        from ai import Action
        popup.destroy()
        self._run_engine_action(
            Action(die_idx=self.selected_die_idx,
                   action_type='advance_political',
                   params={'nation': nation}),
            log_msg=f"Advanced {nation.value} on political track")

    def _on_play_card(self):
        """Play a card from hand. Requires appropriate die to be selected."""
        if self.ai_playing:
            self.log("Wait for AI to finish its turn")
            return
        if self._ai_mode_decision_open():       # P1-07 / R2-SL-32
            return
        selection = self.hand_listbox.curselection()
        if not selection:
            self.log("Select a card from your hand first")
            return

        idx = selection[0]
        hand = self.hands[self.current_side]
        if idx >= len(hand):
            return

        card_id = hand[idx]
        from cards import get_card
        card = get_card(card_id)

        if not card:
            self.log(f"Card not found: {card_id}")
            return

        # Check if a die is selected
        if self.selected_die_idx is None:
            self.log("Select a die first to play a card")
            return

        dice = self.state.dice.get(self.current_side, [])
        if self.selected_die_idx >= len(dice):
            self.log("Invalid die selection")
            return

        die = dice[self.selected_die_idx]
        if die.used:
            self.log("Die already used")
            return

        # Determine which die faces can play this card type
        # Event/Palantir die can play any card
        # Character die plays Character cards
        # Army die plays Army cards
        # Muster die plays Muster cards
        card_type = card.card_type  # "Character", "Army", or "Muster"

        valid_faces = [ActionDieFace.EVENT, ActionDieFace.WILL_OF_WEST]
        if card_type == "Character":
            valid_faces.append(ActionDieFace.CHARACTER)
        elif card_type == "Army":
            valid_faces.extend([ActionDieFace.ARMY, ActionDieFace.MUSTER_ARMY])
        elif card_type == "Muster":
            valid_faces.extend([ActionDieFace.MUSTER, ActionDieFace.MUSTER_ARMY])

        if die.current not in valid_faces:
            face_names = [f.value for f in valid_faces]
            self.log(f"Cannot play {card_type} card with {die.current.value} die. Need: {', '.join(face_names)}")
            return

        try:
            from event_effects import check_event_requirement

            # Check requirements first
            can_play, msg = check_event_requirement(self.state, card_id)
            if not can_play:
                self.log(f"Cannot play: {msg}")
                return

            # GUI and engine hands must agree (kept in step by
            # _sync_hands); a missing card is a real desync and must
            # surface, never be papered over by forging the card in.
            if card_id not in self.state.hands[self.current_side].hand:
                self.log(f"Hand desync: {card_id} not in engine hand")
                return

            # R17-GUI-ENGINEPLAY (row 28): the play runs through the
            # engine's play_event executor (_run_engine_action ->
            # ai._execute_action), the one path self-play and live play
            # use.  The executor spends the die, opens an optioned
            # card's LEARNED card_option pending (USER 2026-07-27: no
            # legacy get_card_choices dialog), settles a card-driven
            # hunt, checks hand limits, and owns the bonus draws:
            # Gandalf the Grey's OPTIONAL same-deck draw at the end of
            # the action, after the first hand-limit discard, with the
            # guide checked then (_maybe_open_gandalf_draw), and the
            # Palantir of Orthanc's LEARNED which-deck draw once the
            # action has fully resolved (1.208m, no self-trigger:
            # _open_palantir_deck_choice).  The GUI used to call
            # play_event_card itself and then force both draws (same
            # deck, at once, before the card's own decisions) from its
            # own superseded copy of the rule.  _run_engine_action
            # drives every decision the play opens (the human's in the
            # required dialog) and arms the AI reply only after them
            # (R2-SL-32).
            from ai import Action
            self._run_engine_action(
                Action(die_idx=self.selected_die_idx,
                       action_type='play_event',
                       params={'card_id': card_id,
                               'side': ('fp' if self.current_side
                                        == Side.FREE_PEOPLES else 'sh')}),
                log_msg=f"Played: {card.card_name}")

        except Exception as e:
            self.log(f"Play card failed: {e}")
            import traceback
            traceback.print_exc()

        self.refresh()

    def _show_choice_dialog(self, title: str, prompt: str, options: list,
                            allow_cancel: bool = False) -> Optional[str]:
        """
        Show a dialog for single selection.
        Options: list of (display_text, value) tuples
        Returns the selected value, or None if cancelled (allow_cancel).

        P1-13: the options sit in a scrolled canvas.  The engine's legal
        surface is uncapped (the nazgul chain offers ~300 options with
        its Done LAST); the old fixed 300 x min(400, ...) window with no
        scroll container clipped every option past about the 8th, and
        the Cancel row with them.  Option texts wrap instead of clipping.
        By default the choice is REQUIRED (an engine pending the human
        must resolve): no Cancel and the window close is refused, so the
        caller never runs an action the human did not choose; None then
        means only that the dialog was destroyed without a choice (main
        window closing).  allow_cancel=True restores Cancel / close.
        """
        required = not allow_cancel
        if required:
            why = self._manual_controls_refusal()
            if why is not None:
                # every option click would be REFUSED by _make_button and
                # the close is refused: a modal the human could not leave
                raise RuntimeError("required decision '%s' cannot be "
                                   "answered here: %s" % (title, why))
        result = {"value": None, "cancelled": False}

        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.configure(bg=COLORS["panel_bg"])
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.focus_set()

        # Grows with the menu up to most of the screen; longer menus scroll
        width = 460
        dialog.geometry("{}x{}".format(width, max(220, min(
            int(dialog.winfo_screenheight() * 0.75),
            140 + len(options) * 40))))

        # Prompt label
        tk.Label(
            dialog, text=prompt,
            bg=COLORS["panel_bg"], fg=COLORS["text"],
            font=("Arial", 11), wraplength=width - 20
        ).pack(side=tk.TOP, pady=10, padx=10)

        # Bottom row - packed BEFORE the option list so pack never clips it
        bottom = tk.Frame(dialog, bg=COLORS["panel_bg"])
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        def cancel():
            result["cancelled"] = True
            dialog.destroy()

        if required:
            def refuse_close():
                self.log("A decision is required - choose one of the "
                         "%d options" % len(options))
                dialog.bell()
            dialog.protocol("WM_DELETE_WINDOW", refuse_close)
            tk.Label(
                bottom, text="Decision required: %d options (scroll for "
                             "more)" % len(options),
                bg=COLORS["panel_bg"], fg=COLORS["text_dim"],
                font=("Arial", 9)
            ).pack()
        else:
            cancel_btn = self._make_button(bottom, "Cancel", cancel, width=10)
            cancel_btn.pack()

        # Option buttons, in a vertically scrolled canvas
        list_frame = tk.Frame(dialog, bg=COLORS["panel_bg"])
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        canvas = tk.Canvas(list_frame, bg=COLORS["panel_bg"],
                           highlightthickness=0, width=width - 40)
        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL,
                                 command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        btn_frame = tk.Frame(canvas, bg=COLORS["panel_bg"])
        frame_item = canvas.create_window((0, 0), window=btn_frame,
                                          anchor="nw")

        def sync_scrollregion(_e=None):
            canvas.configure(scrollregion=(0, 0, btn_frame.winfo_reqwidth(),
                                           btn_frame.winfo_reqheight()))
        btn_frame.bind("<Configure>", sync_scrollregion)
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(
            frame_item, width=e.width))

        def on_wheel(event):
            num = getattr(event, "num", None)
            if num in (4, 5):
                step = -1 if num == 4 else 1
            else:
                step = -1 if event.delta > 0 else 1
            canvas.yview_scroll(2 * step, "units")
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            dialog.bind(seq, on_wheel)

        def select_option(value):
            result["value"] = value
            dialog.destroy()

        for display_text, value in options:
            btn = self._make_button(
                btn_frame, display_text,
                lambda v=value: select_option(v),
                width=25
            )
            btn.configure(wraplength=width - 80, justify=tk.LEFT)
            btn.pack(fill=tk.X, pady=3)
        btn_frame.update_idletasks()
        sync_scrollregion()

        # Wait for dialog to close
        dialog.wait_window()

        if result["cancelled"]:
            return None
        return result["value"]

    def _on_draw_cards(self):
        """Draw cards for both sides - 1 character + 1 strategy each per turn."""
        if self.ai_playing:
            self.log("Wait for AI to finish its turn")
            return
        from game_logic import draw_cards_for_side

        # One draw per turn: next_turn already draws for both sides on
        # every turn advance (this button covers only the turn-1 draw).
        if getattr(self, '_cards_drawn_turn', None) == self.state.turn:
            self.log("Cards already drawn this turn")
            return
        self._cards_drawn_turn = self.state.turn

        for side in [Side.FREE_PEOPLES, Side.SHADOW]:
            result = draw_cards_for_side(self.state, side)
            if result["drawn"]:
                names = [c["card_name"] for c in result["drawn"]]
                self.log(f"{side.value} drew: {', '.join(names)}")
            else:
                self.log(f"{side.value}: No cards to draw")
            if result["over_limit"]:
                self.log(f"{side.value} hand has {result['hand_size']} cards - must discard to 6!")

        self._sync_hands()
        self.refresh()

    def _on_discard_card(self):
        """Discard selected card."""
        if self.ai_playing:
            self.log('Wait for AI to finish its turn')
            return
        from game_logic import discard_card_from_hand

        selection = self.hand_listbox.curselection()
        if not selection:
            self.log("Select a card to discard")
            return

        idx = selection[0]

        # Call game_logic to handle discard (all state changes there)
        result = discard_card_from_hand(self.state, self.current_side, idx)

        if result["success"]:
            self.log(f"Discarded: {result['card_name']}")
            # Sync GUI hands with game state
            self._sync_hands()

        else:
            self.log(f"Discard failed: {result.get('message', 'Unknown error')}")

        self.refresh()

    def _on_fsp_move(self):
        """Move the Fellowship with the selected Character / Will of the
        West die.

        R18-SRC-TOOLS (rows 31, 62; I1): the move is the ENGINE's
        fellowship_move row for the die, executed through
        _run_engine_action (ai._execute_action), and the Hunt it starts is
        the engine's - the Shadow's Flocks of Crebain choice (decider
        'sh'), the automatic re-rolls, the tile draw, the Wizard's Staff /
        Mithril / Axe and Bow / Horn choices, the learned damage and
        casualty choices (1.254), the Gollum-guide reveal rule (1.115),
        the reveal placement (fellowship_reveal_place) and the Mordor
        track; each open decision goes to its owner through
        _drive_engine_decisions (the AI side by search, a human by the
        required dialog).  The engine offers the row only while the
        Fellowship is Hidden (rulebook p.38) and below progress 12 / the
        last Mordor step.  The GUI's own Hunt copy is gone: it called
        move_fellowship with no Hidden check, asked the Shadow's choices
        as Yes/No boxes inside the human FP's action, ignored the Gollum
        guide, placed a cancelled reveal in the first region
        alphabetically and auto-sacrificed Mordor-track damage."""
        if self.ai_playing:
            self.log('Wait for AI to finish its turn')
            return
        if self._ai_mode_decision_open():       # P1-07 / R2-SL-32
            return
        valid = (ActionDieFace.CHARACTER, ActionDieFace.WILL_OF_WEST)
        if self.current_side != Side.FREE_PEOPLES:
            self.log("Only Free Peoples can move the Fellowship")
            return

        if not self._check_die_selected(valid):
            return

        acts = self._engine_die_menu(Side.FREE_PEOPLES,
                                     self.selected_die_idx,
                                     ('fellowship_move',))
        if acts is None:
            return
        if not acts:
            self.log(self._fellowship_move_refusal())
            return
        if len(acts) != 1:
            # one fellowship_move row per die: anything else is an engine
            # defect - refuse loudly, never pick one
            self.log(f"The engine served {len(acts)} Fellowship moves for "
                     f"this die - action not taken")
            return
        fs = self.state.fellowship
        mordor = fs.mordor_entered
        corruption0 = fs.corruption
        # R2-SL-32: _run_engine_action arms the AI reply only after every
        # decision of the Hunt resolution has returned
        if not self._run_engine_action(acts[0]):
            return
        where = (f"Mordor step {fs.mordor_step}/5" if mordor
                 else f"progress {fs.progress}")
        self.log(f"Fellowship moved: {where}, corruption {fs.corruption}"
                 f" ({fs.corruption - corruption0:+d}), "
                 f"{'hidden' if fs.hidden else 'REVEALED'}")

    def _fellowship_move_refusal(self) -> str:
        """Why the engine offers no fellowship_move row for the die."""
        fs = self.state.fellowship
        if not fs.hidden:
            return ("The Fellowship is revealed - it must be hidden again "
                    "before it can move (rulebook p.38)")
        if fs.mordor_entered and fs.mordor_step >= 5:
            return "The Fellowship is at the end of the Mordor track"
        if not fs.mordor_entered and fs.progress >= 12:
            return "The Fellowship cannot move further (progress 12)"
        return "The engine offers no Fellowship move with this die"

    def _check_shadow_movement_blocked(self, region: str) -> tuple:
        """
        Check if Shadow movement to a region is blocked by a persistent card.
        Returns (is_blocked, blocking_card_name).
        """
        from persistent_effects import can_shadow_move_to
        ok, blocker = can_shadow_move_to(self.state, region)
        return (not ok), blocker

    def _sync_hands(self):
        """Synchronize GUI hands with game state hands."""
        for side in [Side.FREE_PEOPLES, Side.SHADOW]:
            self.hands[side] = list(self.state.hands[side].hand)

    def _on_fsp_hide(self):
        """Hide the Fellowship with the selected die.

        R18-SRC-TOOLS (row 31; I1): the hide is the ENGINE's
        fellowship_hide row for the die (a Character / Will of the West
        die, or any die under the Strider Guide's ability), executed
        through _run_engine_action.  The engine offers it only while the
        Fellowship is revealed; the old button called hide_fellowship,
        ignored its refusal ('Fellowship already hidden') and still spent
        the die."""
        if self.ai_playing:
            self.log('Wait for AI to finish its turn')
            return
        if self._ai_mode_decision_open():       # P1-07 / R2-SL-32
            return
        if self.current_side != Side.FREE_PEOPLES:
            self.log("Only Free Peoples can hide the Fellowship")
            return

        # any rolled, unused die: the engine menu decides which may hide
        if not self._check_die_selected(tuple(ActionDieFace)):
            return

        acts = self._engine_die_menu(Side.FREE_PEOPLES,
                                     self.selected_die_idx,
                                     ('fellowship_hide',))
        if acts is None:
            return
        if not acts:
            if self.state.fellowship.hidden:
                self.log("The Fellowship is already hidden - no die spent")
            else:
                self.log("This die cannot hide the Fellowship (a Character "
                         "or Will of the West die; any die with Strider "
                         "as Guide)")
            return
        if len(acts) != 1:
            # one fellowship_hide row per die: anything else is an engine
            # defect - refuse loudly, never pick one
            self.log(f"The engine served {len(acts)} Fellowship hides for "
                     f"this die - action not taken")
            return
        # R2-SL-32: _run_engine_action arms the AI reply after the action
        self._run_engine_action(
            acts[0], "Fellowship hidden (Strider's ability - any die)"
            if acts[0].params.get('strider_ability') else "Fellowship hidden")

    def _on_move_nazgul(self):
        """Shadow Character die: fly Nazgul - the Witch-king included (he
        is a Nazgul for all rules) - and/or move the Mouth of Sauron.

        P1-10 (I1): the menu is the ENGINE's (get_legal_actions'
        nazgul_move / mouth_move rows for the selected die), every move
        executes through ai._execute_action, and further flights on the
        SAME die are the engine's '__nazgul_moves__' chain.  Destinations
        are the engine's: any region except an FP-controlled Stronghold
        not besieged by the Shadow (ruling 1.251: a Nazgul MAY end on an
        FP field army); the Mouth moves up to 3 regions companion-style
        (1.262).  The GUI no longer computes reachability or edits
        state.armies itself.
        """
        if self.ai_playing:
            self.log('Wait for AI to finish its turn')
            return
        if self.current_side != Side.SHADOW:
            self.log("Only Shadow can move Nazgûl")
            return

        # Check for character die
        valid = (ActionDieFace.CHARACTER,)
        if not self._check_die_selected(valid):
            return

        acts = self._engine_die_menu(Side.SHADOW, self.selected_die_idx,
                                     ('nazgul_move', 'mouth_move'))
        if acts is None:
            return
        if not acts:
            self.log("No Nazgûl, Witch-king or Mouth of Sauron can move "
                     "with this die")
            return
        groups, _done = self._figure_move_groups(acts)
        pick = self._pick_figure_move("Move Nazgûl", groups)
        if pick is None:
            self.log("Nazgûl move cancelled")
            return
        self._run_figure_moves(pick, "Moved")

    def _on_fsp_declare(self):
        """
        Declare the fellowship's position.
        This is a FREE action done at the START of the turn, BEFORE rolling dice.
        Does not cost a die.
        """
        if self.ai_playing:
            self.log('Wait for AI to finish its turn')
            return
        if self.current_side != Side.FREE_PEOPLES:
            self.log("Only Free Peoples can declare the Fellowship")
            return

        # Check that dice haven't been rolled yet
        fp_dice = self.state.dice.get(Side.FREE_PEOPLES, [])
        dice_rolled = any(d.current is not None for d in fp_dice)
        if dice_rolled:
            self.log("Cannot declare after rolling - must declare at start of turn before rolling")
            return

        # Declaring while revealed would be a free hide (engine menu gate:
        # _get_declare_actions returns [] when not hidden).
        if not self.state.fellowship.hidden:
            self.log("Revealed fellowship must spend a die to hide")
            return

        # R2-SL-33: one declaration per Fellowship phase, before its tail
        if getattr(self.state, '_fellowship_tail_phase', False):
            self.log("The Fellowship already declared this turn - "
                     "ROLL DICE finishes the Fellowship phase")
            return
        if self._fellowship_phase_closed():
            self.log("The Fellowship phase is over for this turn - "
                     "declare before the Hunt allocation and the roll")
            return

        # Progress 0 / declaring IN PLACE is legal RAW (Mordor entry from
        # a standstill at a gateway; healing by re-declaring in place).
        progress = self.state.fellowship.progress

        # Show destination dialog
        from game_logic import fellowship_region, is_shadow_region
        from board import regions as board_regions

        current = fellowship_region(self.state)
        self.log(f"Declare: Fellowship can travel up to {progress} region(s)")

        # Calculate reachable regions (can enter FP cities/strongholds when declaring)
        reachable = self._get_declare_reachable_regions(current, progress)

        if not reachable:
            self.log("No reachable regions for declaration")
            return

        # Show selection dialog
        dest = self._show_destination_dialog(
            "Declare Fellowship",
            f"Select destination (up to {progress} steps from {current}):\n"
            "(Can enter FP strongholds - heals 1 corruption)",
            sorted(reachable)
        )

        if not dest:
            self.log("Declaration cancelled")
            return

        declared = False
        try:
            # Route through the engine (fellowship_declare is a free
            # action, die_idx=-1) so the Mordor-entry check + rules match
            # self-play/teacher exactly.
            from ai import Action, _execute_action
            _execute_action(self.state, Action(
                die_idx=-1, action_type='fellowship_declare',
                params={'region': sanitize(dest)}))
            declared = True
            self.log(f"Fellowship declared in {dest}")
            meta = board_regions.get(sanitize(dest), {})
            is_settlement = meta.get("is_city") or meta.get("is_stronghold")
            if is_settlement and not is_shadow_region(dest):
                self.log(f"  Healed 1 corruption (now {self.state.fellowship.corruption})")
        except Exception as e:
            self.log(f"Declaration failed: {e}")

        # R2-SL-33: the declaration opened the engine's Fellowship tail
        # (_fellowship_tail_phase); drive it now - Mordor entry, the
        # declaration's effects, the Guide - never leave the bare flag
        if declared:
            self._legacy_fellowship_phase()
        self.refresh()

    def _get_declare_reachable_regions(self, start: str, max_steps: int) -> set:
        """
        Get regions reachable for fellowship declaration.
        Unlike reveal, declaration CAN end in FP cities/strongholds.
        Moving through shadow strongholds does NOT draw tiles when declaring.
        """
        from collections import deque

        start_san = sanitize(start)
        # Declaring IN PLACE (0 steps) is legal RAW - include the start.
        reachable = {start_san}
        visited = {start_san: 0}
        queue = deque([(start_san, 0)])

        while queue:
            current, steps = queue.popleft()

            if steps > 0:
                reachable.add(current)

            if steps >= max_steps:
                continue

            for neighbor in self.board.adjacent(current):
                neighbor_san = sanitize(neighbor)
                if neighbor_san not in visited or visited[neighbor_san] > steps + 1:
                    visited[neighbor_san] = steps + 1
                    queue.append((neighbor_san, steps + 1))

        return reachable

    def _show_destination_dialog(self, title: str, prompt: str, options: list) -> Optional[str]:
        """Show a dialog to select a destination region."""
        result = {"value": None}

        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.configure(bg=COLORS["panel_bg"])
        dialog.geometry("350x450")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text=prompt, bg=COLORS["panel_bg"], fg=COLORS["text"],
                wraplength=320).pack(pady=10)

        listbox = tk.Listbox(dialog, bg=COLORS["bg"], fg=COLORS["text"], height=15)
        for r in options:
            listbox.insert(tk.END, r)
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        def select():
            sel = listbox.curselection()
            if sel:
                result["value"] = listbox.get(sel[0])
            dialog.destroy()

        def cancel():
            dialog.destroy()

        btn_frame = tk.Frame(dialog, bg=COLORS["panel_bg"])
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        self._make_button(btn_frame, "Select", select, bg=COLORS["success"]).pack(side=tk.LEFT, padx=5)
        self._make_button(btn_frame, "Cancel", cancel).pack(side=tk.LEFT, padx=5)

        dialog.wait_window()
        return result["value"]

    def _on_allocate_eyes(self):
        """Allocate eyes BEFORE rolling (Shadow only, Phase 3)."""
        if self.current_side != Side.SHADOW:
            self.log("Only Shadow can allocate eyes")
            return

        if getattr(self.state, 'dice_rolled', False):
            self.log("Cannot allocate eyes after rolling - eyes are auto-allocated on roll")
            return

        # R2-RS-P1-09: the Shadow AI makes its own Hunt allocation at ROLL
        # DICE (a human one would stack on it), and there is one Hunt
        # allocation per turn (the engine eye menu does not subtract eyes
        # already allocated)
        if self.ai_side == Side.SHADOW:
            self.log("The Shadow AI allocates its own Hunt eyes at ROLL "
                     "DICE - no human allocation")
            return
        if getattr(self.state, 'eyes_allocated', 0) > 0:
            self.log(f"Eyes already allocated this turn "
                     f"({self.state.eyes_allocated}) - one Hunt allocation "
                     f"per turn")
            return

        # R2-SL-33: the Fellowship phase (an AI FP's declaration, the
        # tail, the Guide) precedes the Hunt allocation
        if not self._legacy_fellowship_phase():
            return

        # R2-RS-P1-09: the choices are the ENGINE's eye menu (companion
        # maximum, ruling 1.271 minimum), executed through _execute_action;
        # the old per-die tick list went straight to allocate_eyes (no
        # rule).  A decision still open is driven first, as at ROLL DICE.
        import traceback
        try:
            if self._engine_pending_side() is not None:
                self._drive_engine_decisions()
            self._human_eye_allocation(required=False)
        except Exception as e:
            self.log(f"Hunt allocation failed "
                     f"({type(e).__name__}: {e})")
            traceback.print_exc()
        self.refresh()

    def _on_recover_dice(self):
        """Recover all dice to unrolled state (for pre-roll eye allocation)."""
        if getattr(self.state, 'dice_rolled', False):
            self.log("Cannot recover after rolling - this is for pre-roll setup only")
            return

        from game_logic import recover_dice
        try:
            recover_dice(self.state)
            self.log("Dice recovered (ready for eye allocation)")
            self.refresh()
        except RuntimeError as e:
            self.log(f"Recovery failed: {e}")

    def _on_use_ring(self):
        """Use an Elven Ring to change a die face."""
        # R2-S4-ENGINE-AI-TOOLS (rescan v069): like every other human
        # action button - never while the AI plays, never inside an open
        # engine decision (P1-07 guard: it is driven first, then refused)
        if self.ai_playing:
            self.log("Wait for AI to finish its turn")
            return
        if self._ai_mode_decision_open():
            return
        # Check if current side has a ring
        if self.current_side == Side.FREE_PEOPLES:
            rings = self.state.fp_elven_rings
            used_this_turn = getattr(self.state, 'fp_ring_used_this_turn', False)
        else:
            rings = self.state.sh_elven_rings
            used_this_turn = getattr(self.state, 'sh_ring_used_this_turn', False)

        if rings <= 0:
            self.log(f"No Elven Rings available ({self.current_side.value})")
            return

        if used_this_turn:
            self.log("Already used an Elven Ring this turn")
            return

        if self.selected_die_idx is None:
            self.log("Select a die to change with ring")
            return

        die = self.state.dice[self.current_side][self.selected_die_idx]
        if die.used:
            self.log("Cannot change a used die")
            return
        if die.current is None:
            self.log("Cannot change an unrolled die")
            return

        # Shadow cannot change an Eye
        if self.current_side == Side.SHADOW and die.current == ActionDieFace.EYE:
            self.log("Shadow cannot change an Eye die with ring")
            return

        # Show dialog to select new face
        dialog = tk.Toplevel(self.root)
        dialog.title("Use Elven Ring")
        dialog.configure(bg=COLORS["panel_bg"])
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.geometry("300x350")

        tk.Label(
            dialog, text=f"Change {die.current.value} to:",
            bg=COLORS["panel_bg"], fg=COLORS["text"],
            font=("Arial", 12, "bold")
        ).pack(pady=10)

        # Available faces depend on side
        if self.current_side == Side.FREE_PEOPLES:
            # FP cannot change to Will of the West
            faces = [
                ActionDieFace.CHARACTER,
                ActionDieFace.MUSTER,
                ActionDieFace.EVENT,
                ActionDieFace.MUSTER_ARMY,
            ]
        else:
            # Shadow can change to any face including Eye
            faces = [
                ActionDieFace.CHARACTER,
                ActionDieFace.ARMY,
                ActionDieFace.MUSTER,
                ActionDieFace.EVENT,
                ActionDieFace.MUSTER_ARMY,
                ActionDieFace.EYE,
            ]

        selected_face = [None]

        for face in faces:
            def select(f=face):
                selected_face[0] = f
                dialog.destroy()

            self._make_button(dialog, face.value, select, width=20).pack(pady=3)

        tk.Label(
            dialog, text=f"\n({rings} ring(s) available)",
            bg=COLORS["panel_bg"], fg=COLORS["highlight"],
            font=("Arial", 9)
        ).pack()

        self._make_button(dialog, "Cancel", dialog.destroy).pack(pady=10)

        dialog.wait_window()

        if selected_face[0]:
            from game_logic import use_elven_ring
            try:
                result = use_elven_ring(
                    self.state, self.current_side,
                    self.selected_die_idx, selected_face[0]
                )
                self.log(f"Elven Ring: {result}")
                self.selected_die_idx = None
                self.refresh()
            except RuntimeError as e:
                self.log(f"Ring use failed: {e}")

    def _on_change_guide(self):
        """Answer the engine's open Guide choice.

        R18-SRC-TOOLS (row 31; I1): the Guide changes only through the
        engine's choose_guide rows - at the end of the Fellowship phase
        (turn_start's Guide step, which ROLL DICE and FSP Declare run) or
        when the Guide leaves the Fellowship (pending_guide_change).  The
        old button wrote fellowship.guide directly at any moment of any
        phase; it now drives an open Guide choice through
        _drive_engine_decisions and otherwise refuses, naming when the
        choice comes."""
        if self.ai_playing:
            self.log('Wait for AI to finish its turn')
            return
        if self.current_side != Side.FREE_PEOPLES:
            self.log("Only Free Peoples can change guide")
            return
        if getattr(self.state, 'pending_guide_change', None) is None:
            self.log("No Guide choice is open - the Guide is chosen at the "
                     "end of the Fellowship phase (ROLL DICE) or when the "
                     "Guide leaves the Fellowship")
            return
        try:
            self._drive_engine_decisions()
        except Exception as e:
            self.log(f"Could not resolve the Guide choice ({e})")
            import traceback
            traceback.print_exc()
        guide = self.state.fellowship.guide
        self.log(f"Guide: {getattr(guide, 'name', None)}")
        self.refresh()

    # R3-LEGACY-GUI-TURN: the _turn_end_blockers names End Turn settles
    # itself (the half-spent Army/Muster windows, closed below through
    # their engine rows); every other name is an open obligation.
    _END_TURN_WINDOW_BLOCKERS = ('pending_army_move_die', 'pending_muster_die')

    # An obligation forced to expire (_expire_turn_obligations) takes the
    # empty value ai._safe_next_turn's rollover gives it: None for the
    # object-valued ones _turn_end_blockers tests with 'is not None',
    # else the falsy value of its own type.
    _EXPIRE_TO_NONE = ('pending_combat', 'pending_hunt',
                       'pending_guide_change', 'pending_card_option',
                       'pending_card_followup', 'pending_discard',
                       '_wormtongue_attack_pending',
                       '_fellowship_move_die_pending',
                       '_gandalf_draw_pending', '_mithril_stash',
                       '_fellowship_route_event')

    def _on_end_turn(self):
        """End the current turn (hot-seat and --ai GUI).

        R3-LEGACY-GUI-TURN (R2-RS-v068-P1-04 follow-up): the rollover is
        ai._safe_next_turn behind ai._turn_end_blockers - the boundary the
        watch loop, engine_play and the self-play episode loop use - never
        a bare game_logic.next_turn, which in hot-seat rolled the turn over
        an open engine decision (it crossed into the next turn, e.g. a
        hand-limit discard grown by the turn-start draws).
          1. an open engine decision is driven through the one driver
             (human dialog / AI search) and this click is refused; the
             next End Turn continues;
          2. an obligation no engine decision answers (an engine defect: a
             queued draw, a decision whose menu is empty) refuses the
             rollover by name and ASKS - No (the default) keeps the turn
             open, Yes lets it expire with the turn, logged as a DEFECT
             (USER 2026-08-28 LOG -> ASK -> repair -> announce: never a
             refusal without an exit);
          3. a half-spent Army/Muster window closes through its engine
             row (F-2);
          4. R4-GUI-SKIP-DIE-BUTTON (rule 1.67; USER A2-06a/b, ruling
             1.276): a HUMAN seat's usable non-Event die refuses the
             rollover, naming its exits and offering the engine's rows
             that spend it (_refuse_end_turn_for_usable_dice); an AI
             seat's unplayed dice with a legal use (an Event die too:
             the AI's choice) ASK (_ask_forfeit_ai_dice: No keeps the
             turn and asks the AI again, Yes forfeits them as a
             DEFECT); every die left (an Event die, a die with no legal
             use) is discarded through the engine's die-discard row and
             named in the log (_end_turn_discard).  Skip Die
             (_on_skip_die) discards one such die mid-turn;
          5. next_turn's AUTO-boundary control heal, then
             ai._safe_next_turn; a terminal end-of-turn Victory Check keeps
             the turn and refresh() announces it."""
        import ai
        import combat
        import game_logic
        import traceback

        # R2-S4-ENGINE-AI-TOOLS (rescan v069): never while the AI plays
        if getattr(self, 'ai_playing', False):
            self.log("Wait for AI to finish its turn")
            return
        state = self.state

        # 1. never over an open engine decision (v069 in AI mode; now in
        # hot-seat too): driven through the one driver first, this click
        # refused.  (An open multi-step die window is not a decision:
        # closed in step 3.)
        if self._engine_pending_side() is not None:
            self.log("An open decision must be resolved first - action "
                     "not taken")
            try:
                self._drive_engine_decisions()
            except Exception as e:
                self.log(f"Could not resolve the open decision ({e})")
                traceback.print_exc()
            self.refresh()
            if self._engine_pending_side() is None:
                return
            # still open after the driver: no engine choice resolves it -
            # step 2 names it and asks

        # 2. obligations no engine decision answers
        owed = [b for b in self._end_turn_blockers()
                if b not in self._END_TURN_WINDOW_BLOCKERS
                and not b.startswith('unused_action_dice:')]
        if owed and not self._ask_expire_turn_obligations(owed):
            return

        # 3. R2-SL-34 / F-2 (USER 2026-07-19): an Army-die / Muster-die chain
        # dies with the turn (ai._safe_next_turn clears it; next_turn does
        # not) - closed through its owner's engine window-closing row, so
        # it never hands the leftover half to next turn's first action.
        for win in self._multistep_windows():
            try:
                self._close_multistep_window(*win, why='End Turn')
            except Exception as e:
                # F-2 holds even then: the chain is cleared field by field
                # exactly as ai._safe_next_turn does (no die spent) - loud,
                # never carried into the next turn
                self.log(f"DEFECT: could not close the open "
                         f"{self._multistep_text(win[0])} at End Turn "
                         f"({e}) - cleared without its closing row (F-2)")
                import traceback
                traceback.print_exc()
                pre = ('pending_army_move_' if win[0] == 'army_move'
                       else 'pending_muster_')
                for f in (('die', 'from', 'side', 'preexist')
                          if win[0] == 'army_move'
                          else ('die', 'first_type', 'region', 'side')):
                    setattr(self.state, pre + f, None)

        # 4. the unused action dice - the only blockers left once 1-3 have
        # run.  R4-GUI-SKIP-DIE-BUTTON (rule 1.67; USER A2-06a/b): nothing
        # is mutated until every side's dice are classified from its own
        # engine menu (an isolated-clone probe, _die_menu_probe).
        left = self._end_turn_blockers()
        stuck = [b for b in left if not b.startswith('unused_action_dice:')]
        if stuck:
            self.log(f"End Turn not taken - closing the die window left "
                     f"{', '.join(stuck)} open; press End Turn again")
            self.refresh()
            return
        reported = {b.split(':', 1)[1].split('=', 1)[0] for b in left}
        probes = {}         # side -> (menu, must_use, playable)
        for side in (Side.FREE_PEOPLES, Side.SHADOW):
            if side.name not in reported:
                continue
            try:
                probes[side] = self._die_menu_probe(side)
            except Exception as e:
                # an engine defect in a legal menu: loud, and End Turn
                # never wedges on it (as _end_turn_blockers)
                self.log(f"DEFECT: the {side.value} legal menu raised while "
                         f"End Turn checked rule 1.67 ({type(e).__name__}: "
                         f"{e}) - its dice are forfeited without it")
                traceback.print_exc()
        # a human seat's usable non-Event die refuses (rule 1.67); its own
        # Event dice are its End Turn's to discard (A2-06a)
        for side in (Side.FREE_PEOPLES, Side.SHADOW):
            if side == self.ai_side or side not in probes:
                continue
            menu, keep, _playable = probes[side]
            if keep and not self._refuse_end_turn_for_usable_dice(
                    side, menu, keep):
                return
        # an AI seat: every die with a legal use, an Event die too - using
        # or discarding it is the AI's choice, never the human's End Turn
        if self.ai_side in probes and probes[self.ai_side][2] and \
                not self._ask_forfeit_ai_dice(self.ai_side,
                                              probes[self.ai_side][2]):
            return
        for side in (Side.FREE_PEOPLES, Side.SHADOW):
            self._end_turn_discard(side, side in probes)

        # 5. the engine boundary (next_turn's AUTO-boundary control heal
        # first, as next_turn did: self-play only, never under the teacher)
        try:
            if combat.FINAL_HITS_HOOK is None:
                game_logic.enforce_settlement_control(state)
            terminal = ai._safe_next_turn(state)
        except Exception as e:
            self.log(f"End Turn FAILED ({type(e).__name__}: {e}) - the turn "
                     f"was not advanced cleanly; see the console")
            traceback.print_exc()
            self.refresh()
            return

        self.selected_die_idx = None
        self.pending_action = None
        self.action_source = None
        self._sync_hands()
        if terminal:
            # _settle_end_of_turn kept the turn; refresh() announces it
            self.log("End of turn: the Victory Check ends the game")
            self.refresh()
            return
        # the rollover drew for both sides; latch so Draw Cards refuses a
        # duplicate draw this turn.
        self._cards_drawn_turn = self.state.turn
        self.log(f"=== Turn {self.state.turn} ===")
        self.log("  Cards drawn for both sides")
        self.refresh()

    def _end_turn_blockers(self):
        """ai._turn_end_blockers for End Turn (R3-LEGACY-GUI-TURN).  The
        engine check returns the open obligation names first and probes
        the unused dice' legal menus only when there are none, so a raise
        can only come from that probe (an engine defect in a legal menu):
        it is logged as a DEFECT and reported as no blocker - the dice are
        then forfeited and the rollover's own check (every die spent, no
        probe) decides.  End Turn never wedges on it."""
        import ai
        import traceback
        try:
            return ai._turn_end_blockers(self.state)
        except Exception as e:
            self.log(f"DEFECT: the engine's turn-end check raised while "
                     f"probing the unused dice ({type(e).__name__}: {e}) - "
                     f"they are forfeited without it")
            traceback.print_exc()
            return []

    # ---- R4-GUI-SKIP-DIE-BUTTON: the engine's die-discard rows ----------
    # USER decisions A2-06a ("yes, players can discard event dice if they
    # wish" / "yes, any time") and A2-06b ("yes, those dice can be
    # discarded if no legal use"), ruling 1.276: an Event die may be
    # discarded for no effect at any time; a non-Event die only when it
    # has no legal use (one engine 'skip' row per distinct face,
    # ai._die_discard_rows); 1.67 stands for a usable non-Event die.
    _END_TURN_167_TITLE = 'End Turn refused - rule 1.67'
    _END_TURN_AI_DICE_TITLE = 'End Turn - AI dice unplayed'
    _SKIP_DIE_TITLE = 'Skip Die'

    def _die_menu_probe(self, side):
        """(menu, must_use, playable) for `side` from ONE engine menu built
        on an isolated clone (get_legal_actions; the read-only way
        ai._turn_end_blockers probes too):
          menu      the side's legal rows;
          must_use  [(index, die)] of its unused non-Event, non-Eye dice
                    whose face the menu gives no die-discard row - the
                    dice rule 1.67 says must be used (an Event die and a
                    die with no legal use carry a 'skip' row);
          playable  [(index, die)] of its unused non-Eye dice whose face
                    an ordinary row spends (ai._menu_spent_die_indices) -
                    every die with a legal use, an Event die included."""
        import ai
        menu = ai.get_legal_actions(ai.clone_state(self.state), side)
        skip_faces = {a.params.get('face') for a in menu
                      if a.action_type == 'skip'
                      and a.params.get('side') is not None}
        pool = self.state.dice.get(side, [])
        spent = {pool[i].current for i in ai._menu_spent_die_indices(menu)
                 if 0 <= i < len(pool)}
        unused = [(i, d) for i, d in enumerate(pool)
                  if not d.used and d.current is not None
                  and d.current != ActionDieFace.EYE]
        must_use = [(i, d) for i, d in unused
                    if d.current != ActionDieFace.EVENT
                    and d.current.name not in skip_faces]
        playable = [(i, d) for i, d in unused if d.current in spent]
        return menu, must_use, playable

    @staticmethod
    def _dice_text(dice):
        """'2 action dice (Army, Character)': the count and faces of
        [(index, die)]."""
        n = len(dice)
        return (f"{n} action {'die' if n == 1 else 'dice'} "
                f"({', '.join(d.current.value for _i, d in dice)})")

    def _refuse_end_turn_for_usable_dice(self, side, menu, dice) -> bool:
        """End Turn is refused while `side` (a HUMAN seat) still holds a
        usable non-Event die (rule 1.67; USER A2-06b: no discard row for
        it).  The refusal names its exits and offers the engine's own rows
        that spend those dice, so it always has one the human can take
        here even where no manual button builds that row (I1: the engine
        menu, run through _run_engine_action).  'No - keep the turn open'
        is the default; Cancel / the window close keep it open too.
        Engine-defect exit (USER 2026-08-28 LOG -> ASK -> kill the die ->
        announce): once an offered row has FAILED to execute this turn,
        the refusal also offers to forfeit those dice as a DEFECT, so a
        broken executor cannot wedge the turn.  True only for that
        consented forfeit (the rollover may proceed)."""
        import ai
        pool = self.state.dice.get(side, [])
        faces = {d.current for _i, d in dice}
        rows = [a for a in menu
                if isinstance(a.die_idx, int) and 0 <= a.die_idx < len(pool)
                and a.action_type not in ai._DIE_DISCARD_NONSPENDING
                and pool[a.die_idx].current in faces]
        what = self._dice_text(dice)
        self.log(f"End Turn refused - rule 1.67: {side.value} still hold "
                 f"{what} with a legal use, and a usable die must be used. "
                 f"Use it (the action buttons, or one of its engine-legal "
                 f"actions offered now); Skip Die discards only an Event "
                 f"die or a die with no legal use")
        prompt = (f"{side.value} still hold {what} with a legal use.  Rule "
                  f"1.67: a usable action die must be used - it cannot be "
                  f"discarded (Skip Die discards only an Event die or a die "
                  f"with no legal use).\n\nUse one now, or keep the turn "
                  f"open and play it with the action buttons.")
        opts = [("No - keep the turn open", None)]
        opts += [(f"Use the {pool[a.die_idx].current.value} die: "
                  f"{self._describe_action(a)}", k)
                 for k, a in enumerate(rows)]
        failed = (getattr(self, '_end_turn_row_failed', None)
                  == (self.state.turn, side))
        if failed:
            opts.append(("End the turn anyway - an engine action for this "
                         "die failed; forfeit it (logged as a DEFECT)",
                         'forfeit'))
        pick = self._show_choice_dialog(self._END_TURN_167_TITLE, prompt,
                                        opts, allow_cancel=True)
        if pick is None:
            self.log("End Turn not taken - the turn stays open")
            self.refresh()
            return False
        if pick == 'forfeit':
            for _i, d in dice:
                d.used = True
            self._end_turn_row_failed = None
            self.log(f"DEFECT: End Turn forced over the {side.value} "
                     f"{what} with a legal use - forfeited with the turn "
                     f"after an engine action for it failed (rule 1.67)")
            return True
        action = rows[pick]
        if not self._run_engine_action(
                action,
                log_msg=f"{side.value}: {self._describe_action(action)}"):
            # nothing else is open here, so False is an executor failure
            # (logged by _run_engine_action): the next End Turn offers the
            # DEFECT forfeit beside the rows
            self._end_turn_row_failed = (self.state.turn, side)
            self.log("End Turn not taken - that action failed; press End "
                     "Turn again to choose another, or to forfeit the die "
                     "as a DEFECT")
        return False

    def _ask_forfeit_ai_dice(self, side, dice) -> bool:
        """The AI seat `side` still holds dice with a legal use that it
        has not played (`dice`: _die_menu_probe's playable - an Event die
        too, since using or discarding it is the AI's choice; a failed
        search, or a reply not yet due).  Rule 1.67 forbids ending the
        turn over a usable die and a failing search is a bug (USER
        2026-10-23): no move is substituted.  LOG -> ASK (USER
        2026-08-28): No (the default; Cancel / the window close too)
        keeps the turn open and asks the AI to play them; Yes forfeits
        them with the turn, logged as a DEFECT - the refusal always has
        an exit.  True when the rollover may proceed."""
        what = self._dice_text(dice)
        self.log(f"End Turn refused - rule 1.67: the {side.value} AI still "
                 f"holds {what} with a legal use that it has not played")
        prompt = (f"The {side.value} AI still holds {what} with a legal "
                  f"use that it has not played (rule 1.67: a usable die "
                  f"must be used; using or discarding an Event die is the "
                  f"AI's choice).\n\nNo keeps the turn open and asks the "
                  f"AI to play them (Skip Die discards your own Event or "
                  f"unusable dice).  Yes ends the turn anyway: those dice "
                  f"are forfeited (logged as a DEFECT).")
        pick = self._show_choice_dialog(
            self._END_TURN_AI_DICE_TITLE, prompt,
            [("No - keep the turn open; the AI plays them", False),
             ("Yes - end the turn; the AI's usable dice are forfeited",
              True)],
            allow_cancel=True)
        if pick is not True:
            self.log(f"End Turn not taken - the turn stays open; the "
                     f"{side.value} AI is asked to play its dice")
            self._arm_ai_reply()
            self.refresh()
            return False
        for _i, d in dice:
            d.used = True
        self.log(f"DEFECT: End Turn forced over the {side.value} AI's "
                 f"unplayed {what} - forfeited with the turn (rule 1.67; "
                 f"the AI did not play them)")
        return True

    def _end_turn_discard(self, side, probed):
        """Step 4 of End Turn for one side, after the 1.67 checks: every
        unused die left may be discarded by the rules (an Event die,
        A2-06a; a die with no legal use, A2-06b) - each through the
        engine's own die-discard row (I1), the lot named in the log.
        `probed` False (the engine's dice probe raised, logged as a
        DEFECT, or reported nothing for this side): forfeited without
        the rows, as before.  A die no row discards (an executor refusal,
        an unused Eye) is forfeited loudly - End Turn never wedges here."""
        import ai
        import traceback
        pool = self.state.dice.get(side, [])
        discarded = []
        if probed:
            for _ in range(len(pool) + 1):
                try:
                    rows = [a for a in ai.get_legal_actions(
                                ai.clone_state(self.state), side)
                            if a.action_type == 'skip'
                            and a.params.get('side') is not None]
                    if not rows:
                        break
                    face = pool[rows[0].die_idx].current
                    ai._execute_action(self.state, rows[0])
                except Exception as e:
                    self.log(f"DEFECT: End Turn could not discard a "
                             f"{side.value} die through its engine row "
                             f"({type(e).__name__}: {e}) - forfeited "
                             f"without it")
                    traceback.print_exc()
                    break
                discarded.append(face)
        rest = [d for d in pool if not d.used and d.current is not None]
        for d in rest:
            d.used = True
        if discarded:
            n = len(discarded)
            self.log(f"End Turn: {side.value} forfeits {n} unused action "
                     f"{'die' if n == 1 else 'dice'} "
                     f"({', '.join(f.value for f in discarded)}) - "
                     f"discarded for no effect through the engine's "
                     f"die-discard row (an Event die, or no legal use)")
        if rest:
            n = len(rest)
            self.log(f"End Turn: {side.value} forfeits {n} unused action "
                     f"{'die' if n == 1 else 'dice'} "
                     f"({', '.join(d.current.value for d in rest)})"
                     + (" without an engine discard row" if probed
                        else ""))

    def _on_skip_die(self):
        """'Skip Die': discard one of the current side's action dice for
        no effect through the engine's own die-discard rows (USER A2-06a
        / A2-06b, ruling 1.276; ai._die_discard_rows): an Event die at any
        time; a non-Event die only when it has no legal use.  The
        selected die is discarded when its face has a row; with no die
        selected the rows are offered (Cancel = nothing discarded).  A
        usable non-Event die is refused by name (rule 1.67).  The exit
        End Turn's 1.67 refusal names; watch / replay / LIVE / mirror
        refuse it at the button (_make_button)."""
        from ai import Action
        import traceback
        if self.ai_playing:
            self.log("Wait for AI to finish its turn")
            return
        side = self.current_side
        if self.ai_side is not None and side == self.ai_side:
            self.log(f"Skip Die: the {side.value} dice are the AI's - "
                     f"nothing discarded")
            return
        if self._ai_mode_decision_open():        # P1-07 / R2-SL-34
            return
        if self._engine_pending_side() is not None:
            self.log("An open decision must be resolved first - action "
                     "not taken")
            return
        wins = self._multistep_windows()
        if wins:
            self.log(f"The open {self._multistep_text(wins[0][0])} must be "
                     f"made, or declined with Pass, first - action not "
                     f"taken")
            return
        try:
            menu, must_use, _playable = self._die_menu_probe(side)
        except Exception as e:
            self.log(f"Skip Die: the {side.value} legal menu raised "
                     f"({type(e).__name__}: {e}) - nothing discarded")
            traceback.print_exc()
            return
        rows = [a for a in menu if a.action_type == 'skip'
                and a.params.get('side') is not None]
        pool = self.state.dice.get(side, [])
        sel = self.selected_die_idx
        if sel is not None:
            die = pool[sel] if 0 <= sel < len(pool) else None
            if die is None or die.used or die.current is None:
                self.log("Skip Die: the selected die is not an unused "
                         "action die")
                return
            if die.current == ActionDieFace.EYE:
                self.log("Skip Die: an Eye die is never discarded")
                return
            match = next((a for a in rows
                          if a.params.get('face') == die.current.name), None)
            if match is None:
                self.log(f"Skip Die refused - rule 1.67: the {side.value} "
                         f"{die.current.value} die has a legal use and "
                         f"must be used; only an Event die or a die with "
                         f"no legal use may be discarded")
                return
            # the engine names the first unused die of a face; any die of
            # that face pays (the executor checks only the named face)
            action = Action(die_idx=sel, action_type='skip',
                            params=dict(match.params))
        else:
            if not rows:
                if must_use:
                    self.log(f"Skip Die: no {side.value} die can be "
                             f"discarded - rule 1.67: "
                             f"{self._dice_text(must_use)} with a legal "
                             f"use must be used")
                else:
                    self.log(f"Skip Die: {side.value} have no unused "
                             f"action die")
                return
            pick = self._show_choice_dialog(
                self._SKIP_DIE_TITLE,
                f"{side.value}: discard which die for no effect?  (An "
                f"Event die may be discarded at any time; any other die "
                f"only when it has no legal use - rule 1.67.)",
                [(self._describe_action(a), k) for k, a in enumerate(rows)],
                allow_cancel=True)
            if pick is None:
                self.log("Skip Die cancelled - nothing discarded")
                return
            action = rows[pick]
        self._run_engine_action(
            action, log_msg=f"{side.value}: {self._describe_action(action)}")

    def _ask_expire_turn_obligations(self, owed) -> bool:
        """R3-LEGACY-GUI-TURN: End Turn found obligations no engine
        decision answers (`owed`, _turn_end_blockers names).  LOG them,
        ASK (No is the default and keeps the turn open; the window close
        is No), and on Yes expire them (_expire_turn_obligations) with a
        DEFECT line.  True when the rollover may proceed."""
        names = ', '.join(owed)
        self.log(f"End Turn refused - open engine obligation(s) that no "
                 f"decision answers: {names}")
        hint = ""
        if (not getattr(self.state, 'dice_rolled', False)
                and '_fellowship_tail_phase' in owed):
            hint = " (ROLL DICE finishes the turn-start Fellowship phase)"
        prompt = (f"The turn cannot end: {names} is still open and no "
                  f"engine decision answers it{hint}.\n\nEnd the turn "
                  f"anyway?  Each listed obligation then expires with the "
                  f"turn (logged as a DEFECT).")
        pick = self._show_choice_dialog(
            "End Turn blocked", prompt,
            [("No - keep the turn open", False),
             ("Yes - end the turn; the obligation(s) expire with it",
              True)],
            allow_cancel=True)
        if pick is not True:
            self.log("End Turn not taken - the turn stays open")
            self.refresh()
            return False
        self._expire_turn_obligations(owed)
        self.log(f"DEFECT: End Turn forced over {names} - expired with the "
                 f"turn (no engine decision answered it)")
        return True

    def _expire_turn_obligations(self, owed):
        """Give each named obligation the empty value the rollover gives
        it (_EXPIRE_TO_NONE -> None, else the falsy value of its type);
        a name _turn_end_blockers still reports afterwards is set to None,
        so the consented rollover always proceeds."""
        st = self.state
        for name in owed:
            cur = getattr(st, name, None)
            if name in self._EXPIRE_TO_NONE or cur is None:
                new = None
            elif isinstance(cur, bool):
                new = False
            elif isinstance(cur, int):
                new = 0
            elif isinstance(cur, (list, tuple, dict, set)):
                new = type(cur)()
            else:
                new = None
            setattr(st, name, new)
        for name in set(self._end_turn_blockers()) & set(owed):
            setattr(st, name, None)

    def _on_switch_side(self):
        """Switch active side."""
        if self.ai_playing:
            self.log("Wait for AI to finish its turn")
            return

        if self.current_side == Side.FREE_PEOPLES:
            self.current_side = Side.SHADOW
        else:
            self.current_side = Side.FREE_PEOPLES

        self.selected_die_idx = None
        self.pending_action = None
        self.action_source = None

        self.log(f"Switched to {self.current_side.value}")
        self.refresh()

    # --- AI Integration ---

    @staticmethod
    def _current_net_paths():
        """Newest current-generation checkpoint pair: corpus_live (the
        v41 pair the live head runs) first, then the v41 chain — the
        old per-generation names die with each encoding generation, so
        this list must track the CURRENT one (v41 = brain everywhere,
        USER 2026-08-22)."""
        import os
        project_root = os.path.dirname(BASE_DIR)
        for sfx in ('live', 'v41jt', 'v41td', 'v41'):
            fp = os.path.join(project_root, 'wotr_fp_corpus_%s.pt' % sfx)
            sh = os.path.join(project_root, 'wotr_sh_corpus_%s.pt' % sfx)
            if os.path.exists(fp) and os.path.exists(sh):
                return fp, sh
        return None, None

    def enable_ai(self, side: Side):
        """Enable AI control for the given side."""
        from ai import load_dual_nets
        import os
        self.ai_side = side
        fp_path, sh_path = self._current_net_paths()
        dual = load_dual_nets(fp_path, sh_path) if fp_path else None
        if dual:
            self.fp_net, self.sh_net = dual
            self.ai_net = self.fp_net if side == Side.FREE_PEOPLES else self.sh_net
            self.log(f"AI enabled for {side.value} (dual nets: "
                     f"{os.path.basename(fp_path)}, {os.path.basename(sh_path)})")
        else:
            self.log("No current-generation nets found "
                     "(wotr_{fp,sh}_corpus_v31*.pt) - train first "
                     "(run_v31_pipeline).")
            self.ai_side = None

    # ------------------------------------------------------------------
    # Self-play (Watch AI vs AI)
    # ------------------------------------------------------------------

    def watch_selfplay(self):
        """Start AI-vs-AI self-play visualization."""
        from ai import load_dual_nets
        fp_path, sh_path = self._current_net_paths()
        dual = load_dual_nets(fp_path, sh_path) if fp_path else None
        if not dual:
            self.log("No current-generation nets found "
                     "(wotr_{fp,sh}_corpus_v31*.pt) - train first "
                     "(run_v31_pipeline).")
            return
        self.fp_net, self.sh_net = dual
        self.ai_net = self.fp_net  # legacy compat
        self.log(f"Watch mode: AI vs AI (dual nets: "
                 f"{os.path.basename(fp_path)}, {os.path.basename(sh_path)})")

        # Initialize decks and draw starting hands
        from game_logic import initialize_decks, draw_event_card
        initialize_decks(self.state)
        for side in [Side.FREE_PEOPLES, Side.SHADOW]:
            draw_event_card(self.state, side, 'character')
            draw_event_card(self.state, side, 'strategy')
        self._sync_hands()
        self.log("Starting hands drawn (1 character + 1 strategy each)")

        self.selfplay_running = True
        self.selfplay_turn = 0
        self.selfplay_paused = True  # Start in step-by-step mode
        self.selfplay_step_pending = False  # True when space pressed and waiting

        # Bind keyboard shortcuts for watch mode
        self.root.bind('<space>', lambda e: self._selfplay_step())
        self.root.bind('<Return>', lambda e: self._selfplay_toggle_auto())
        self.root.bind('<plus>', lambda e: self._selfplay_set_speed(-200))
        self.root.bind('<equal>', lambda e: self._selfplay_set_speed(-200))
        self.root.bind('<minus>', lambda e: self._selfplay_set_speed(200))

        self.log("[STEP MODE] Press Space for next action, Enter for auto-play")
        self._selfplay_next_turn()

    def _selfplay_turn_start(self):
        """Pre-roll half of a watch-mode turn (P1-06), phase for phase the
        twin of ai._run_single_episode_impl and engine_play's driver:
        hand-limit discards -> gateway auto-declare (the sanctioned
        self-play hardcode) -> declare phase (offered at progress >= 0:
        in-place declaration is legal) -> begin/resume_fellowship_tail
        (stationary Balrog, declaration Hunt effects, turn-start Mordor
        entry + refill, FINAL ruling 2026-08-16) -> Guide choice -> Shadow
        eye allocation on the engine's eye menu (companion maximum, ruling
        1.271 min-1 flag).  Every decision is searched under the engine's
        own phase flag, so its menu is the real one.  Returns False when
        the game ended or watch mode halted on a missing required choice.

        R3-LEGACY-GUI-TURN: the phases are THE shared turn-start driver
        (turn_start.steps, also run by engine_play and the legacy GUI);
        this method only answers its requests - the search budgets below
        (discard 50 sims; declare / eyes noise-free argmax; tail and Guide
        100 sims, temperature 0.1) and the watch log lines."""
        import ai
        import turn_start
        state = self.state

        def _label(side):
            return "FP" if side == Side.FREE_PEOPLES else "Shadow"

        def _search(side, sims=100, temperature=0.1, add_noise=True):
            return ai.mcts_action(state, side, self.fp_net, self.sh_net,
                                  num_simulations=sims,
                                  temperature=temperature,
                                  add_noise=add_noise)

        def _halt(why):
            self.log(f"Watch mode halted: {why}")
            self.selfplay_running = False
            self._sync_hands()
            self.refresh()
            return False

        _what = {'discard': 'hand-limit discard', 'decision': 'decision',
                 'declare': 'declaration', 'fellowship_tail':
                 'Fellowship effect', 'guide': 'Guide choice',
                 'eye': 'Hunt allocation'}

        def _choose(req):
            if req.kind == 'discard':
                return _search(req.side, sims=50)
            if req.kind in ('declare', 'eye'):
                return _search(req.side, temperature=0.0, add_noise=False)
            return _search(req.side)

        def _execute(req):
            act = req.action
            if req.kind == 'discard':
                self.log(f"{_label(req.side)} [hand limit]: discards "
                         f"{act.params.get('card_id', '?')}")
            elif req.kind == 'declare':
                self.log(f"FP [Fellowship Phase]: Declares fellowship at "
                         f"{act.params.get('region', '?')}", "fp")
            elif req.kind == 'guide':
                self.log(f"FP [Fellowship Phase]: Guide "
                         f"{act.params.get('companion', '?')}", "fp")
            elif req.kind != 'eye':
                self.log(f"{_label(req.side)} [Fellowship Phase]: "
                         f"{act.action_type} {act.params}")
            try:
                ai._execute_action(state, act)
            except Exception as e:
                _halt(f"{_label(req.side)} {_what.get(req.kind, req.kind)} "
                      f"failed ({type(e).__name__}: {e})")
                raise

        def _event(ev):
            if ev.kind == 'auto_declared':
                self.log(f"FP [Fellowship Phase]: Declares fellowship at "
                         f"{ev.detail['region']} (Mordor gateway)", "fp")
            elif ev.kind == 'game_over':
                result = ev.detail['result']
                self.log(f"\n*** GAME OVER: {result['victory']} wins by "
                         f"{result.get('victory_type', '')}! ***")
            elif ev.kind == 'mordor_entered':
                self.log("FP [Fellowship Phase]: Fellowship enters Mordor",
                         "fp")

        try:
            outcome = turn_start.run(turn_start.steps(state), _choose,
                                     _execute, _event)
        except turn_start.TurnStartError as e:
            return _halt(str(e))
        if outcome == 'game_over':
            self.selfplay_running = False
            self._sync_hands()
            self.refresh()
            return False
        return True

    def _fellowship_phase_closed(self) -> bool:
        """True once this turn's Fellowship-phase tail has run in the
        hot-seat / --ai GUI (R2-SL-33): the declaration opportunity is
        over, and begin_fellowship_tail must not run again this turn
        (that would be a second stationary-Balrog opportunity and a
        re-opened Guide choice)."""
        done = getattr(self, '_fellowship_phase_done', None)
        return (done is not None and done[0] is self.state
                and done[1] == self.state.turn)

    def _legacy_fellowship_phase(self) -> bool:
        """Turn-start Fellowship phase for the hot-seat and --ai GUI
        (R2-SL-33), the twin of _selfplay_turn_start (P1-06), engine_play's
        driver and the episode loop: open engine decisions (the hand-limit
        discards opened by the turn-start draws) -> an AI FP's gateway
        auto-declare or declare search under _declare_phase (a human FP
        declares with FSP Declare, which runs this phase straight after)
        -> begin/resume_fellowship_tail (stationary Balrog, declaration
        effects, turn-start Mordor entry + refill, FINAL ruling
        2026-08-16), every open choice to its owner (human dialog or AI
        search) -> the end-of-phase Guide choice -> _eye_allocation_phase
        cleared (the AI Shadow then allocates in _ai_allocate_eyes, a human
        Shadow with Allocate Eyes).  The tail runs once per turn.  Returns
        True when the Hunt allocation and the roll may follow; False when
        the game ended or a required choice failed (logged, nothing
        rolled).  Engine flags are never cleared silently, so the next
        ROLL DICE resumes from the decision still open.

        R3-LEGACY-GUI-TURN: the phases are THE shared turn-start driver
        (turn_start.steps, also run by engine_play and watch mode), with
        eyes=False (the Hunt allocation follows in
        _pre_roll_hunt_allocation / Alloc Eyes) and fellowship=False once
        this turn's tail has run.  This method answers its requests: open
        decisions and Fellowship effects through _choose_engine_decision
        (the one driver's pick), an AI FP's declaration and the Guide by
        search or the human's dialog."""
        import ai
        import traceback
        import turn_start
        state = self.state
        if getattr(state, 'dice_rolled', False):
            return True
        fp_side = Side.FREE_PEOPLES
        fp_net = getattr(self, 'fp_net', None) or self.ai_net
        sh_net = getattr(self, 'sh_net', None) or self.ai_net

        def _label(side):
            return "Free Peoples" if side == fp_side else "Shadow"

        def _is_ai(side):
            return (self.ai_side is not None and side == self.ai_side
                    and self.ai_net is not None)

        def _ask(side, acts, what, temperature=0.1, add_noise=False):
            if len(acts) == 1:
                return acts[0]
            if _is_ai(side):
                act = ai.mcts_action(state, side, fp_net, sh_net,
                                     num_simulations=100,
                                     temperature=temperature,
                                     add_noise=add_noise)
                if act is None or act not in acts:
                    raise RuntimeError(f"{_label(side)} AI {what} has no "
                                       f"legal selection (got {act})")
                return act
            opts = [(self._describe_action(a), i)
                    for i, a in enumerate(acts)]
            pick = self._show_choice_dialog(f"{_label(side)} decision",
                                            f"{what}:", opts)
            if pick is None:
                raise RuntimeError(f"{_label(side)} {what} dialog closed "
                                   f"without a choice - no action taken, "
                                   f"the decision is still open")
            return acts[pick]

        def _choose(req):
            if req.kind == 'declare':
                # 1.142c noise-free argmax
                return _ask(req.side, req.actions, "Fellowship declaration",
                            temperature=0.0)
            if req.kind == 'guide':
                return _ask(req.side, req.actions,
                            "Guide at the end of the Fellowship phase")
            return self._choose_engine_decision(req.side, req.actions)

        def _execute(req):
            act = req.action
            if req.kind == 'declare':
                self.log(f"FP AI [Fellowship Phase]: declares at "
                         f"{act.params.get('region', '?')}", "fp")
            elif req.kind == 'guide':
                self.log(f"FP [Fellowship Phase]: Guide "
                         f"{act.params.get('companion', '?')}", "fp")
            ai._execute_action(state, act)

        def _event(ev):
            if ev.kind == 'auto_declared':
                self.log(f"FP AI [Fellowship Phase]: declares at "
                         f"{ev.detail['region']} (Mordor gateway)", "fp")
            elif ev.kind == 'fellowship_tail_done':
                # the tail runs once per turn (no second stationary-Balrog
                # opportunity, no re-opened Guide choice)
                self._fellowship_phase_done = (state, state.turn)
            elif ev.kind == 'game_over':
                result = ev.detail['result']
                self.log(f"Game over in the Fellowship phase "
                         f"({result.get('victory')} by "
                         f"{result.get('victory_type', '')}) - dice not "
                         f"rolled")
            elif ev.kind == 'mordor_entered':
                self.log("FP [Fellowship Phase]: the Fellowship enters "
                         "Mordor", "fp")

        try:
            phase = not self._fellowship_phase_closed()
            if (phase and getattr(state, '_eye_allocation_phase', False)
                    and not getattr(state, '_fellowship_tail_phase', False)):
                # left by an earlier turn whose phase ended without its
                # roll; begin_fellowship_tail would skip this turn's tail
                # behind it
                self.log("Fellowship phase: clearing a Hunt-allocation "
                         "flag left open by an earlier turn")
                state._eye_allocation_phase = False
            outcome = turn_start.run(
                turn_start.steps(state, auto_declare=_is_ai(fp_side),
                                 declare=_is_ai(fp_side), fellowship=phase,
                                 eyes=False, fallback=self.current_side),
                _choose, _execute, _event)
            self._sync_hands()
            if outcome == 'game_over':
                self.refresh()
                return False
            return True
        except Exception as e:
            self.log(f"Fellowship phase not finished - dice not rolled "
                     f"({type(e).__name__}: {e})")
            traceback.print_exc()
            self.refresh()
            return False

    def _selfplay_next_turn(self):
        """Start a new turn: Fellowship phase, eye allocation, dice roll,
        then actions."""
        if not self.selfplay_running:
            return
        from ai import _safe_next_turn
        import game_logic

        if self.selfplay_turn > 0:
            if _safe_next_turn(self.state):
                result = game_logic.check_all_victory_conditions(self.state)
                self.log(f"\n*** GAME OVER: {result['victory']} wins by "
                         f"{result.get('victory_type', '')}! ***")
                self.selfplay_running = False
                self._sync_hands()
                self.refresh()
                return

        self.selfplay_turn += 1
        if self.selfplay_turn > 25:
            self.log("Game reached turn limit (25)")
            self.selfplay_running = False
            return

        self.log(f"\n{'='*40}", "turn")
        self.log(f"=== TURN {self.selfplay_turn} ===", "turn")
        self.log(f"{'='*40}", "turn")

        # P1-06: Fellowship phase (discards, auto-declare, declare, tail
        # with turn-start Mordor entry, Guide) + eye allocation, exactly as
        # the self-play episode loop runs them
        if not self._selfplay_turn_start():
            return
        eyes_allocated = getattr(self.state, 'eyes_allocated', 0)
        eyes_in_box = len(self.state.hunt_box)

        # Roll dice
        game_logic.roll_action_dice(self.state, pool=Side.FREE_PEOPLES)
        game_logic.roll_action_dice(self.state, pool=Side.SHADOW)

        fp_dice = [d.current.value for d in self.state.dice[Side.FREE_PEOPLES]
                   if d.current and not d.used]
        sh_dice = [d.current.value for d in self.state.dice[Side.SHADOW]
                   if d.current and not d.used]
        self.log(f"FP dice: {', '.join(fp_dice)}", "die")
        self.log(f"Shadow dice: {', '.join(sh_dice)}", "die")
        if eyes_allocated > 0:
            self.log(f"Eyes allocated: {eyes_allocated} -> hunt box ({eyes_in_box} total)", "die")

        self.selfplay_active_side = Side.FREE_PEOPLES
        self.selfplay_turn_owner = None
        self.refresh()
        # In auto mode, schedule first action; in step mode, wait for space
        if not self.selfplay_paused:
            self.root.after(self.selfplay_speed, self._selfplay_one_action)

    def _selfplay_one_action(self):
        """Play one AI action, refresh, check victory, schedule next."""
        if not self.selfplay_running:
            return
        # In step mode, only proceed if a step was requested
        if self.selfplay_paused and not self.selfplay_step_pending:
            return
        self.selfplay_step_pending = False

        from ai import (get_legal_actions, _execute_action, mcts_action,
                        _pending_decision_side, _turn_owner_after,
                        search_turn_owner)
        import game_logic

        # P1-05: an open engine pending decision (card option / followup,
        # discard, hunt-reveal placement, combat, hunt, guide...) belongs
        # to ONE side and costs no die: route it to its owner through the
        # shared cascade BEFORE any dice check, exactly like
        # _run_single_episode (pre-ask routing).  Plain alternation asked
        # the non-owner (RuntimeError from get_legal_actions) or let the
        # Shadow pick the FP's reveal placement.
        pending_side = _pending_decision_side(
            self.state, fallback=self.selfplay_active_side)
        if pending_side is not None:
            self.selfplay_active_side = pending_side

        active = self.selfplay_active_side
        other = Side.SHADOW if active == Side.FREE_PEOPLES else Side.FREE_PEOPLES

        fp_left = self._count_unused_dice(Side.FREE_PEOPLES)
        sh_left = self._count_unused_dice(Side.SHADOW)
        my_left = self._count_unused_dice(active)

        if pending_side is None:
            # If both sides out of dice, next turn
            if fp_left == 0 and sh_left == 0:
                # _selfplay_next_turn settles inactivity before military victory.
                if self.selfplay_paused:
                    # In step mode, go to next turn immediately on this step
                    self._selfplay_next_turn()
                else:
                    self.root.after(self.selfplay_speed, self._selfplay_next_turn)
                return

            # Skip side if no dice left (transparent, don't consume a step)
            if my_left == 0:
                self.selfplay_active_side = other
                self.root.after(50, self._selfplay_one_action)
                return

        actions = get_legal_actions(self.state, active)
        if not actions:
            if pending_side is not None:
                # the owner MUST resolve its pending; handing the turn over
                # would just route back here (silent 50 ms spin) - halt loudly
                self.log(f"Watch mode halted: {active.value} owns an open "
                         f"pending decision but has no legal action")
                self.selfplay_running = False
                return
            self.selfplay_active_side = other
            self.root.after(50, self._selfplay_one_action)
            return

        # Use MCTS for strongest play (200 sims); C02: the search knows
        # whose action an answered root pending belongs to
        _owner9 = getattr(self, 'selfplay_turn_owner', None)
        with search_turn_owner(_owner9):
            action = mcts_action(self.state, active, self.fp_net, self.sh_net,
                                 num_simulations=200, temperature=0.1)
        if action is None:
            if pending_side is not None:
                self.log(f"Watch mode halted: search returned no action for "
                         f"{active.value}'s open pending decision")
                self.selfplay_running = False
                return
            self.selfplay_active_side = other
            self.root.after(50, self._selfplay_one_action)
            return

        # Pass guard (mirrors _run_single_episode's pass gate)
        opp_left = self._count_unused_dice(other)
        # P1-05: an engine pending decision routed here is never rewritten
        is_pending = (pending_side is not None or
                      self.state.pending_army_move_die is not None or
                      self.state.pending_muster_die is not None)
        if (action.action_type == 'pass' and not is_pending
                and opp_left <= my_left):
            non_pass = [a for a in actions if a.action_type != 'pass']
            if non_pass:
                action = non_pass[0]

        side_label = "FP" if active == Side.FREE_PEOPLES else "Shadow"

        # Capture die face before execution (die gets marked used)
        die_label = "free"
        if action.die_idx >= 0:
            dice_list = self.state.dice.get(active, [])
            if action.die_idx < len(dice_list):
                d = dice_list[action.die_idx]
                if d.current:
                    die_label = d.current.value
        try:
            _desc9 = self._describe_action(action)
        except Exception:
            _desc9 = str(getattr(action, 'action_type', action))
        self.log(f"{side_label} [{die_label}]: {_desc9}")

        # Capture pre-action state for hunt logging
        _pre_corr = self.state.fellowship.corruption
        _pre_prog = self.state.fellowship.progress
        _pre_companions = [c.name for c in self.state.fellowship.companions]

        # C02: the turn owner is read off the PRE-action state (the burst's
        # combat / hunt / guide answers below keep it)
        _owner9 = _turn_owner_after(self.state, active, _owner9)
        self.selfplay_turn_owner = _owner9
        try:
            _execute_action(self.state, action)
        except Exception as e:
            # P1-12: the engine refused a searched legal action - an
            # engine/search bug.  Halt loudly; carrying on (no die spent,
            # the side bounces back) re-picks the refused action forever.
            import traceback
            traceback.print_exc()
            self.log(f"{side_label}: Action failed ({e}) - watch mode "
                     f"halted")
            self.selfplay_running = False
            self._sync_hands()
            self.refresh()
            return

        # Log hunt results after fellowship move
        if action.action_type == 'fellowship_move':
            _post_corr = self.state.fellowship.corruption
            _post_companions = [c.name for c in self.state.fellowship.companions]
            _lost = [c for c in _pre_companions if c not in _post_companions]
            _dmg = _post_corr - _pre_corr
            _pending = getattr(self.state, '_pending_hunt_damage', 0)
            _ph = self.state.pending_hunt
            hunt_box_size = len(self.state.hunt_box)
            if _dmg > 0 or _lost or _pending > 0 or _ph:
                self.log(f"  Hunt: {hunt_box_size} dice in box, "
                         f"damage={_dmg + _pending}, corruption={_post_corr}/12", "hunt")
                if _lost:
                    self.log(f"  Companion lost: {', '.join(_lost)}", "hunt")
            else:
                self.log(f"  Hunt: {hunt_box_size} dice in box, no damage", "hunt")

        # Log combat details if a battle round just happened
        last_result = getattr(self.state, '_last_battle_result', None)
        if last_result:
            cards = getattr(self.state, '_last_combat_cards', (None, None))
            atk_card, def_card = cards if cards else (None, None)
            pc = getattr(self.state, 'pending_combat', None) or {}
            atk_r = pc.get('attacker_region', '?')
            def_r = pc.get('defender_region', '?')
            self.log(f"  *** COMBAT: {atk_r} -> {def_r} ***", "combat")
            if atk_card:
                self.log(f"    Attacker plays: {atk_card}", "combat")
            if def_card:
                self.log(f"    Defender plays: {def_card}", "combat")
            atk_cas = len(getattr(last_result, 'attacker_casualties', []))
            def_cas = len(getattr(last_result, 'defender_casualties', []))
            self.log(f"    Casualties: ATK -{atk_cas}, DEF -{def_cas}", "casualty")
            if getattr(last_result, 'attacker_destroyed', False):
                self.log(f"    Attacker army destroyed!", "casualty")
            if getattr(last_result, 'defender_destroyed', False):
                self.log(f"    Defender army destroyed!", "casualty")
            self.state._last_battle_result = None

        self._sync_hands()
        self.refresh()

        # Check victory. TIMING (rulebook p.17, ruling 1.212a): ring
        # endings are immediate mid-action; military victory is resolved
        # ONLY in the end-of-turn Victory Check phase (see turn boundary in
        # _selfplay_one_action).
        result = game_logic.check_all_victory_conditions(self.state)
        if game_logic.should_game_end(result, end_of_turn=False):
            winner = result['victory']
            vtype = result.get('victory_type', '')
            self.log(f"\n*** GAME OVER: {winner} wins by {vtype}! ***")
            self.selfplay_running = False
            return

        # Burst-resolve pending multi-phase states (combat, hunt, guide)
        # so they complete without visual confusion from side alternation
        burst_steps = 0
        while burst_steps < 20:
            pc = getattr(self.state, 'pending_combat', None)
            ph = getattr(self.state, 'pending_hunt', None)
            pg = getattr(self.state, 'pending_guide_change', None)
            if pc is None and ph is None and pg is None:
                break
            burst_steps += 1
            # Determine deciding side
            if pc is not None:
                burst_side = pc['deciding_side']
            elif ph is not None:
                burst_side = Side.FREE_PEOPLES
            elif pg is not None:
                burst_side = Side.FREE_PEOPLES
            else:
                break
            self.selfplay_active_side = burst_side
            burst_actions = get_legal_actions(self.state, burst_side)
            if not burst_actions:
                break
            with search_turn_owner(_owner9):
                burst_action = mcts_action(self.state, burst_side,
                                           self.fp_net, self.sh_net,
                                           num_simulations=100, temperature=0.1)
            bl = "FP" if burst_side == Side.FREE_PEOPLES else "Shadow"
            if burst_action is None:
                # P1-12: never substitute burst_actions[0] - nobody chose
                # it.  Halt loudly; the pending stays open (I1).
                self.log(f"Watch mode halted: search returned no action "
                         f"for {bl}'s open pending decision")
                self.selfplay_running = False
                self._sync_hands()
                self.refresh()
                return
            self.log(f"  {bl} [free]: {self._describe_action(burst_action)}")
            try:
                _execute_action(self.state, burst_action)
            except Exception as e:
                # P1-12: an engine refusal is an engine/search bug - halt
                # loudly and leave the battle / hunt / guide change exactly
                # as the engine left it (never force-clear a pending).
                import traceback
                traceback.print_exc()
                self.log(f"  {bl}: Burst action failed ({e}) - watch mode "
                         f"halted")
                self.selfplay_running = False
                self._sync_hands()
                self.refresh()
                return
            # Log combat details if a round just happened
            last_result = getattr(self.state, '_last_battle_result', None)
            if last_result:
                cards = getattr(self.state, '_last_combat_cards', (None, None))
                atk_card, def_card = cards if cards else (None, None)
                bpc = getattr(self.state, 'pending_combat', None) or {}
                self.log(f"  *** COMBAT round ***", "combat")
                if atk_card:
                    self.log(f"    Attacker plays: {atk_card}", "combat")
                if def_card:
                    self.log(f"    Defender plays: {def_card}", "combat")
                atk_cas = len(getattr(last_result, 'attacker_casualties', []))
                def_cas = len(getattr(last_result, 'defender_casualties', []))
                self.log(f"    Casualties: ATK -{atk_cas}, DEF -{def_cas}", "casualty")
                if getattr(last_result, 'attacker_destroyed', False):
                    self.log(f"    Attacker army destroyed!", "casualty")
                if getattr(last_result, 'defender_destroyed', False):
                    self.log(f"    Defender army destroyed!", "casualty")
                self.state._last_battle_result = None
            # Check victory mid-burst. Ring endings are immediate; military
            # victory waits for the end-of-turn check (ruling 1.212a).
            vr = game_logic.check_all_victory_conditions(self.state)
            if game_logic.should_game_end(vr, end_of_turn=False):
                self.log(f"\n*** GAME OVER: {vr['victory']} wins by {vr.get('victory_type', '')}! ***")
                self.selfplay_running = False
                self._sync_hands()
                self.refresh()
                return
        self._sync_hands()
        self.refresh()

        # P1-05: post-action routing IS _run_single_episode's (single
        # source of truth, never plain alternation): ai._active_after_action
        # - ring use keeps the seat (the burst may have re-seated
        # selfplay_active_side), a multi-step die action its owner, an open
        # pending goes to its decider, else alternation from the TURN OWNER
        # (C02: never from the side that answered a cross-side decision; a
        # just-resolved battle passes control to the non-attacker).
        from ai import _active_after_action
        self.selfplay_active_side = _active_after_action(self.state, action, active,
                                                         _owner9)

        # In auto mode, schedule next; in step mode, wait for space
        if not self.selfplay_paused:
            self.root.after(self.selfplay_speed, self._selfplay_one_action)

    def _selfplay_step(self):
        """Advance exactly one action (step mode), or pause if auto-playing."""
        if not self.selfplay_running:
            return
        if not self.selfplay_paused:
            # Currently auto-playing: pause
            self.selfplay_paused = True
            self.log("[STEP MODE] Press Space for next action, Enter for auto-play")
            return
        # Step mode: trigger one action
        self.selfplay_step_pending = True
        self._selfplay_one_action()

    def _selfplay_toggle_auto(self):
        """Toggle between step mode and auto-play mode (Enter key)."""
        if not self.selfplay_running:
            return
        if self.selfplay_paused:
            # Switch to auto-play
            self.selfplay_paused = False
            self.log(f"[AUTO-PLAY] {self.selfplay_speed}ms/action. "
                     f"Space to pause, +/- to adjust speed")
            self.root.after(100, self._selfplay_one_action)
        else:
            # Switch back to step mode
            self.selfplay_paused = True
            self.log("[STEP MODE] Press Space for next action, Enter for auto-play")

    def _selfplay_set_speed(self, delta):
        """Adjust self-play speed (ms between actions). Lower = faster."""
        self.selfplay_speed = max(100, min(3000, self.selfplay_speed + delta))
        self.log(f"Speed: {self.selfplay_speed}ms per action")

    def _ai_allocate_eyes(self):
        """AI Shadow pre-roll Hunt allocation (P1-09) on the ENGINE's eye
        menu, the twin of the episode loop / watch-mode turn start: the
        search runs under the engine's own _eye_allocation_phase flag, so
        its menu is get_legal_actions' (0..min(max(companions, 1), unused),
        ruling 1.271 mandatory 1-eye minimum after an FP Hunt Box
        retrieval), and the pick executes through ai._execute_action (I1).
        1.142: noise-free argmax.  Raises when no legal selection comes
        back, so the dice are never rolled on a skipped allocation.
        R2-RS-P1-09: one Hunt allocation per turn (the engine eye menu does
        not subtract eyes already allocated), and refuses by name while
        another engine decision is open - get_legal_actions serves that
        decision AHEAD of the eye menu, so the search would be handed its
        menu (callers drive it first: _pre_roll_hunt_allocation)."""
        import ai
        state = self.state
        if not any(not d.used for d in state.dice.get(Side.SHADOW, [])):
            return
        if getattr(state, 'eyes_allocated', 0) > 0:
            self.log(f"Shadow AI: Hunt allocation already made this turn "
                     f"({state.eyes_allocated} eye(s))")
            return
        kind, owner = ai._pending_decision_kind_side(state, self.current_side)
        if kind is not None and kind != 'eye':
            raise RuntimeError(
                f"an open engine decision ({kind}, "
                f"{owner.value if owner is not None else '?'}) must be "
                f"resolved before the Shadow AI Hunt allocation")
        fp_net = getattr(self, 'fp_net', None) or self.ai_net
        sh_net = getattr(self, 'sh_net', None) or self.ai_net
        state._eye_allocation_phase = True
        try:
            menu = [a.params.get('count') for a in
                    ai.get_legal_actions(state, Side.SHADOW)
                    if a.action_type == 'eye_allocate']
            eye = ai.mcts_action(state, Side.SHADOW, fp_net, sh_net,
                                 num_simulations=200, temperature=0.0,
                                 add_noise=False)
        finally:
            state._eye_allocation_phase = False
        if (eye is None or eye.action_type != 'eye_allocate'
                or eye.params.get('count') not in menu):
            raise RuntimeError(
                f"Shadow AI Hunt allocation has no legal selection "
                f"(menu {menu}, got {eye})")
        ai._execute_action(state, eye)
        n_eyes = eye.params.get('count', 0)
        self.log(f"Shadow AI: Pre-allocated {n_eyes} die(s) as eyes "
                 f"to hunt box")

    def _pre_roll_hunt_allocation(self) -> bool:
        """R2-RS-P1-09: ROLL DICE's step between the turn-start Fellowship
        phase and the roll (hot-seat / --ai GUI).
        1) Every engine decision still open is driven through the one
           driver first (the AI searches its own, the human is asked for
           theirs).  The Fellowship phase drives the turn-start ones (the
           hand-limit discards); this also covers one opened after that
           phase closed.  get_legal_actions serves an open decision AHEAD
           of the eye menu, so an eye search on top of it is handed that
           decision's menu (the AI Shadow's own discard wedged ROLL DICE).
        2) The Shadow's Hunt allocation on the engine eye menu: the AI
           Shadow searches it (_ai_allocate_eyes); a human Shadow
           allocates with Alloc Eyes and is asked here (required) only
           when the ruling 1.271 minimum is owed and nothing is allocated
           - with no minimum, rolling unallocated is the legal 0-eye pick.
        True when the roll may follow.  Every failure is logged in the GUI
        and returns False with nothing rolled - never raised into the Tk
        callback; the next ROLL DICE retries from the decision still
        open."""
        import traceback
        what = "an open decision"
        try:
            if self._engine_pending_side() is not None:
                self._drive_engine_decisions()
            if self._engine_pending_side() is not None:
                raise RuntimeError("an open engine decision has no legal "
                                   "choice")
            what = "the Hunt allocation"
            if self.ai_side == Side.SHADOW:
                if self.ai_net:
                    self._ai_allocate_eyes()
            elif getattr(self.state, 'eyes_allocated', 0) == 0:
                acts = self._engine_eye_menu()
                if acts and min(a.params.get('count', 0)
                                for a in acts) >= 1:
                    self._human_eye_allocation(required=True)
        except Exception as e:
            self.log(f"Roll refused - {what} is not finished, dice not "
                     f"rolled ({type(e).__name__}: {e})")
            traceback.print_exc()
            self.refresh()
            return False
        return True

    def _engine_eye_menu(self) -> list:
        """The engine's Shadow Hunt-allocation menu (R2-RS-P1-09): the
        eye_allocate rows of get_legal_actions under _eye_allocation_phase
        (0..min(max(companions, 1), unused Shadow dice); ruling 1.271
        minimum of 1 after an FP Hunt Box retrieval).  [] when no Shadow
        die is unused.  Raises when the engine serves anything else there
        (an open decision comes first in get_legal_actions)."""
        import ai
        state = self.state
        if not any(not d.used for d in state.dice.get(Side.SHADOW, [])):
            return []
        state._eye_allocation_phase = True
        try:
            acts = ai.get_legal_actions(state, Side.SHADOW)
        finally:
            state._eye_allocation_phase = False
        if not acts or any(a.action_type != 'eye_allocate' for a in acts):
            raise RuntimeError(
                "the engine eye menu is unavailable (served %s)"
                % sorted({a.action_type for a in acts}))
        return acts

    def _human_eye_allocation(self, required: bool) -> bool:
        """A human Shadow's Hunt allocation (R2-RS-P1-09): the choices are
        the ENGINE's eye_allocate menu and the pick executes through
        ai._execute_action (I1) - never a per-die tick list handed to
        game_logic.allocate_eyes, which applies no rule.  required=True
        (ROLL DICE, 1.271 minimum owed) has no Cancel; Alloc Eyes may
        cancel.  True when an allocation was executed; raises when a
        required choice gets no answer (nothing allocated or rolled)."""
        import ai
        acts = self._engine_eye_menu()
        if not acts:
            self.log("No unused Shadow die to allocate as an Eye")
            return False
        counts = [a.params.get('count', 0) for a in acts]
        note = (" - ruling 1.271: the Free Peoples took a die from the "
                "Hunt Box last turn, so at least 1" if counts[0] >= 1
                else "")
        opts = [(f"{n} eye(s) to the Hunt Box", n) for n in counts]
        pick = self._show_choice_dialog(
            "Shadow Hunt allocation",
            f"Shadow dice to allocate as Eyes before the roll{note}:",
            opts, allow_cancel=not required)
        if pick is None:
            if required:
                raise RuntimeError("Shadow Hunt allocation dialog closed "
                                   "without a choice - nothing allocated")
            self.log("Hunt allocation cancelled - nothing allocated")
            return False
        ai._execute_action(self.state, acts[counts.index(pick)])
        self.log(f"Shadow: Pre-allocated {pick} eye(s) to the Hunt Box")
        return True

    def _on_pass_turn(self):
        """Pass without using a die (only when opponent has more unused dice)."""
        if self.ai_playing:
            self.log("Wait for AI to finish its turn")
            return
        # R2-SL-34: Pass while this side's own second army move / second
        # recruit is open DECLINES it through the engine's window-closing
        # row (that die) - never a hand-over that leaves the window open
        # for the AI to take.  Hot-seat too: after 'Yes' it is the
        # window's only decline.  (An owner-less legacy window - side None,
        # served to either side - is declined by whoever passes.)
        own = [w for w in self._multistep_windows()
               if w[1] in (None, self.current_side)]
        if own:
            if self._ai_mode_decision_open(continues=own[0][0]):
                return
            try:
                self._close_multistep_window(*own[0], why='Pass')
            except Exception as e:
                self.log(f"Could not decline the "
                         f"{self._multistep_text(own[0][0])} ({e})")
                import traceback
                traceback.print_exc()
                self.refresh()
                return
            self.selected_die_idx = None
            self.pending_action = None
            self.action_source = None
            self._arm_ai_reply()
            self.refresh()
            return
        if not self.ai_side:
            self.log("Pass is only available in AI mode")
            return
        human_side = self.current_side
        ai_unused = self._count_unused_dice(self.ai_side)
        human_unused = self._count_unused_dice(human_side)
        if ai_unused <= human_unused:
            self.log("Cannot pass - opponent doesn't have more dice than you")
            return
        self.log(f"{human_side.value}: Pass")
        # Trigger AI response (same as after an action, but no die consumed)
        self._after_human_action()

    def _after_human_action(self):
        """After human acts or passes, check dice state and trigger one AI action."""
        if not self.ai_side or not self.ai_net or self.ai_playing:
            return
        if self._ai_waits_for_human_window():   # R2-SL-34
            return
        human_side = Side.FREE_PEOPLES if self.ai_side == Side.SHADOW else Side.SHADOW
        ai_unused = self._count_unused_dice(self.ai_side)
        human_unused = self._count_unused_dice(human_side)
        if ai_unused == 0 and human_unused == 0:
            self.log("All dice used - action phase complete")
            return
        if ai_unused > 0:
            # R2-SL-32: never start the AI inside a decision the human
            # owns (get_legal_actions refuses to ask the AI for it)
            open_side = self._engine_pending_side()
            if open_side is not None and open_side != self.ai_side:
                self._defer_ai_reply(open_side)
                return
            self._ai_reply_deferred = None
            self.ai_playing = True
            self.root.after(500, self._ai_play_one_action)

    def _defer_ai_reply(self, open_side):
        """R2-SL-32: a decision the HUMAN owns is still open, so the AI
        does not act.  ai_playing stays False (every human button keeps
        working): the human resolves the decision in its dialog or, on
        their next action button, through _ai_mode_decision_open.  The
        reply looks again in 1 s and runs once the decision is resolved;
        it never carries into the next turn.  Logged once per wait."""
        self.ai_playing = False
        if getattr(self, '_ai_reply_deferred', None) is None:
            self.log(f"{open_side.value} decision still open - the "
                     f"{self.ai_side.value} AI replies once it is resolved")
        self._ai_reply_deferred = open_side
        if hasattr(self, '_ai_after_id'):
            self.root.after_cancel(self._ai_after_id)
        self._ai_after_id = self.root.after(1000, self._retry_ai_reply,
                                            self.state.turn)

    def _retry_ai_reply(self, turn):
        """R2-SL-32: the deferred AI reply (armed by _defer_ai_reply)."""
        if self.state.turn != turn:
            self._ai_reply_deferred = None
            self.log(f"AI reply from turn {turn} dropped - that turn "
                     f"has ended")
            return
        self._after_human_action()

    # R5-DOC-COMMENTS-3 (R4-GUI-SKIP-DIE-BUTTON FU1): the exits the logs name
    # once the AI stopped without acting (ai_playing cleared, nothing
    # re-armed).  A finished human action arms the AI reply (_arm_ai_reply);
    # Pass works while the AI holds more unused dice (_on_pass_turn); End
    # Turn does not simply end the turn over the AI's usable dice - it asks
    # first (_ask_forfeit_ai_dice, rule 1.67: No asks the AI again, Yes
    # forfeits them as a DEFECT).
    _AI_STOPPED_EXITS = ("you have control: your next action, or Pass while "
                         "the AI holds more unused dice, asks the AI again; "
                         "End Turn asks before it forfeits the AI's usable "
                         "dice")

    def _ai_play_one_action(self):
        """AI plays exactly ONE action, then returns control to the human.

        R2-SL-32: this runs as a Tk after() callback.  An exception that
        escaped it (a raising legal menu or search) was printed and
        swallowed by Tk with ai_playing still True, and every human button
        then answered 'Wait for AI to finish its turn' for the rest of the
        session.  A failure now always clears the flag and is shown in the
        GUI log; no other action is substituted (I1 / P1-12).  Every
        normal exit of _ai_play_one_action_body clears the flag itself or
        deliberately re-arms the AI's next die."""
        try:
            self._ai_play_one_action_body()
        except Exception as e:
            self.ai_playing = False
            self.log(f"{getattr(self.ai_side, 'value', self.ai_side)} AI: "
                     f"its action FAILED ({type(e).__name__}: {e}) - no "
                     f"action taken, {self._AI_STOPPED_EXITS}")
            import traceback
            traceback.print_exc()
            self.refresh()

    def _ai_play_one_action_body(self):
        """The AI's one action; called only through _ai_play_one_action."""
        from ai import mcts_action, get_legal_actions, _execute_action

        # R2-SL-32: never inside a decision the human owns; an AI-owned
        # decision left open is finished through the one driver first,
        # never searched as a fresh top-level menu
        open_side = self._engine_pending_side()
        if open_side is not None and open_side != self.ai_side:
            self._defer_ai_reply(open_side)
            return
        if open_side is not None:
            self._drive_engine_decisions()
            left = self._engine_pending_side()
            if left is not None:
                raise RuntimeError(
                    f"{left.value} decision still open after the engine "
                    f"driver - the {self.ai_side.value} AI cannot act")
        # R2-SL-34: never inside the human's own open multi-step window
        # (a reply armed before it opened); ai_playing cleared, logged
        if self._ai_waits_for_human_window():
            return

        actions = get_legal_actions(self.state, self.ai_side)
        if not actions:
            self.ai_playing = False
            return

        # Same MCTS decision procedure as self-play training (fewer sims for
        # interactive responsiveness)
        action = mcts_action(self.state, self.ai_side, self.fp_net, self.sh_net,
                             num_simulations=200, temperature=0.1,
                             add_noise=False)
        if action is None:
            # R2-S4-ENGINE-AI-TOOLS (rescan v070, P1-12 sibling): never a
            # silent skip - the AI did not act; say so and stop (state
            # untouched, nothing re-armed; Pass asks the AI again)
            self.ai_playing = False
            self.log(f"{self.ai_side.value} AI search returned no action - "
                     f"AI stopped, no action taken, {self._AI_STOPPED_EXITS}")
            return

        self.log(f"{self.ai_side.value} AI: {self._describe_action(action)}")

        failed = False      # R2-SL-32: re-arm below only on success
        try:
            _execute_action(self.state, action)
        except Exception as e:
            failed = True
            self.log(f"{self.ai_side.value} AI: Action failed ({e}) - no "
                     f"action taken, {self._AI_STOPPED_EXITS}")
            import traceback
            traceback.print_exc()
        else:
            # P1-07: the AI's action is not over until every engine
            # decision it opened (its own chain AND the human's decisions
            # inside it: defender cards, retreats, casualties) and its own
            # multi-step die are resolved.  ai_playing stays True until
            # then, so no human button can run mid-chain.
            try:
                self._ai_finish_action()
            except Exception as e:
                failed = True
                self.log(f"{self.ai_side.value} AI: could not finish "
                         f"its action ({e})")
                import traceback
                traceback.print_exc()

        self._sync_hands()
        self.refresh()
        self.ai_playing = False

        # If human has no dice left, AI continues playing remaining dice.
        # R2-SL-32: never after a failure - that retried the same failing
        # action every 800 ms with ai_playing True between tries; the
        # human keeps control: a human action, or Pass while the AI holds
        # more unused dice, asks the AI again; End Turn asks first
        # (_ask_forfeit_ai_dice, rule 1.67: No asks the AI again, Yes
        # forfeits its usable dice as a DEFECT) - _AI_STOPPED_EXITS
        human_side = Side.FREE_PEOPLES if self.ai_side == Side.SHADOW else Side.SHADOW
        human_unused = self._count_unused_dice(human_side)
        ai_unused = self._count_unused_dice(self.ai_side)
        if failed:
            return
        if human_unused == 0 and ai_unused > 0:
            self.ai_playing = True
            self.root.after(800, self._ai_play_one_action)
        elif ai_unused == 0 and human_unused == 0:
            self.log("All dice used - action phase complete")

    def _ai_multistep_open(self) -> bool:
        """True while the AI's just-taken action still holds an engine
        multi-step die (second army move / second muster recruit) or an
        Elven Ring free action: the non-pending part of the engine's
        'required decision' set (ai._next_acting_side keeps the acting
        side's seat for exactly these).  R2-SL-34: only a window the AI
        OWNS (pending_*_side) - the human's half-spent Muster/Army die
        made the AI play its whole pool inside the human's window."""
        if any(owner == self.ai_side
               for _k, owner, _d in self._multistep_windows()):
            return True
        rfs = getattr(self.state, '_ring_free_side', None)
        return rfs == self.ai_side and self._count_unused_dice(rfs) > 0

    def _multistep_windows(self):
        """R2-SL-34: every open engine multi-step die window as
        (kind, owner side, die index) - 'army_move' = the Army die's
        second move, 'muster' = the Muster die's second recruit.  The
        owner is the engine's pending_*_side; get_legal_actions serves
        the window's menu (and its closing pass row) to that side only."""
        st = self.state
        wins = []
        if st.pending_army_move_die is not None:
            wins.append(('army_move',
                         getattr(st, 'pending_army_move_side', None),
                         st.pending_army_move_die))
        if st.pending_muster_die is not None:
            wins.append(('muster', getattr(st, 'pending_muster_side', None),
                         st.pending_muster_die))
        return wins

    def _human_multistep_window(self):
        """R2-SL-34 (AI mode): the open multi-step window a non-AI side
        owns, as (kind, owner, die), else None."""
        if self.ai_side is None:
            return None
        return next((w for w in self._multistep_windows()
                     if w[1] != self.ai_side), None)

    @staticmethod
    def _multistep_text(kind):
        return 'second army move' if kind == 'army_move' else 'second recruit'

    def _ai_waits_for_human_window(self) -> bool:
        """R2-SL-34: True (logged, ai_playing cleared) while the human's
        own multi-step window is open - the AI never acts inside it.  No
        retry timer: every exit of the window re-arms the reply (the
        second move / recruit, the 'No' answer, Pass); End Turn closes it."""
        win = self._human_multistep_window()
        if win is None:
            return False
        self.ai_playing = False
        self.log(f"Your {self._multistep_text(win[0])} is still open - "
                 f"make it, or press Pass to decline it; the "
                 f"{self.ai_side.value} AI replies after that")
        return True

    def _close_multistep_window(self, kind, owner, die_idx, why):
        """R2-SL-34: close an open multi-step window through the engine's
        own window-closing row (pass on the window's die - the row
        get_legal_actions serves its owner): the die is spent and the
        unused half forfeited."""
        from ai import Action, _execute_action
        _execute_action(self.state, Action(die_idx=die_idx,
                                           action_type='pass', params={}))
        self.log(f"{getattr(owner, 'value', owner)}: "
                 f"{self._multistep_text(kind)} declined ({why})")

    def _multistep_blocks(self, continues=None) -> bool:
        """R2-SL-34 half of the P1-07 guard (AI mode): no top-level human
        action while a multi-step window is open, except that window's
        own continuation (`continues` = its kind).  The engine would run
        it as the window's second half or beside it.  A human-owned
        window is refused with its exits named; an AI-owned one (left by
        a failed AI action) is finished by the AI first."""
        for kind, owner, _die in self._multistep_windows():
            if owner == self.ai_side:
                self.log(f"The {self.ai_side.value} AI's "
                         f"{self._multistep_text(kind)} is still open - "
                         f"the AI finishes it first; action not taken")
                self.ai_playing = True
                try:
                    self._ai_finish_action()
                except Exception as e:
                    self.log(f"{self.ai_side.value} AI: could not finish "
                             f"its {self._multistep_text(kind)} ({e})")
                    import traceback
                    traceback.print_exc()
                finally:
                    self.ai_playing = False
                self.refresh()
                return True
            if continues != kind:
                self.log(f"Your {self._multistep_text(kind)} is still "
                         f"open - make it, or press Pass to decline it; "
                         f"action not taken")
                self.refresh()
                return True
        return False

    def _ai_finish_action(self):
        """Drive the AI's just-executed action to completion (P1-07).
        Every open engine decision goes through _drive_engine_decisions
        (the AI picks its own, the human is prompted for theirs); an open
        multi-step die of the AI is played by the AI itself.  Raises when
        an AI-owned decision cannot be resolved; a human-owned decision
        left open is logged and re-driven by _ai_mode_decision_open before
        any new human action."""
        from ai import mcts_action, _execute_action
        for _ in range(400):
            if self._engine_pending_side() is not None:
                self._drive_engine_decisions()
                left = self._engine_pending_side()
                if left is None:
                    continue
                if left != self.ai_side:
                    self.log(f"{left.value} decision still open - "
                             f"resolve it before acting")
                    return
                raise RuntimeError(
                    f"{self.ai_side.value} AI decision left open by the "
                    f"engine driver")
            if not self._ai_multistep_open():
                return
            action = mcts_action(self.state, self.ai_side,
                                 self.fp_net, self.sh_net,
                                 num_simulations=200, temperature=0.1,
                                 add_noise=False)
            if action is None:
                raise RuntimeError(
                    f"{self.ai_side.value} AI search returned no action "
                    f"for its open multi-step die")
            self.log(f"{self.ai_side.value} AI: "
                     f"{self._describe_action(action)}")
            _execute_action(self.state, action)
        raise RuntimeError(
            f"{self.ai_side.value} AI action still open after 400 steps")

    def _ai_mode_decision_open(self, continues=None) -> bool:
        """P1-07 guard for top-level human actions in AI mode: none may
        start while an engine decision is open (ai._execute_action has no
        owner guard, so the engine would run it inside the AI's chain or
        battle).  The open decision is driven through the one driver
        instead; returns True when the caller must refuse the action.
        R2-SL-34: nor while a multi-step die window is open, unless the
        action is that window's continuation (`continues` = 'army_move' /
        'muster', the caller's own second half) - see _multistep_blocks."""
        if self.ai_side is None:
            return False
        if self._engine_pending_side() is None:
            return self._multistep_blocks(continues)
        self.log("An open decision must be resolved first - "
                 "action not taken")
        try:
            self._drive_engine_decisions()
        except Exception as e:
            self.log(f"Could not resolve the open decision ({e})")
            import traceback
            traceback.print_exc()
        self.refresh()
        return True

    def _describe_action(self, action):
        """Return a human-readable string for an AI action."""
        t = action.action_type
        p = action.params
        if t == 'army_move':
            # R3-COSMETIC-BRIDGE-TEXT (R2-E5-01-SURFACE FU3): the split (units,
            # the named leader set, Nazgul) - one menu holds a row per figure
            # set since D5, and the human choice dialog labels rows with this
            import engine_play
            return (f"Moves army {p['from']} -> {p['to']}"
                    + engine_play.army_move_split_text(p))
        elif t == 'skip':
            # R3-COSMETIC-BRIDGE-TEXT: the die-discard row (USER A2-06)
            import engine_play
            return engine_play.die_discard_text(p)
        elif t == 'army_attack':
            return f"Attacks {p['to']} from {p['from']}"
        elif t == 'advance_political':
            return f"Advances {p['nation'].value} politically"
        elif t == 'muster_recruit':
            return f"Musters {p['type']} in {p['region']}"
        elif t == 'draw_card':
            return f"Draws {p['deck']} card"
        elif t == 'play_event':
            return f"Plays event: {p['card_id']}"
        elif t == 'muster_minion':
            return f"Musters {p['minion']} at {p['region']}"
        elif t == 'fellowship_move':
            return "Moves fellowship"
        elif t == 'fellowship_hide':
            return "Hides fellowship"
        elif t == 'transform_gandalf':
            return f"Transforms Gandalf the White at {p['region']}"
        elif t == 'crown_aragorn':
            return "Crowns Aragorn"
        elif t == 'fellowship_declare':
            return f"Declares fellowship at {p.get('region', '?')}"
        elif t == 'choose_guide':
            return f"Guide: {p.get('companion', '?')}"
        elif t == 'pass':
            return "Passes (wastes die)"
        # R18-SRC-TOOLS: the Hunt decisions FSP Move opens through the
        # engine (the removed hunt dialog's buttons carried this text)
        elif t == 'hunt_take_corruption':
            return "Take the Hunt damage as Corruption"
        elif t == 'hunt_sacrifice_guide':
            return ("Eliminate the Guide (absorbs Hunt damage up to his "
                    "level)")
        elif t == 'hunt_sacrifice_random':
            return ("A random Companion takes the Hunt damage (absorbs up "
                    "to his level)")
        elif t == 'hunt_guide_separate':
            return (f"Separate the Guide {p.get('companion', '?')} "
                    f"(Hunt damage -1)")
        elif t == 'hunt_gollum_reveal':
            return "Gollum: reveal the Fellowship (Hunt damage -1)"
        elif t == 'hunt_gollum_no_reveal':
            return "Gollum: stay hidden (full Hunt damage)"
        elif t == 'fellowship_reveal_place':
            region = str(p.get('region', '?'))
            name = board_module.regions.get(sanitize(region), {}).get(
                'orig', region)
            return f"Reveal the Fellowship in {name}"
        return str(action)

    # --- Zoom Controls ---

    def _zoom_in(self):
        self.zoom_level = min(1.5, self.zoom_level + 0.1)
        self.zoom_label.configure(text=f"{int(self.zoom_level*100)}%")
        self._load_images()
        self.refresh()

    def _zoom_out(self):
        self.zoom_level = max(0.2, self.zoom_level - 0.1)
        self.zoom_label.configure(text=f"{int(self.zoom_level*100)}%")
        self._load_images()
        self.refresh()

    def _zoom_fit(self):
        # Fit to canvas size
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 100:
            canvas_w = 800
        if canvas_h < 100:
            canvas_h = 600

        try:
            img = Image.open(self.board.background_path)
            scale_w = canvas_w / img.width
            scale_h = canvas_h / img.height
            self.zoom_level = min(scale_w, scale_h) * 0.95
            self.zoom_label.configure(text=f"{int(self.zoom_level*100)}%")
            self._load_images()
            self.refresh()
        except:
            pass

    def log(self, message: str, tag: str = None):
        """Add message to log, optionally with a color tag."""
        self.log_text.configure(state=tk.NORMAL)
        if tag:
            self.log_text.insert(tk.END, f"> {message}\n", tag)
        else:
            self.log_text.insert(tk.END, f"> {message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _report_callback_exception(self, exc_type, exc, tb):
        """Tk's report_callback_exception for this window (installed in
        __init__; R3-LEGACY-GUI-TURN, R2-RS-P1-09 FU2).  Tk's default
        printed an exception escaping a callback to stderr only, so the
        user saw a button do nothing.  The traceback still goes to stderr
        and the failure is named in the GUI log.  Nothing is retried or
        substituted and no flag is touched: the handlers that hold a flag
        (ai_playing, an engine phase flag) clear it themselves."""
        import sys
        import traceback
        traceback.print_exception(exc_type, exc, tb)
        try:
            self.log(f"UI callback FAILED ({exc_type.__name__}: {exc}) - "
                     f"nothing was substituted; traceback on the console")
        except Exception as log_err:
            print(f"[gui] the callback failure above could not be shown in "
                  f"the GUI log ({type(log_err).__name__}: {log_err})",
                  file=sys.stderr)

    def run(self):
        """Start the GUI main loop."""
        self.root.mainloop()

    # ------------------------------------------------------------------
    # Trace replay mode
    # ------------------------------------------------------------------

    def enter_replay_mode(self, trace_path: str):
        """Load a recorded self-play trace and step through it."""
        from game_trace import load_trace

        trace = load_trace(trace_path)
        self.replay_steps = trace['steps']
        self.replay_meta = trace.get('meta', {})
        self.ai_side = None
        self.selfplay_running = False
        self.root.title(f"War of the Ring — Replay: {os.path.basename(trace_path)}")

        # Control bar at the bottom
        bar = tk.Frame(self.root, bg=COLORS["panel_bg"])
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Button(bar, text="|<", width=3,
                  command=lambda: self._replay_goto(0)).pack(side=tk.LEFT, padx=2, pady=3)
        tk.Button(bar, text="< Prev", width=7,
                  command=lambda: self._replay_goto(self.replay_idx - 1)).pack(side=tk.LEFT, padx=2)
        tk.Button(bar, text="Next >", width=7,
                  command=lambda: self._replay_goto(self.replay_idx + 1)).pack(side=tk.LEFT, padx=2)
        tk.Button(bar, text=">|", width=3,
                  command=lambda: self._replay_goto(len(self.replay_steps) - 1)).pack(side=tk.LEFT, padx=2)
        meta = self.replay_meta
        winner = meta.get('winner')
        outcome = (f"{str(winner).upper()} wins ({meta.get('victory_type', '?')}) "
                   f"in {meta.get('turns', '?')} turns" if winner in ('fp', 'sh') else
                   f"Game {meta.get('status', 'unfinished')} ({meta.get('reason') or 'no decided winner'})")
        self.replay_label = tk.Label(bar, text=outcome, anchor='w',
                                     bg=COLORS["panel_bg"], fg=COLORS["text"],
                                     font=("Arial", 10))
        self.replay_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.root.bind('<Left>', lambda e: self._replay_goto(self.replay_idx - 1))
        self.root.bind('<Right>', lambda e: self._replay_goto(self.replay_idx + 1))

        self.log(f"Replay loaded: {len(self.replay_steps)} steps — {outcome}")
        self.log("Use Next/Prev buttons or Left/Right arrow keys.")
        self._replay_goto(0)

    def _replay_goto(self, idx: int):
        """Jump to a trace step: swap in its state snapshot and refresh."""
        from game_trace import describe_step

        if not self.replay_steps:
            return
        idx = max(0, min(idx, len(self.replay_steps) - 1))
        self.replay_idx = idx
        step = self.replay_steps[idx]

        self.state = step['state']
        # Sync GUI-local hand mirror from the snapshot
        try:
            self.hands = {
                Side.FREE_PEOPLES: list(self.state.hands[Side.FREE_PEOPLES].hand),
                Side.SHADOW: list(self.state.hands[Side.SHADOW].hand),
            }
        except Exception:
            pass
        if step.get('side') == 'SH':
            self.current_side = Side.SHADOW
        elif step.get('side') == 'FP':
            self.current_side = Side.FREE_PEOPLES

        desc = describe_step(step)
        meta = self.replay_meta
        winner = meta.get('winner')
        outcome = (f"{str(winner).upper()} wins ({meta.get('victory_type', '?')})" if winner in ('fp', 'sh') else
                   f"Game {meta.get('status', 'unfinished')} ({meta.get('reason') or 'no decided winner'})")
        self.replay_label.config(
            text=f"[{idx + 1}/{len(self.replay_steps)}] {desc}    |    {outcome}")
        if step['kind'] != 'initial':
            self.log(desc)
        self.refresh()

    # ------------------------------------------------------------------
    # LIVE ENGINE-PLAY mode (2026-08-14, USER: the GUI is the face).
    # engine_play.EngineSession is the SOLE driver: legal menus,
    # execution, alternation, pendings and victory timing all come from
    # the engine.  This block is pure front-end: it renders through the
    # SAME refresh()/board path the trace viewer uses (a live game is a
    # trace being appended one decision at a time) and presents the
    # human's legal menu in a panel.  AI picks (MCTS) run on a worker
    # thread; every Tk update stays on the main thread via the
    # queue/after pattern.
    # ------------------------------------------------------------------

    def enter_live_mode(self, mode='hvai', human='fp', sims=300,
                        seed=None, max_turns=40, shadow_vp=10,
                        net_tag='auto', autoclose=False):
        """Start a live engine-driven game.

        mode 'hvai' (human vs AI; `human` = 'fp'|'sh') or 'watch'
        (AI vs AI, every action logged and rendered).  `autoclose`
        destroys the window when the game ends (automated smoke)."""
        import queue
        import engine_play as ep
        self._ep = ep
        self.live_human = None
        if mode == 'hvai':
            self.live_human = (Side.FREE_PEOPLES if human == 'fp'
                               else Side.SHADOW)
        # hidden-info rule: a human playing Shadow never sees the
        # hidden Fellowship's true position (board marker + label)
        self.live_hide_fellowship = self.live_human is Side.SHADOW
        self._live_logq = queue.Queue()
        self._live_aiq = queue.Queue()
        self._live_frames = 0      # render counter (smoke-test probe)
        self._live_done = False
        self._live_autoclose = autoclose
        self.live_current_dp = None
        self._live_map = []
        session = ep.EngineSession(
            seed=seed, mcts_sims=sims, max_turns=max_turns,
            shadow_vp=shadow_vp, human_side=self.live_human,
            log=lambda s: self._live_logq.put(str(s)))
        try:
            fp_net, sh_net = ep.load_nets(net_tag)
            # R2-MODEL-CANDIDATES: inside the dialog block - attach_nets
            # refuses an AI seat net with an all-zero value head (INV-RING-1)
            session.attach_nets(fp_net, sh_net)
            # R17-GUI-ENGINEPLAY (row 75): the AI seats search with the
            # live AI's own contract (engine_play.SEARCH_CONTRACTS, on the
            # WOTR_LIVE_SEARCH_BACKEND the bat sets); a seat or native box
            # that search refuses fails here, before play, in this dialog
            session.prepare_search()
        except BaseException as e:
            # visible even under pythonw (no console)
            try:
                messagebox.showerror('AlphaWOTR', 'net load failed: %s'
                                     % e)
            except Exception:
                pass
            raise
        self.live_session = session
        # adopt the session's state (same swap the replay viewer does)
        self.state = session.state
        self.current_side = self.live_human or Side.FREE_PEOPLES
        self._sync_hands()
        who = ('watch AI vs AI' if self.live_human is None else
               'you play %s' % ('FREE PEOPLES'
                                if self.live_human is Side.FREE_PEOPLES
                                else 'SHADOW'))
        _title = ('War of the Ring - LIVE (%s, seed %d, %d sims)'
                  % (who, session.seed, sims))
        if autoclose:
            # USER 2026-08-14: automated smokes on a real desktop must
            # be unmistakable to a human watching the screen
            _title = '[SMOKE TEST] ' + _title
        self.root.title(_title)
        self._build_live_controls(sims)
        if self.live_human is not None:
            self._build_live_panel()
        self.log('LIVE engine-play: %s' % who, 'turn')
        self.log('The engine enforces everything: only listed legal '
                 'actions are playable; turn order, pendings and '
                 'victory timing are the engine\'s.', 'die')
        self._live_drain_log()
        self.root.after(200, self._live_next)

    def _build_live_controls(self, init_sims):
        """Bottom control bar (both live modes): the AI-strength slider.
        The slider writes EngineSession.sims_override; ai_pick re-reads
        it per decision, so a change takes effect at the NEXT AI
        decision -- no restart.  (Same bottom-bar pattern as the replay
        control bar.)"""
        bar = tk.Frame(self.root, bg=COLORS['panel_bg'])
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.live_sims_label = tk.Label(
            bar,
            text='AI search (sims/decision): %d - higher = stronger, '
                 'slower (applies from the next AI decision)'
                 % init_sims,
            bg=COLORS['panel_bg'], fg=COLORS['text'], anchor='w')
        self.live_sims_label.pack(side=tk.LEFT, padx=8)
        # P1-14: the slider must hold the requested --sims EXACTLY.  Tk's
        # scale 'set' rounds to -resolution and clamps to [from, to], and
        # at its first redraw runs -command with that altered value, which
        # _live_sims_changed wrote into sims_override: --sims 2000 searched
        # 1000, --sims 10 searched 30, --sims 305 searched 310, while the
        # title claimed the requested budget.  Widen the range to contain
        # the requested value and use resolution 1 when it is off the
        # 10-grid, so set() is an identity and the slider never overrides.
        init_sims = int(init_sims)
        self.live_sims_scale = tk.Scale(
            bar, from_=min(30, init_sims), to=max(1000, init_sims),
            resolution=(10 if init_sims % 10 == 0 else 1),
            orient=tk.HORIZONTAL, showvalue=0, length=260,
            bg=COLORS['panel_bg'], fg=COLORS['text'],
            highlightthickness=0, troughcolor=COLORS['bg'],
            command=self._live_sims_changed)
        self.live_sims_scale.set(init_sims)
        self.live_sims_scale.pack(side=tk.LEFT, padx=8, pady=2)

    def _live_sims_changed(self, val):
        try:
            n = int(float(val))
        except (TypeError, ValueError):
            return
        s = getattr(self, 'live_session', None)
        if s is None:
            return
        s.sims_override = n
        self.live_sims_label.configure(
            text='AI search (sims/decision): %d - higher = stronger, '
                 'slower (applies from the next AI decision)' % n)

    def _build_live_panel(self):
        """Decision panel: the ONLY way to act in live mode."""
        top = tk.Toplevel(self.root)
        top.title('Your decisions (legal actions only)')
        top.geometry('620x560')
        top.configure(bg=COLORS['panel_bg'])
        self.live_panel = top
        self.live_status = tk.Label(
            top, text='starting...', justify=tk.LEFT, anchor='w',
            wraplength=600, bg=COLORS['panel_bg'], fg=COLORS['text'],
            font=('Arial', 11, 'bold'))
        self.live_status.pack(fill=tk.X, padx=8, pady=(8, 4))
        frow = tk.Frame(top, bg=COLORS['panel_bg'])
        frow.pack(fill=tk.X, padx=8)
        tk.Label(frow, text='Filter:', bg=COLORS['panel_bg'],
                 fg=COLORS['text']).pack(side=tk.LEFT)
        self.live_filter_var = tk.StringVar()
        self.live_filter_var.trace_add('write', self._live_apply_filter)
        tk.Entry(frow, textvariable=self.live_filter_var,
                 bg=COLORS['bg'], fg=COLORS['text'],
                 insertbackground=COLORS['text']).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        lframe = tk.Frame(top, bg=COLORS['panel_bg'])
        lframe.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        sb = tk.Scrollbar(lframe)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.live_list = tk.Listbox(
            lframe, bg=COLORS['bg'], fg=COLORS['text'],
            selectbackground=COLORS['highlight'],
            selectforeground='black', font=('Consolas', 9),
            yscrollcommand=sb.set)
        self.live_list.pack(fill=tk.BOTH, expand=True)
        sb.configure(command=self.live_list.yview)
        self.live_list.bind('<Double-Button-1>',
                            self._live_play_selected)
        self.live_play_btn = tk.Button(
            top, text='Play selected action',
            command=self._live_play_selected,
            bg=COLORS['success'], fg='white',
            font=('Arial', 11, 'bold'), state=tk.DISABLED)
        self.live_play_btn.pack(fill=tk.X, padx=8, pady=(0, 4))
        tk.Label(top, text='Pendings (battles, hunts, card choices) '
                           'appear here as menus too. Filter matches '
                           'text or action type.',
                 bg=COLORS['panel_bg'], fg=COLORS['text_dim'],
                 wraplength=600, justify=tk.LEFT).pack(
            fill=tk.X, padx=8, pady=(0, 6))
        top.protocol('WM_DELETE_WINDOW', self._live_panel_close)

    def _live_panel_close(self):
        if self._live_done:
            self.live_panel.destroy()
            return
        if messagebox.askyesno('Quit game?',
                               'Abandon the current game?'):
            self.live_session.abandon()
            self._live_finish()

    def _live_drain_log(self):
        import queue
        try:
            while True:
                self.log(self._live_logq.get_nowait())
        except queue.Empty:
            pass
        if not self._live_done:
            self.root.after(150, self._live_drain_log)

    def _live_refresh(self):
        self._sync_hands()
        self.refresh()
        self._live_frames += 1

    def _live_next(self):
        """Advance to the next decision point (main thread only)."""
        if self._live_done:
            return
        s = self.live_session
        dp = s.current()
        self._live_refresh()
        if dp is None:
            self._live_finish()
            return
        if self.live_human is not None and dp.side is self.live_human:
            self._live_populate(dp)
            return
        # AI decision: MCTS on a worker thread, result via queue
        side_nm = ('FP' if dp.side is Side.FREE_PEOPLES else 'SHADOW')
        if self.live_human is not None:
            self.live_status.configure(
                text='AI (%s) thinking... (%s, %d sims)'
                     % (side_nm, dp.kind,
                        self.live_session.sims_for_current()))
            self.live_list.delete(0, tk.END)
            self.live_play_btn.configure(state=tk.DISABLED)
        import threading
        threading.Thread(target=self._live_ai_worker,
                         daemon=True).start()
        self.root.after(120, self._live_poll_ai)

    def _live_ai_worker(self):
        """Worker thread: ONLY computes the MCTS pick (no Tk, no
        state mutation -- submit happens back on the main thread)."""
        try:
            self._live_aiq.put(('ok', self.live_session.ai_pick()))
        except Exception as e:
            self._live_aiq.put(('err', repr(e)))

    def _live_poll_ai(self):
        import queue
        try:
            kind, got = self._live_aiq.get_nowait()
        except queue.Empty:
            if not self._live_done:
                self.root.after(120, self._live_poll_ai)
            return
        if self._live_done:
            return
        if kind == 'err':
            self.log('AI search FAILED: %s - game halted' % got)
            self._live_finish(error='AI error: %s' % got)
            return
        s = self.live_session
        dp = s.current()
        if dp is None:
            self._live_finish()
            return
        # P1-01: masked for the human seat (the AI's face-down discards,
        # committed attacker combat card, table-kill payment pair)
        desc = self._ep.describe_action(s.state, dp.side,
                                        dp.actions[got],
                                        viewer=self.live_human)
        nm = 'FP' if dp.side is Side.FREE_PEOPLES else 'SH'
        _n9 = s.last_pick_sims  # what ai_pick actually used (slider-live)
        try:
            s.submit(got, dp.side)
        except Exception as e:
            # P1-03: never let an engine exception escape this Tk
            # callback (the after() chain died, status stuck on
            # 'thinking').  An error inside the decision halts the
            # session (EngineSession.halt_error): halt the game loudly.
            _c = getattr(s, 'halt_error', None) or e
            self.log('T%d %s (AI): %s [sims=%s] - FAILED, game halted'
                     % (dp.turn, nm, desc, _n9),
                     'fp' if nm == 'FP' else 'sh')
            self._live_finish(
                error=('engine refused the AI move: %s' % e
                       if isinstance(e, self._ep.IllegalPlay) else
                       'engine error: %s: %s' % (type(_c).__name__, _c)))
            return
        self.log('T%d %s (AI): %s [sims=%s]' % (dp.turn, nm, desc, _n9),
                 'fp' if nm == 'FP' else 'sh')
        self.root.after(30, self._live_next)

    def _live_populate(self, dp):
        """Fill the decision panel with the human's legal menu."""
        self.live_current_dp = dp
        nm = ('FREE PEOPLES' if dp.side is Side.FREE_PEOPLES
              else 'SHADOW')
        txt = 'YOUR MOVE (%s) - %s' % (nm, dp.kind)
        if dp.note:
            txt += '\n%s' % dp.note
        txt += ('\n%d legal action(s). Double-click or Play selected.'
                % len(dp.actions))
        self.live_status.configure(text=txt)
        self.live_play_btn.configure(state=tk.NORMAL)
        self._live_apply_filter()
        try:
            self.live_panel.deiconify()
            self.live_panel.lift()
        except Exception:
            pass

    def _live_apply_filter(self, *_a):
        dp = self.live_current_dp
        if dp is None or dp is not self.live_session.current():
            return
        ft = (self.live_filter_var.get() or '').strip().lower()
        self.live_list.delete(0, tk.END)
        self._live_map = []
        st = self.live_session.state
        for i, a in enumerate(dp.actions):
            d = self._ep.describe_action(st, dp.side, a)
            if (ft and ft not in d.lower()
                    and ft not in a.action_type.lower()):
                continue
            self.live_list.insert(tk.END, '[%3d] %s' % (i + 1, d))
            self._live_map.append(i)
        if not self._live_map and ft:
            self.live_list.insert(
                tk.END, '(no legal action matches %r)' % ft)

    def _live_play_selected(self, _event=None):
        dp = self.live_session.current()
        if self._live_done or dp is None:
            return
        sel = self.live_list.curselection()
        if not sel:
            self.log('REFUSED: select an action from the legal '
                     'list first')
            return
        if not self._live_map or sel[0] >= len(self._live_map):
            return
        self._live_play_index(self._live_map[sel[0]])

    def _live_play_index(self, real_idx):
        """Submit menu index `real_idx` as the HUMAN (test-harness
        entry point).  ALL enforcement lives in EngineSession.submit --
        wrong side / bad index / stale menus are refused there."""
        s = self.live_session
        dp = s.current()
        if dp is None:
            self.log('REFUSED: game is over')
            return False
        desc = '?'
        if isinstance(real_idx, int) and 0 <= real_idx < len(dp.actions):
            desc = self._ep.describe_action(s.state, dp.side,
                                            dp.actions[real_idx])
        try:
            s.submit(real_idx, self.live_human)
        except self._ep.IllegalPlay as e:
            self.log('REFUSED: %s' % e)
            return False
        except Exception as e:
            # P1-03: an engine error inside the decision halts the
            # session (EngineSession.halt_error; engine_play.EngineHalted)
            # - not a retryable refusal.  Clear the stale menu and halt
            # the game loudly instead of letting it escape the callback.
            _c = getattr(s, 'halt_error', None) or e
            self.log('T%d %s (you): %s - FAILED, game halted'
                     % (dp.turn, 'FP' if dp.side is Side.FREE_PEOPLES
                        else 'SH', desc))
            self.live_play_btn.configure(state=tk.DISABLED)
            self.live_list.delete(0, tk.END)
            self.live_current_dp = None
            self._live_finish(error='engine error: %s: %s'
                              % (type(_c).__name__, _c))
            return False
        nm = 'FP' if dp.side is Side.FREE_PEOPLES else 'SH'
        self.log('T%d %s (you): %s' % (dp.turn, nm, desc),
                 'fp' if nm == 'FP' else 'sh')
        self.live_play_btn.configure(state=tk.DISABLED)
        self.live_list.delete(0, tk.END)
        self.live_current_dp = None
        self.root.after(30, self._live_next)
        return True

    def _live_finish(self, error=None):
        if self._live_done:
            return
        self._live_done = True
        import queue
        try:
            while True:
                self.log(self._live_logq.get_nowait())
        except queue.Empty:
            pass
        self._live_refresh()
        res = self.live_session.result_info or {}
        # P1-03: a halted session (or one that ended with no result) is
        # a HALT, never a 'LIVE GAME OVER: ? wins by ?' announcement
        _halt = getattr(self.live_session, 'halt_error', None)
        if error is None and _halt is not None:
            error = 'engine error: %s: %s' % (type(_halt).__name__, _halt)
        elif error is None and not res:
            error = 'engine error: the session ended with no result'
        if error is not None:
            msg = 'game halted on %s' % error
        elif res.get('winner') == 'timeout':
            # never the bare word 'timeout' in front of a human (USER
            # 2026-08-14): a capped run is a STOP, not a game ending
            msg = ('TURN CAP REACHED (--max-turns %d): game stopped '
                   'undecided after %s turn(s) - NOT a real game '
                   'ending' % (self.live_session.max_turns,
                               res.get('turns', '?')))
        elif res.get('winner') == 'abandoned':
            msg = 'game abandoned after %s turn(s)' % res.get('turns',
                                                              '?')
        else:
            msg = ('%s wins by %s after %s turn(s)'
                   % (str(res.get('winner', '?')).upper(),
                      res.get('victory_type', '?'),
                      res.get('turns', '?')))
            if res.get('message'):
                msg += '\n%s' % res['message']
        _head = ('LIVE GAME HALTED' if error is not None
                 else 'LIVE GAME STOPPED'
                 if res.get('winner') in ('timeout', 'abandoned')
                 else 'LIVE GAME OVER')
        self.log('=== %s: %s ===' % (_head, msg.replace('\n', ' ')),
                 'turn')
        if self.live_human is not None:
            try:
                self.live_status.configure(text='%s\n%s' % (_head, msg))
                self.live_play_btn.configure(state=tk.DISABLED)
            except Exception:
                pass
        if self._live_autoclose:
            self.root.after(800, self.root.destroy)
        elif error is None:
            try:
                messagebox.showinfo(
                    'Game stopped' if res.get('winner') in
                    ('timeout', 'abandoned') else 'Game over', msg)
            except Exception:
                pass


def launch_gui_replay(trace_path: str, board: "Board", step: int = 0):
    """Launch the GUI in trace replay mode (optionally jumped to a
    specific step - the card-usage finder opens the exact action)."""
    from game_trace import load_trace
    import game_state as gs
    state = gs.setup_initial_game(board)  # placeholder; replaced by snapshots
    gui = WotrGUI(state, board)
    gui.enter_replay_mode(trace_path)
    if step:
        gui._replay_goto(int(step))
    gui.run()


def launch_gui(state: "GameState", board: "Board", ai_side: Optional[str] = None):
    """Launch the GUI, optionally with AI controlling one side."""
    gui = WotrGUI(state, board)
    if ai_side:
        side = Side.SHADOW if ai_side == 'shadow' else Side.FREE_PEOPLES
        gui.enable_ai(side)
    gui.run()


def launch_gui_selfplay(state: "GameState", board: "Board"):
    """Launch the GUI in AI-vs-AI watch mode."""
    gui = WotrGUI(state, board)
    gui.root.after(1000, gui.watch_selfplay)  # Start after GUI loads
    gui.run()


def launch_gui_live(mode='hvai', human='fp', sims=300, seed=None,
                    max_turns=40, shadow_vp=10, net_tag='auto',
                    autoclose=False):
    """Launch the GUI in LIVE engine-play mode (EngineSession driver).

    Import order matters: engine_play first (it sets KMP/CUDA env
    BEFORE torch loads).  The GUI is built on a throwaway state and
    enter_live_mode swaps in the session's state (the replay-viewer
    pattern), so the session's seeded RNG never interleaves with GUI
    construction -- same seed = same game, GUI or console."""
    import engine_play  # noqa: F401  (env setup + engine import)
    import game_state as gs
    b = board_module.load_board()
    gui = WotrGUI(gs.setup_initial_game(b), b)
    gui.enter_live_mode(mode=mode, human=human, sims=sims, seed=seed,
                        max_turns=max_turns, shadow_vp=shadow_vp,
                        net_tag=net_tag, autoclose=autoclose)
    gui.run()
    return gui


if __name__ == '__main__':
    # convenience entry: python src/gui.py [--trace] <trace.gz>
    # (the canonical launcher is main.py --replay-trace)
    import argparse as _ap
    import board as _board
    _p = _ap.ArgumentParser()
    _p.add_argument('trace', nargs='?', default=None)
    _p.add_argument('--trace', dest='trace_opt', default=None)
    _p.add_argument('--step', type=int, default=0)
    # LIVE engine-play mode (2026-08-14): play vs the AI / watch AI vs
    # AI on the rules engine itself (engine_play.EngineSession driver)
    _p.add_argument('--live', choices=('fp', 'sh', 'watch'),
                    default=None,
                    help='LIVE play: fp/sh = your side vs the AI; '
                         'watch = AI vs AI')
    _p.add_argument('--sims', type=int, default=300,
                    help='MCTS sims per AI decision (live mode)')
    _p.add_argument('--seed', type=int, default=None)
    _p.add_argument('--max-turns', type=int, default=40)
    _p.add_argument('--shadow-vp', type=int, default=10)
    _p.add_argument('--net', default='auto')
    _p.add_argument('--autoclose', action='store_true',
                    help='close the window when the live game ends '
                         '(automated smoke)')
    _a = _p.parse_args()
    if _a.live:
        launch_gui_live(
            mode=('watch' if _a.live == 'watch' else 'hvai'),
            human=(_a.live if _a.live in ('fp', 'sh') else 'fp'),
            sims=_a.sims, seed=_a.seed, max_turns=_a.max_turns,
            shadow_vp=_a.shadow_vp, net_tag=_a.net,
            autoclose=_a.autoclose)
        raise SystemExit(0)
    _tp = _a.trace_opt or _a.trace
    if _tp:
        if not os.path.exists(_tp):
            # a stale card-index entry (deleted or quarantine-RENAMED
            # batch dir): first resolve the basename across every trace
            # root; only a truly-deleted trace errors - via a dialog,
            # since the launcher usually has no visible console (USER
            # 2026-07-28 / 2026-08-11).
            import trace_resolve as _tr
            _rp, _roots = _tr.resolve_trace(_tp)
            if _rp:
                print('stale trace path resolved -> %s'
                      % _rp.encode('ascii', 'replace')
                          .decode('ascii'))
                _tp = _rp
            else:
                _msg = _tr.missing_message(_tp, _roots)
                print(_msg.encode('ascii', 'replace')
                      .decode('ascii'))
                try:
                    import tkinter as _tk
                    from tkinter import messagebox as _mb
                    _r = _tk.Tk()
                    _r.withdraw()
                    _mb.showerror('Trace not found', _msg)
                    _r.destroy()
                except Exception:
                    pass
                raise SystemExit(2)
        launch_gui_replay(_tp, _board.load_board(), step=_a.step)
    else:
        import game_state as _gs
        _b = _board.load_board()
        launch_gui(_gs.setup_initial_game(_b), _b)


    # (the engine's table-card-kill path is authoritative; no GUI
    #  pay-off dialog)
