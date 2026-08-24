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
FEL_DIR = os.path.join(BASE_DIR, "fellowship")

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
        self.selfplay_turn = 0
        self.selfplay_speed = 800  # ms between actions
        self.selfplay_paused = False

        # Trace replay mode (step through a recorded self-play game)
        self.replay_steps = None
        self.replay_meta = {}
        self.replay_idx = 0

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

        # Bind click and hover events.  LIVE mode guard: manual
        # state-editing controls are refused while an EngineSession is
        # driving the game -- the engine is the only mutator; zoom
        # controls stay usable (view-only).
        def _dispatch(_e=None, _cmd=command, _txt=text):
            if (getattr(self, 'live_session', None) is not None
                    and _txt not in ('-', '+', 'Fit')):
                self.log("REFUSED: manual controls are disabled in "
                         "LIVE mode - use the decision panel "
                         "(engine-legal actions only)")
                return
            _cmd()
        btn.bind("<Button-1>", _dispatch)
        btn.bind("<Enter>", lambda e: btn.configure(bg=hover_color))
        btn.bind("<Leave>", lambda e: btn.configure(bg=bg_color))

        return btn

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
        from game_logic import check_all_victory_conditions

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

        result = check_all_victory_conditions(self.state)
        if not result["victory"]:
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
        """Handle companion separation from fellowship."""
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

        # Get companion names and calculate max movement
        companions = self.state.fellowship.companions
        selected_companions = []
        for idx in selection:
            if idx < len(companions):
                selected_companions.append(companions[idx])

        if not selected_companions:
            self.log("No valid companions selected")
            return

        # Movement = fellowship progress + highest level of separating companions
        highest_level = max(c.level for c in selected_companions)
        movement = self.state.fellowship.progress + highest_level
        selected_names = [c.name for c in selected_companions]

        self.log(f"Separation movement: {self.state.fellowship.progress} (progress) + {highest_level} (level) = {movement}")

        # Ask for destination region
        from game_logic import fellowship_region
        start_region = fellowship_region(self.state)
        dest = self._show_companion_movement_dialog(
            "Separate Companions",
            f"Move {', '.join(selected_names)} (max {movement} regions):",
            start_region, movement
        )
        if not dest:
            self.log("Separation cancelled")
            return

        try:
            # Remove companions from fellowship and place in destination
            for comp in selected_companions:
                self.state.fellowship.companions.remove(comp)
                # Add to army at destination (as a leader)
                dest_san = sanitize(dest)
                # Armies ONLY via get_or_create_army: raw exact-key
                # creation forges duplicates beside original-name keys.
                from game_logic import get_or_create_army
                get_or_create_army(self.state, dest_san).leaders.append(comp)

            # Re-assign guide if needed
            from game_logic import change_guide
            try:
                change_guide(self.state)
            except:
                pass

            self._use_selected_die()
            self.log(f"Separated {', '.join(selected_names)} to {dest}")
        except Exception as e:
            self.log(f"Separation failed: {e}")
            import traceback
            traceback.print_exc()

        self.refresh()

    def _on_move_companions(self):
        """Move companions already on the board (not in fellowship)."""
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

        # Find all companions on the board (not in fellowship)
        companions_on_board = []
        for region_key, army in self.state.armies.items():
            for leader in army.leaders:
                if isinstance(leader, Companion):
                    companions_on_board.append((region_key, leader))

        if not companions_on_board:
            self.log("No companions on the board to move")
            return

        # Show dialog to select which companions to move
        self._show_companion_board_movement_dialog(companions_on_board)

    def _show_companion_board_movement_dialog(self, companions_on_board):
        """
        Dialog to move companions that are already on the board.
        Each companion can move INDEPENDENTLY to different destinations with 1 die.
        Movement = companion's level.
        """
        dialog = tk.Toplevel(self.root)
        dialog.title("Move Companions")
        dialog.configure(bg=COLORS["panel_bg"])
        dialog.geometry("550x650")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(
            dialog, text="Move Companions on Board",
            bg=COLORS["panel_bg"], fg=COLORS["highlight"],
            font=("Arial", 12, "bold")
        ).pack(pady=10)

        tk.Label(
            dialog, text="Each companion can move independently (1 die moves ALL).\nClick a companion to select their destination:",
            bg=COLORS["panel_bg"], fg=COLORS["text"]
        ).pack(pady=5)

        # Scrollable frame for companions
        canvas = tk.Canvas(dialog, bg=COLORS["panel_bg"], highlightthickness=0)
        scrollbar = tk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=COLORS["panel_bg"])

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

        # Track destinations for each companion
        # Key: index in companions_on_board, Value: destination region or None (stay)
        destinations = {}  # Will store {index: destination_region}
        dest_labels = {}   # Will store {index: label_widget}

        def select_destination(idx, region, comp):
            """Open destination selection for this companion."""
            movement = comp.level  # Movement = companion's level
            dest = self._show_companion_movement_dialog(
                f"Move {comp.name}",
                f"Select destination (max {movement} regions):",
                region, movement
            )
            if dest:
                destinations[idx] = dest
                dest_labels[idx].config(text=f"→ {dest}", fg=COLORS["success"])
            else:
                # Clear destination if cancelled
                if idx in destinations:
                    del destinations[idx]
                dest_labels[idx].config(text="(staying)", fg=COLORS["text_dim"])

        def clear_destination(idx):
            """Clear a companion's destination."""
            if idx in destinations:
                del destinations[idx]
            dest_labels[idx].config(text="(staying)", fg=COLORS["text_dim"])

        # Create row for each companion
        for idx, (region, comp) in enumerate(companions_on_board):
            row_frame = tk.Frame(scroll_frame, bg=COLORS["bg"], relief=tk.RAISED, bd=1)
            row_frame.pack(fill=tk.X, padx=5, pady=3)

            # Companion info
            info_frame = tk.Frame(row_frame, bg=COLORS["bg"])
            info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)

            tk.Label(
                info_frame, text=f"{comp.name} (Lv.{comp.level})",
                bg=COLORS["bg"], fg=COLORS["highlight"],
                font=("Arial", 10, "bold")
            ).pack(anchor="w")

            tk.Label(
                info_frame, text=f"Currently in: {region}",
                bg=COLORS["bg"], fg=COLORS["text"]
            ).pack(anchor="w")

            # Destination label
            dest_label = tk.Label(
                info_frame, text="(staying)",
                bg=COLORS["bg"], fg=COLORS["text_dim"]
            )
            dest_label.pack(anchor="w")
            dest_labels[idx] = dest_label

            # Buttons
            btn_frame = tk.Frame(row_frame, bg=COLORS["bg"])
            btn_frame.pack(side=tk.RIGHT, padx=5, pady=5)

            self._make_button(
                btn_frame, "Move",
                lambda i=idx, r=region, c=comp: select_destination(i, r, c),
                bg=COLORS["button_bg"]
            ).pack(side=tk.LEFT, padx=2)

            self._make_button(
                btn_frame, "Clear",
                lambda i=idx: clear_destination(i),
                bg=COLORS["text_dim"]
            ).pack(side=tk.LEFT, padx=2)

        result = {"done": False, "moved": False}

        def execute_moves():
            if not destinations:
                messagebox.showwarning("No Moves", "Select at least one destination to move companions")
                return

            # Execute all moves
            moved_companions = []
            for idx, dest in destinations.items():
                region_key, comp = companions_on_board[idx]

                # Remove from current army (drop the entry when emptied,
                # matching the engine's companion_move executor)
                _src_army = self.state.armies[region_key]
                _src_army.leaders.remove(comp)
                if not _src_army.units and not _src_army.leaders:
                    del self.state.armies[region_key]

                # Add to destination
                dest_san = sanitize(dest)
                # Armies ONLY via get_or_create_army: raw exact-key
                # creation forges duplicates beside original-name keys.
                from game_logic import get_or_create_army
                get_or_create_army(self.state, dest_san).leaders.append(comp)

                moved_companions.append(f"{comp.name} → {dest}")

            self._use_selected_die()
            self.log(f"Moved companions: {', '.join(moved_companions)}")
            result["moved"] = True
            result["done"] = True
            dialog.destroy()

        def cancel():
            result["done"] = True
            dialog.destroy()

        # Bottom button frame
        btn_frame = tk.Frame(dialog, bg=COLORS["panel_bg"])
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        self._make_button(btn_frame, "Execute All Moves", execute_moves, bg=COLORS["success"]).pack(side=tk.LEFT, padx=5)
        self._make_button(btn_frame, "Cancel", cancel).pack(side=tk.LEFT, padx=5)

        dialog.wait_window()
        if result["moved"]:
            self.refresh()

    def _show_companion_movement_dialog(self, title: str, prompt: str,
                                        start_region: str, max_movement: int) -> Optional[str]:
        """Show dialog to select destination for companion movement."""
        result = {"value": None}

        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.configure(bg=COLORS["panel_bg"])
        dialog.geometry("350x500")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text=prompt, bg=COLORS["panel_bg"], fg=COLORS["text"]).pack(pady=10)

        # Calculate reachable regions using BFS, stopping at shadow strongholds
        reachable = self._get_companion_reachable_regions(start_region, max_movement)

        tk.Label(
            dialog, text=f"Reachable from {start_region}:",
            bg=COLORS["panel_bg"], fg=COLORS["text_dim"]
        ).pack(pady=5)

        listbox = tk.Listbox(dialog, bg=COLORS["bg"], fg=COLORS["text"], height=15)
        for r in sorted(reachable):
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

    def _get_companion_reachable_regions(self, start: str, max_steps: int) -> set:
        """
        Get all regions reachable by companions within max_steps.
        Companions must stop at unconquered shadow strongholds.
        """
        from board import regions as board_regions
        from collections import deque

        start_san = sanitize(start)
        reachable = set()
        visited = {start_san: 0}  # region -> steps to reach
        queue = deque([(start_san, 0)])

        while queue:
            current, steps = queue.popleft()

            if steps > 0:  # Don't include starting region
                reachable.add(current)

            # Check if this is an unconquered shadow stronghold (must stop)
            reg_info = board_regions.get(current, {})
            is_shadow_stronghold = reg_info.get("is_stronghold", False)
            # Check if shadow-controlled (would need to check army presence or owner)
            # For now, check if shadow army present
            army = self.state.armies.get(current)
            shadow_controlled = False
            if army and army.units:
                shadow_nations = {Nation.ISENGARD, Nation.SAURON, Nation.SOUTHRON_EASTERLINGS}
                shadow_controlled = army.units[0].nation in shadow_nations

            if is_shadow_stronghold and shadow_controlled and steps > 0:
                # Must stop here, can't continue
                continue

            if steps >= max_steps:
                continue

            # Explore adjacent regions
            for adj in self.board.adjacent(current):
                adj_san = sanitize(adj)
                if adj_san not in visited or visited[adj_san] > steps + 1:
                    visited[adj_san] = steps + 1
                    queue.append((adj_san, steps + 1))

        return reachable

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
        try:
            _execute_action(self.state, action)
        except Exception as e:
            self.log(f"Action failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        if log_msg:
            self.log(log_msg)
        self.selected_die_idx = None
        # resolve any pending decisions the action created (combat/hunt/
        # card-option/muster-double/etc.) through the same engine driver
        if self._engine_pending_side() is not None:
            self._drive_engine_decisions()
        if (self.ai_side and self.current_side != self.ai_side
                and not self.ai_playing):
            if hasattr(self, '_ai_after_id'):
                self.root.after_cancel(self._ai_after_id)
            self._ai_after_id = self.root.after(600, self._after_human_action)
        self.refresh()
        return True

    def _use_selected_die(self):
        """Mark the selected die as used."""
        if self.selected_die_idx is not None:
            self.state.dice[self.current_side][self.selected_die_idx].used = True
            self.selected_die_idx = None
            # In AI mode: schedule AI to respond after human's action completes
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

        # If AI controls Shadow, let it decide on eye allocation before rolling
        if self.ai_side == Side.SHADOW and self.ai_net:
            self._ai_allocate_eyes()

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
        # A second move is available iff the engine left a pending die
        if self.state.pending_army_move_die is not None:
            self.log("  (1 move left - select another army, or click Done)")
            self._offer_second_move()
        else:
            self.selected_die_idx = None
            if (self.ai_side and self.current_side != self.ai_side
                    and not self.ai_playing):
                self._ai_after_id = self.root.after(600, self._after_human_action)
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
            self.log("Select another army, then Move Army (2nd move)")
            self.pending_action = "move"
            # the next _complete_move sees pending_army_move_die and the
            # engine applies it as the second move + consumes the die
        else:
            # Done: pass consumes the pending die
            _execute_action(self.state, Action(
                die_idx=self.state.pending_army_move_die,
                action_type='pass', params={}))
            self.selected_die_idx = None
            if (self.ai_side and self.current_side != self.ai_side
                    and not self.ai_playing):
                self._ai_after_id = self.root.after(600, self._after_human_action)
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
        from ai import (get_legal_actions, _execute_action, mcts_action)
        guard = 0
        while guard < 400:
            guard += 1
            side = self._engine_pending_side()
            if side is None:
                break
            acts = get_legal_actions(self.state, side)
            if not acts:
                break
            if len(acts) == 1:
                _execute_action(self.state, acts[0])
                continue
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
                _execute_action(self.state, a or acts[0])
                continue
            # human: present the legal decisions
            sidename = ('Free Peoples' if side == Side.FREE_PEOPLES
                        else 'Shadow')
            opts = [(self._describe_action(a), i) for i, a in enumerate(acts)]
            pick = self._show_choice_dialog(
                f"{sidename} decision", "Choose an action:", opts)
            if pick is None:
                # The pending must resolve; a cancelled dialog defaults to
                # the first legal action, and that pick must be visible.
                self.log(f"Dialog cancelled - defaulting to: "
                         f"{self._describe_action(acts[0])}")
            _execute_action(self.state, acts[0 if pick is None else pick])
        self._sync_hands()
        self.refresh()

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
            self.log("Click another settlement to muster (2nd recruit)")
            self.pending_action = "muster"
        else:
            _execute_action(self.state, Action(
                die_idx=self.state.pending_muster_die,
                action_type='pass', params={}))
            self.selected_die_idx = None
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
            from event_effects import check_event_requirement, play_event_card

            # Check requirements first
            can_play, msg = check_event_requirement(self.state, card_id)
            if not can_play:
                self.log(f"Cannot play: {msg}")
                return

            # stale sweep 2026-07-27 (USER order): the legacy
            # get_card_choices dialog is deleted — play with empty
            # choices and let the LEARNED option-spec pending
            # (pending_card_option) surface the decision, exactly as
            # in self-play.
            choices = {}

            # GUI and engine hands must agree (kept in step by
            # _sync_hands); a missing card is a real desync and must
            # surface, never be papered over by forging the card in.
            if card_id not in self.state.hands[self.current_side].hand:
                self.log(f"Hand desync: {card_id} not in engine hand")
                return

            # Play the card with choices
            result = play_event_card(self.state, self.current_side, card_id, choices)

            if result["success"]:
                # Save the die face before using
                die_face = die.current

                # Use the die
                self._use_selected_die()
                self.log(f"Played: {card.card_name}")
                # Sync hands immediately
                self._sync_hands()
                # Log the effects
                for effect in result.get("effects", []):
                    self.log(f"  → {effect}")

                # Check for Palantír of Orthanc card draw bonus (Shadow only)
                if self.current_side == Side.SHADOW:
                    self._check_palantir_card_draw(getattr(card, 'deck', None))

                # Check for Gandalf's ability (FP only, Event/Will of the West die)
                if self.current_side == Side.FREE_PEOPLES:
                    self._check_gandalf_card_draw(
                        die_face, getattr(card, 'deck', None))

                # A card-driven hunt (e.g. Nazgul Strike!'s
                # roll-for-the-Hunt) leaves tile damage in
                # _pending_hunt_damage; settle it exactly like the
                # engine executor does, or the corruption is lost.
                import game_logic as _gl_phd
                _phd = getattr(self.state, '_pending_hunt_damage', 0)
                if _phd > 0 and self.state.pending_hunt is None:
                    if not _gl_phd._offer_hunt_reduction(self.state):
                        _gl_phd._settle_hunt_damage(self.state)

                # full-pass review 2026-07-27: a card played with empty
                # choices may open a learned pending (pending_card_option
                # e.g. __overstack__/bof_separation, pending_combat) —
                # surface it NOW, not on the next unrelated action.
                self._drive_engine_decisions()

            else:
                self.log(f"Failed: {result.get('message', 'Unknown error')}")

        except Exception as e:
            self.log(f"Play card failed: {e}")
            import traceback
            traceback.print_exc()

        self.refresh()

    def _show_choice_dialog(self, title: str, prompt: str, options: list) -> Optional[str]:
        """
        Show a dialog for single selection.
        Options: list of (display_text, value) tuples
        Returns the selected value, or None if cancelled.
        """
        result = {"value": None, "cancelled": False}

        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.configure(bg=COLORS["panel_bg"])
        dialog.transient(self.root)
        dialog.grab_set()

        # Center the dialog
        dialog.geometry("300x{}".format(min(400, 100 + len(options) * 40)))

        # Prompt label
        tk.Label(
            dialog, text=prompt,
            bg=COLORS["panel_bg"], fg=COLORS["text"],
            font=("Arial", 11), wraplength=280
        ).pack(pady=10, padx=10)

        # Option buttons
        btn_frame = tk.Frame(dialog, bg=COLORS["panel_bg"])
        btn_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        def select_option(value):
            result["value"] = value
            dialog.destroy()

        for display_text, value in options:
            btn = self._make_button(
                btn_frame, display_text,
                lambda v=value: select_option(v),
                width=25
            )
            btn.pack(pady=3)

        # Cancel button
        cancel_frame = tk.Frame(dialog, bg=COLORS["panel_bg"])
        cancel_frame.pack(fill=tk.X, padx=10, pady=10)

        def cancel():
            result["cancelled"] = True
            dialog.destroy()

        cancel_btn = self._make_button(cancel_frame, "Cancel", cancel, width=10)
        cancel_btn.pack()

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
        """Move the Fellowship and resolve hunt interactively."""
        if self.ai_playing:
            self.log('Wait for AI to finish its turn')
            return
        valid = (ActionDieFace.CHARACTER, ActionDieFace.WILL_OF_WEST)
        if self.current_side != Side.FREE_PEOPLES:
            self.log("Only Free Peoples can move the Fellowship")
            return

        if not self._check_die_selected(valid):
            return

        try:
            from game_logic import move_fellowship, get_hunt_params, roll_hunt_dice, reroll_hunt_dice

            # Get hunt parameters BEFORE adding die to box (threshold based on existing FP dice)
            hunt_params = get_hunt_params(self.state)

            # Move fellowship with skip_hunt=True so we handle hunt interactively.
            # skip_auto_sacrifice=False is EXPLICIT (auto-resolver purge
            # 2026-08-16, defaults inverted): the legacy manual GUI has no
            # pending-decision loop, and its MORDOR branch relies on
            # move_fellowship's internal tile resolution (incl. the
            # auto-sacrifice policy) - outside Mordor skip_hunt=True keeps
            # the internal roll (and thus damage) at zero, so the flag
            # only bites on the Mordor track, exactly where this GUI
            # needs it.
            mv = move_fellowship(self.state, skip_hunt=True,
                                 skip_auto_sacrifice=False)
            if not mv.get("success", False):
                self.log(f"Fellowship move failed: {mv.get('message', '')}")
                self.refresh()
                return
            self.log(f"Fellowship moved. Progress: {self.state.fellowship.progress}")

            # Add character die to hunt box AFTER move (per rules).
            # 'The Last Battle' persistent card: FP movement dice bypass
            # the box (mirrors the engine executor's guard).
            from persistent_effects import should_add_dice_to_hunt_box
            die = self.state.dice[self.current_side][self.selected_die_idx]
            if should_add_dice_to_hunt_box(self.state) \
                    and die not in self.state.hunt_box:
                self.state.hunt_box.append(die)
            self._use_selected_die()

            # In Mordor the engine resolves the hunt tile inside
            # move_fellowship; rolling again here would double the hunt.
            if self.state.fellowship.mordor_entered:
                self.log(f"  {mv.get('message', '')}")
                self.refresh()
                return

            # Use pre-calculated hunt parameters
            threshold = hunt_params["threshold"]
            num_dice = hunt_params["num_dice"]
            fp_dice = hunt_params["fp_dice_in_box"]
            fp_mod = hunt_params["fp_modifier"]
            reroll_sources = hunt_params["reroll_sources"]

            self.log(f"Hunt: threshold {threshold}+, {num_dice} dice (hunt level {hunt_params['hunt_level']}), FP modifier +{fp_mod}")

            if num_dice == 0:
                self.log("Hunt: No dice in hunt box - no hunt roll")
                self.refresh()
                return

            # Check for Flocks of Crebain (adds to FP modifier for total modifier)
            hunt_dice_modifier = fp_mod
            if self._has_persistent_card("Flocks of Crebain"):
                if messagebox.askyesno(
                    "Flocks of Crebain",
                    "Shadow: Discard Flocks of Crebain to add +1 to all Hunt dice?\n\n"
                    "(This includes re-rolls)",
                    parent=self.root
                ):
                    hunt_dice_modifier += 1
                    self._remove_persistent_card("Flocks of Crebain")
                    self.log("  Flocks of Crebain used: +1 to all Hunt dice")

            # Roll hunt dice (fp_modifier handles the FP dice bonus;
            # Crebain is additional on top)
            roll_result = roll_hunt_dice(num_dice, threshold, fp_modifier=hunt_dice_modifier)
            rolls = roll_result["rolls"]
            modified_rolls = roll_result["modified"]
            hits = roll_result["hits"]

            if hunt_dice_modifier > 0:
                self.log(f"Hunt roll: {rolls} (+{hunt_dice_modifier} = {modified_rolls}) (threshold {threshold}+, nat 1 misses) = {hits} hit(s)")
            else:
                self.log(f"Hunt roll: {rolls} (threshold {threshold}+) = {hits} hit(s)")

            # Offer re-rolls if sources available and there are misses
            misses_count = num_dice - hits
            if reroll_sources and misses_count > 0:
                for src in reroll_sources:
                    self.log(f"  Re-roll available: {src}")

                rerolls_available = min(len(reroll_sources), misses_count)
                if messagebox.askyesno(
                    "Hunt Re-rolls",
                    f"Shadow has {rerolls_available} re-roll(s) available.\n"
                    f"Hits so far: {hits}/{num_dice}\n\nRe-roll misses?",
                    parent=self.root
                ):
                    reroll_result = reroll_hunt_dice(rerolls_available, threshold,
                                                     fp_modifier=hunt_dice_modifier)
                    reroll_rolls = reroll_result["rolls"]
                    modified_rerolls = reroll_result["modified"]
                    new_hits = reroll_result["hits"]
                    hits += new_hits
                    if hunt_dice_modifier > 0:
                        self.log(f"  Re-roll: {reroll_rolls} (+{hunt_dice_modifier} = {modified_rerolls}) = {new_hits} additional hit(s)")
                    else:
                        self.log(f"  Re-roll: {reroll_rolls} = {new_hits} additional hit(s)")

            # If hit, draw hunt tile
            if hits > 0:
                self._draw_hunt_tile(hits)
            else:
                self.log("Hunt: No hits - fellowship safe")

        except Exception as e:
            self.log(f"FSP move failed: {e}")
            import traceback
            traceback.print_exc()

        self.refresh()

    def _draw_hunt_tile(self, hunt_hits: int):
        """Draw a hunt tile and resolve its effect."""
        from game_logic import draw_hunt_tile, remove_persistent_card

        if not self.state.hunt_pool:
            self.log("Hunt pool empty!")
            return

        # Check for Wizard's Staff first
        if self._has_persistent_card("Wizard's Staff"):
            if messagebox.askyesno(
                "Wizard's Staff",
                "Use Wizard's Staff to cancel the hunt tile draw?",
                parent=self.root
            ):
                self.log("Wizard's Staff used - tile draw cancelled!")
                remove_persistent_card(self.state, "Wizard's Staff")
                return

        # Draw tile using game_logic (handles randomness)
        result = draw_hunt_tile(self.state, hunt_hits)

        if not result["success"]:
            self.log(result.get("message", "Failed to draw tile"))
            return

        tile = result["tile"]
        damage = result["damage"]
        reveal = result["reveal"]

        self.log(f"Hunt tile drawn: {result['tile_name']}")
        if result["is_eye"]:
            self.log(f"  Eye tile: {damage} damage (= hunt hits)")
        else:
            self.log(f"  Damage: {damage}")

        if reveal:
            self.log("  Fellowship must be revealed!")

        # Show hunt resolution dialog.  draw_hunt_tile already filed the
        # tile into hunt_discard (drawn = used pile); a Mithril redraw
        # moves it back to the pool inside the dialog.
        self._resolve_hunt_damage(damage, reveal, tile, hunt_hits)

    def _has_persistent_card(self, card_name: str) -> bool:
        """Check if a persistent card is in play.  state.persistent_cards
        is always a PersistentCardState (typed field with a default
        factory); no dict format exists."""
        return self.state.persistent_cards.has_card(card_name)

    def _remove_persistent_card(self, card_name: str):
        """Remove a persistent card from play."""
        from game_logic import remove_persistent_card

        result = remove_persistent_card(self.state, card_name)
        if result["success"]:
            self.log(f"  {card_name} discarded from table")

    def _check_shadow_movement_blocked(self, region: str) -> tuple:
        """
        Check if Shadow movement to a region is blocked by a persistent card.
        Returns (is_blocked, blocking_card_name).
        """
        from persistent_effects import can_shadow_move_to
        ok, blocker = can_shadow_move_to(self.state, region)
        return (not ok), blocker

    def _check_palantir_card_draw(self, played_deck=None):
        """
        The Palantír of Orthanc: each time Shadow plays a card, draw a
        card of the SAME DECK as the one played (USER RULE 2026-07-08 —
        no deck prompt).
        """
        from persistent_effects import should_draw_extra_card_on_event

        if not should_draw_extra_card_on_event(self.state):
            return

        from game_logic import draw_event_card
        deck = str(played_deck).lower() if played_deck else "character"
        if deck not in ("character", "strategy"):
            deck = "character"

        try:
            draw_event_card(self.state, Side.SHADOW, deck)
            self.log(f"  Palantír bonus: Drew {deck.title()} card")
            # Sync to GUI hand
            self._sync_hands()
        except RuntimeError as e:
            self.log(f"  Palantír bonus: Could not draw ({e})")

    def _check_gandalf_card_draw(self, die_face, played_deck=None):
        """
        Gandalf the Grey's guide ability (USER RULE 2026-07-08): when FP
        plays a card with an Event or Will of the West die, draw a card
        of the SAME DECK as the one just played — no deck prompt.
        """
        # Check if Gandalf the Grey is the guide
        guide = self.state.fellowship.guide
        if not guide or guide.name != "Gandalf the Grey":
            return

        # Check if the die was Event or Will of the West
        if die_face not in (ActionDieFace.EVENT, ActionDieFace.WILL_OF_WEST):
            return

        from game_logic import draw_event_card
        deck = str(played_deck).lower() if played_deck else "character"
        if deck not in ("character", "strategy"):
            deck = "character"

        try:
            draw_event_card(self.state, Side.FREE_PEOPLES, deck)
            self.log(f"  Gandalf's Wisdom: Drew {deck.title()} card")
            # Sync to GUI hand
            self._sync_hands()
        except RuntimeError as e:
            self.log(f"  Gandalf's Wisdom: Could not draw ({e})")

    def _sync_hands(self):
        """Synchronize GUI hands with game state hands."""
        for side in [Side.FREE_PEOPLES, Side.SHADOW]:
            self.hands[side] = list(self.state.hands[side].hand)

    def _resolve_hunt_damage(self, damage: int, reveal: bool, tile,
                             hunt_hits: int = 0) -> bool:
        """Show dialog for FP to resolve hunt damage.  Returns True when
        Mithril Coat replaced the tile (it went back to the pool, so the
        caller must not file it into a used pile)."""
        if damage == 0 and not reveal:
            self.log("  No damage to resolve")
            return False

        dialog = tk.Toplevel(self.root)
        dialog.title("Hunt Damage Resolution")
        dialog.configure(bg=COLORS["panel_bg"])
        dialog.geometry("450x500")
        dialog.transient(self.root)
        dialog.grab_set()

        remaining_damage = [damage]  # Use list for mutable reference
        reveal_flag = [reveal]       # Gollum's guidance can force a reveal
        tile_replaced = [False]      # Mithril Coat drew a replacement tile

        # Header
        tk.Label(
            dialog, text=f"Hunt Damage: {damage}",
            bg=COLORS["panel_bg"], fg=COLORS["error"],
            font=("Arial", 14, "bold")
        ).pack(pady=10)

        status_var = tk.StringVar(value=f"Remaining damage: {damage}")
        status_label = tk.Label(dialog, textvariable=status_var,
                               bg=COLORS["panel_bg"], fg=COLORS["highlight"])
        status_label.pack(pady=5)

        def update_status():
            status_var.set(f"Remaining damage: {remaining_damage[0]}")

        options_frame = tk.Frame(dialog, bg=COLORS["panel_bg"])
        options_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Mithril Coat option (draw another tile)
        if self._has_persistent_card("Mithril Coat and Sting"):
            def use_mithril():
                self.log("  Mithril Coat used - drawing replacement tile")
                # Put current tile back (it was filed into hunt_discard
                # at draw time — conservation: one pile at a time)
                tile_replaced[0] = True
                if tile in self.state.hunt_discard:
                    self.state.hunt_discard.remove(tile)
                self.state.hunt_pool.append(tile)
                self._remove_persistent_card("Mithril Coat and Sting")
                dialog.destroy()
                # Re-trigger with the ORIGINAL hunt hits (an Eye
                # replacement tile's damage = hunt hits, not this
                # tile's possibly-reduced damage).
                self._draw_hunt_tile(hunt_hits)

            self._make_button(options_frame, "Use Mithril Coat (draw new tile)",
                            use_mithril, width=35).pack(pady=3)

        # Axe and Bow (-1 damage)
        if self._has_persistent_card("Axe and Bow") and remaining_damage[0] > 0:
            def use_axe_bow():
                remaining_damage[0] = max(0, remaining_damage[0] - 1)
                self.log("  Axe and Bow: -1 damage")
                self._remove_persistent_card("Axe and Bow")
                update_status()

            self._make_button(options_frame, "Use Axe and Bow (-1 damage)",
                            use_axe_bow, width=35).pack(pady=3)

        # Horn of Gondor (-1 damage)
        if self._has_persistent_card("Horn of Gondor") and remaining_damage[0] > 0:
            def use_horn():
                remaining_damage[0] = max(0, remaining_damage[0] - 1)
                self.log("  Horn of Gondor: -1 damage")
                self._remove_persistent_card("Horn of Gondor")
                update_status()

            self._make_button(options_frame, "Use Horn of Gondor (-1 damage)",
                            use_horn, width=35).pack(pady=3)

        # Sacrifice guide (absorbs the guide's LEVEL, not all damage;
        # engine twin: _apply_fellowship_damage sacrifice_guide=True)
        guide = self.state.fellowship.guide
        if guide and guide.name != "Gollum" and remaining_damage[0] > 0:
            def sacrifice_guide():
                from game_logic import _on_companion_casualty, change_guide
                guide_name = guide.name
                absorbed = getattr(guide, 'level', 1)
                self.state.fellowship.companions.remove(guide)
                # USER 2026-08-15: hobbit guide = separation, no WWSAT
                _on_companion_casualty(
                    self.state, guide_name,
                    casualty=guide_name not in ('Meriadoc', 'Peregrin'))
                try:
                    change_guide(self.state, allow_pending=True)
                except Exception as e:
                    self.log(f"  Guide reassignment failed: {e}")
                self.log(f"  {guide_name} sacrificed as casualty "
                         f"(absorbs {absorbed})")
                remaining_damage[0] = max(0, remaining_damage[0] - absorbed)
                update_status()

            self._make_button(options_frame, f"Sacrifice Guide ({guide.name})",
                            sacrifice_guide, width=35).pack(pady=3)

        # Sacrifice random companion (absorbs the victim's LEVEL)
        companions = [c for c in self.state.fellowship.companions if c.name != "Gollum"]
        if companions and remaining_damage[0] > 0:
            def sacrifice_random():
                from game_logic import sacrifice_random_companion
                levels = {c.name: getattr(c, 'level', 1)
                          for c in self.state.fellowship.companions}
                result = sacrifice_random_companion(self.state)
                if result["success"]:
                    self.log(f"  {result['message']}")
                    absorbed = levels.get(result.get("victim"), 1)
                    remaining_damage[0] = max(0,
                                              remaining_damage[0] - absorbed)
                    update_status()

            self._make_button(options_frame, "Sacrifice Random Companion",
                            sacrifice_random, width=35).pack(pady=3)

        # Gollum's ability (if guide and not eye tile): single use,
        # and using it reveals the Fellowship.
        if guide and guide.name == "Gollum" and not tile.is_eye:
            _gollum_used = [False]

            def use_gollum():
                if _gollum_used[0]:
                    return
                _gollum_used[0] = True
                remaining_damage[0] = max(0, remaining_damage[0] - 1)
                reveal_flag[0] = True
                self.log("  Gollum guides safely: -1 damage, reveals")
                update_status()

            self._make_button(options_frame, "Gollum's Guidance (-1, reveals)",
                            use_gollum, width=35).pack(pady=3)

        # Take corruption damage
        def take_damage():
            self.state.fellowship.corruption += remaining_damage[0]
            self.log(f"  Fellowship takes {remaining_damage[0]} corruption (total: {self.state.fellowship.corruption})")
            if self.state.fellowship.corruption >= 12:
                messagebox.showwarning("Corruption!", "Fellowship has 12+ corruption!\nRing-bearers corrupted - Shadow wins!")
            remaining_damage[0] = 0
            update_status()

        self._make_button(options_frame, "Take Corruption Damage (remaining)",
                        take_damage, width=35).pack(pady=3)

        # Handle reveal warning
        if reveal:
            progress = self.state.fellowship.progress
            tk.Label(options_frame, text=f"Fellowship MUST be revealed!\n(Must move up to {progress} steps, cannot end in FP stronghold/city)",
                    bg=COLORS["panel_bg"], fg=COLORS["error"],
                    font=("Arial", 10, "bold"), wraplength=400).pack(pady=10)

        # Done button
        def done():
            # Apply any remaining damage as corruption
            if remaining_damage[0] > 0:
                self.state.fellowship.corruption += remaining_damage[0]
                self.log(f"  Remaining {remaining_damage[0]} taken as corruption")

            dialog.destroy()

            # Handle reveal - show destination dialog
            if reveal_flag[0]:
                self._handle_reveal_destination()

        self._make_button(options_frame, "Done", done, bg=COLORS["success"],
                        width=15).pack(pady=20)

        dialog.wait_window()
        return tile_replaced[0]

    def _handle_reveal_destination(self):
        """Handle fellowship reveal - must choose destination and reset progress."""
        from game_logic import fellowship_region
        from board import regions as board_regions

        progress = self.state.fellowship.progress
        current = fellowship_region(self.state)

        self.log(f"  Fellowship revealed! Must move up to {progress} step(s)")

        if progress == 0:
            # No progress, just reveal in place
            self.state.fellowship.hidden = False
            self.log(f"  Fellowship revealed in {current} (no progress to move)")
            return

        # Calculate reachable regions (cannot end in FP cities/strongholds when revealed)
        reachable = self._get_reveal_reachable_regions(current, progress)

        if not reachable:
            # Shouldn't happen, but fallback to staying in place
            self.state.fellowship.hidden = False
            self.state.fellowship.progress = 0
            self.log(f"  No valid destinations - revealed in {current}")
            return

        # Show destination dialog
        dest = self._show_destination_dialog(
            "Reveal Destination",
            f"Fellowship revealed! Select destination (up to {progress} steps):\n"
            "(Cannot end in FP stronghold/city when revealed)",
            sorted(reachable)
        )

        if not dest:
            # User cancelled - pick first available
            dest = sorted(reachable)[0]
            self.log(f"  No selection - defaulting to {dest}")

        try:
            # Build path to destination (simple: just set destination)
            # For proper path, would need pathfinding
            self.state.fellowship.region = self.board.region(sanitize(dest))
            self.state.fellowship.progress = 0
            self.state.fellowship.hidden = False

            self.log(f"  Fellowship revealed in {dest}")

            # Check for shadow stronghold (extra tile).  The stronghold
            # must still be SHADOW-controlled (not FP-captured).
            from game_logic import is_shadow_region
            _dest_san = sanitize(dest)
            _ec = getattr(self.state, 'enemy_controlled', {}) or {}
            dest_meta = board_regions.get(_dest_san, {})
            if (dest_meta.get("is_stronghold") and is_shadow_region(dest)
                    and _dest_san not in _ec.get(Side.FREE_PEOPLES, set())):
                self.log("  Revealed in Shadow stronghold - drawing extra tile!")
                self._draw_extra_shadow_stronghold_tile()

        except Exception as e:
            self.log(f"  Reveal failed: {e}")
            # Fallback - just reveal in place
            self.state.fellowship.hidden = False
            self.state.fellowship.progress = 0

    def _get_reveal_reachable_regions(self, start: str, max_steps: int) -> set:
        """
        Get regions reachable for fellowship reveal.
        When revealed, fellowship CANNOT end in FP cities/strongholds.
        """
        from board import regions as board_regions
        from game_logic import is_shadow_region
        from collections import deque

        start_san = sanitize(start)
        reachable = set()
        visited = {start_san: 0}
        queue = deque([(start_san, 0)])

        while queue:
            current, steps = queue.popleft()

            if steps > 0:
                # Valid reveal destination = not an FP-CONTROLLED
                # city/stronghold.  Control, not native faction: a
                # Shadow-captured FP city is legal, an FP-captured
                # Shadow settlement is not.
                ec = getattr(self.state, 'enemy_controlled', {}) or {}
                meta = board_regions.get(current, {})
                is_settlement = meta.get("is_city") or meta.get("is_stronghold")
                fp_controlled = ((not is_shadow_region(current)
                                  and current not in ec.get(Side.SHADOW, set()))
                                 or current in ec.get(Side.FREE_PEOPLES, set()))
                if not (is_settlement and fp_controlled):
                    reachable.add(current)

            if steps >= max_steps:
                continue

            for neighbor in self.board.adjacent(current):
                neighbor_san = sanitize(neighbor)
                if neighbor_san not in visited or visited[neighbor_san] > steps + 1:
                    visited[neighbor_san] = steps + 1
                    queue.append((neighbor_san, steps + 1))

        return reachable

    def _draw_extra_shadow_stronghold_tile(self):
        """Draw extra hunt tile when revealed in/passing through shadow stronghold."""
        from game_logic import draw_extra_stronghold_tile
        result = draw_extra_stronghold_tile(self.state)

        self.log(f"  {result['message']}")
        if result["success"] and result["damage"] > 0:
            self.log(f"  Corruption now: {self.state.fellowship.corruption}")

    def _on_fsp_hide(self):
        """Hide the Fellowship. Strider as guide allows hiding with any die."""
        if self.ai_playing:
            self.log('Wait for AI to finish its turn')
            return
        if self.current_side != Side.FREE_PEOPLES:
            self.log("Only Free Peoples can hide the Fellowship")
            return

        # Check if Strider/Aragorn is the guide (allows any die)
        guide = self.state.fellowship.guide
        strider_is_guide = guide and guide.name in ("Strider", "Aragorn")

        if strider_is_guide:
            # Strider's ability: can hide with any die
            if self.selected_die_idx is None:
                self.log("Select any die to hide (Strider's ability)")
                return
            dice = self.state.dice.get(self.current_side, [])
            if self.selected_die_idx >= len(dice) or dice[self.selected_die_idx].used:
                self.log("Select an unused die")
                return
        else:
            # Normal: requires Character or Will of the West
            valid = (ActionDieFace.CHARACTER, ActionDieFace.WILL_OF_WEST)
            if not self._check_die_selected(valid):
                return

        try:
            from game_logic import hide_fellowship
            hide_fellowship(self.state)
            self._use_selected_die()
            if strider_is_guide:
                self.log("Fellowship hidden (Strider's ability - any die)")
            else:
                self.log("Fellowship hidden")
        except Exception as e:
            self.log(f"FSP hide failed: {e}")

        self.refresh()

    def _on_move_nazgul(self):
        """
        Move Nazgûl (and Mouth of Sauron) independently with 1 character die.
        - Nazgûl can move anywhere (unless FP army there)
        - Witch King has infinite movement
        - Mouth of Sauron moves 3 spaces (subject to companion rules - stops at enemy strongholds)
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

        # Find all Nazgûl and Mouth of Sauron on board
        nazgul_on_board = []  # (region, nazgul_obj)
        mouth_on_board = None  # (region, minion_obj)

        from game_logic import Minion

        for region_key, army in self.state.armies.items():
            for leader in army.leaders:
                if isinstance(leader, Nazgul):
                    nazgul_on_board.append((region_key, leader))
                elif isinstance(leader, Minion) and leader.name == "Mouth of Sauron":
                    mouth_on_board = (region_key, leader)

        if not nazgul_on_board and not mouth_on_board:
            self.log("No Nazgûl or Mouth of Sauron on board")
            return

        self._show_nazgul_movement_dialog(nazgul_on_board, mouth_on_board)

    def _show_nazgul_movement_dialog(self, nazgul_on_board, mouth_on_board):
        """Dialog to move Nazgûl and Mouth of Sauron independently."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Move Nazgûl")
        dialog.configure(bg=COLORS["panel_bg"])
        dialog.geometry("550x600")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(
            dialog, text="Move Nazgûl & Minions",
            bg=COLORS["panel_bg"], fg=COLORS["highlight"],
            font=("Arial", 12, "bold")
        ).pack(pady=10)

        tk.Label(
            dialog, text="1 Character die moves ALL Nazgûl independently.\nNazgûl can go anywhere (no FP armies).\nMouth of Sauron: 3 spaces, stops at enemy strongholds.",
            bg=COLORS["panel_bg"], fg=COLORS["text"], justify=tk.LEFT
        ).pack(pady=5)

        # Scrollable frame
        canvas = tk.Canvas(dialog, bg=COLORS["panel_bg"], highlightthickness=0)
        scrollbar = tk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=COLORS["panel_bg"])

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

        # Track destinations
        destinations = {}
        dest_labels = {}
        all_movers = []  # (index, region, obj, name, max_move)

        idx = 0
        for region, naz in nazgul_on_board:
            all_movers.append((idx, region, naz, "Nazgûl", 99))  # Infinite movement
            idx += 1

        if mouth_on_board:
            region, mouth = mouth_on_board
            all_movers.append((idx, region, mouth, "Mouth of Sauron", 3))

        def get_nazgul_reachable(start_region, max_move, is_mouth=False):
            """Get regions Nazgûl/Mouth can reach."""
            from collections import deque
            from board import adj, regions as board_regions

            start_san = sanitize(start_region)
            reachable = set()
            visited = {start_san: 0}
            queue = deque([(start_san, 0)])

            while queue:
                current, steps = queue.popleft()

                if current != start_san:
                    # Check if FP army present (Nazgûl can't go there)
                    has_fp_army = False
                    if current in self.state.armies:
                        army = self.state.armies[current]
                        for unit in army.units:
                            if unit.nation in [Nation.ELVES, Nation.DWARVES, Nation.GONDOR,
                                              Nation.THE_NORTH, Nation.ROHAN]:
                                has_fp_army = True
                                break
                    if not has_fp_army:
                        reachable.add(current)

                if steps >= max_move:
                    continue

                for neighbor in adj.get(current, []):
                    if neighbor not in visited or visited[neighbor] > steps + 1:
                        # Mouth of Sauron stops on ENTERING an enemy
                        # (FP-CONTROLLED) stronghold: native FP and not
                        # Shadow-captured, or Shadow-native but
                        # FP-captured.  Never passes through one.
                        if is_mouth:
                            from game_logic import is_shadow_region
                            info = board_regions.get(neighbor, {})
                            ec = getattr(self.state, 'enemy_controlled',
                                         {}) or {}
                            fp_ctrl = ((not is_shadow_region(neighbor)
                                        and neighbor not in ec.get(
                                            Side.SHADOW, set()))
                                       or neighbor in ec.get(
                                           Side.FREE_PEOPLES, set()))
                            if info.get("is_stronghold") and fp_ctrl:
                                visited[neighbor] = steps + 1
                                reachable.add(neighbor)  # Can stop here
                                continue  # But can't continue past
                        visited[neighbor] = steps + 1
                        queue.append((neighbor, steps + 1))

            return reachable

        def select_destination(idx, region, obj, name, max_move):
            is_mouth = name == "Mouth of Sauron"
            reachable = get_nazgul_reachable(region, max_move, is_mouth)

            # Show selection dialog
            dest_dialog = tk.Toplevel(dialog)
            dest_dialog.title(f"Move {name}")
            dest_dialog.configure(bg=COLORS["panel_bg"])
            dest_dialog.geometry("300x400")
            dest_dialog.transient(dialog)
            dest_dialog.grab_set()

            tk.Label(dest_dialog, text=f"Move {name} from {region}",
                    bg=COLORS["panel_bg"], fg=COLORS["text"]).pack(pady=10)

            listbox = tk.Listbox(dest_dialog, bg=COLORS["bg"], fg=COLORS["text"], height=15)
            for r in sorted(reachable):
                listbox.insert(tk.END, r)
            listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

            def select():
                sel = listbox.curselection()
                if sel:
                    dest = listbox.get(sel[0])
                    destinations[idx] = dest
                    dest_labels[idx].config(text=f"→ {dest}", fg=COLORS["success"])
                dest_dialog.destroy()

            def cancel():
                dest_dialog.destroy()

            btn_frame = tk.Frame(dest_dialog, bg=COLORS["panel_bg"])
            btn_frame.pack(fill=tk.X, padx=10, pady=10)
            self._make_button(btn_frame, "Select", select, bg=COLORS["success"]).pack(side=tk.LEFT, padx=5)
            self._make_button(btn_frame, "Cancel", cancel).pack(side=tk.LEFT, padx=5)

            dest_dialog.wait_window()

        def clear_destination(idx):
            if idx in destinations:
                del destinations[idx]
            dest_labels[idx].config(text="(staying)", fg=COLORS["text_dim"])

        # Create row for each mover
        for idx, region, obj, name, max_move in all_movers:
            row_frame = tk.Frame(scroll_frame, bg=COLORS["bg"], relief=tk.RAISED, bd=1)
            row_frame.pack(fill=tk.X, padx=5, pady=3)

            info_frame = tk.Frame(row_frame, bg=COLORS["bg"])
            info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)

            tk.Label(
                info_frame, text=name,
                bg=COLORS["bg"], fg=COLORS["shadow_primary"],
                font=("Arial", 10, "bold")
            ).pack(anchor="w")

            tk.Label(
                info_frame, text=f"Currently in: {region}",
                bg=COLORS["bg"], fg=COLORS["text"]
            ).pack(anchor="w")

            dest_label = tk.Label(
                info_frame, text="(staying)",
                bg=COLORS["bg"], fg=COLORS["text_dim"]
            )
            dest_label.pack(anchor="w")
            dest_labels[idx] = dest_label

            btn_frame = tk.Frame(row_frame, bg=COLORS["bg"])
            btn_frame.pack(side=tk.RIGHT, padx=5, pady=5)

            self._make_button(
                btn_frame, "Move",
                lambda i=idx, r=region, o=obj, n=name, m=max_move: select_destination(i, r, o, n, m),
                bg=COLORS["button_bg"]
            ).pack(side=tk.LEFT, padx=2)

            self._make_button(
                btn_frame, "Clear",
                lambda i=idx: clear_destination(i),
                bg=COLORS["text_dim"]
            ).pack(side=tk.LEFT, padx=2)

        result = {"done": False, "moved": False}

        def execute_moves():
            if not destinations:
                messagebox.showwarning("No Moves", "Select at least one destination")
                return

            moved_items = []
            for idx, dest in destinations.items():
                _, region, obj, name, _ = all_movers[idx]

                # Remove from current army (drop the entry when emptied,
                # matching the engine's mover executors)
                _src_army = self.state.armies[region]
                _src_army.leaders.remove(obj)
                if not _src_army.units and not _src_army.leaders:
                    del self.state.armies[region]

                # Add to destination
                dest_san = sanitize(dest)
                # Armies ONLY via get_or_create_army: raw exact-key
                # creation forges duplicates beside original-name keys.
                from game_logic import get_or_create_army
                get_or_create_army(self.state, dest_san).leaders.append(obj)

                moved_items.append(f"{name} → {dest}")

            self._use_selected_die()
            self.log(f"Moved: {', '.join(moved_items)}")
            result["moved"] = True
            result["done"] = True
            dialog.destroy()

        def cancel():
            result["done"] = True
            dialog.destroy()

        btn_frame = tk.Frame(dialog, bg=COLORS["panel_bg"])
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        self._make_button(btn_frame, "Execute All Moves", execute_moves, bg=COLORS["success"]).pack(side=tk.LEFT, padx=5)
        self._make_button(btn_frame, "Cancel", cancel).pack(side=tk.LEFT, padx=5)

        dialog.wait_window()
        if result["moved"]:
            self.refresh()

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

        try:
            # Route through the engine (fellowship_declare is a free
            # action, die_idx=-1) so the Mordor-entry check + rules match
            # self-play/teacher exactly.
            from ai import Action, _execute_action
            _execute_action(self.state, Action(
                die_idx=-1, action_type='fellowship_declare',
                params={'region': sanitize(dest)}))
            self.log(f"Fellowship declared in {dest}")
            meta = board_regions.get(sanitize(dest), {})
            is_settlement = meta.get("is_city") or meta.get("is_stronghold")
            if is_settlement and not is_shadow_region(dest):
                self.log(f"  Healed 1 corruption (now {self.state.fellowship.corruption})")
        except Exception as e:
            self.log(f"Declaration failed: {e}")

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

        # Show dialog to select dice to allocate as eyes
        dialog = tk.Toplevel(self.root)
        dialog.title("Allocate Eyes (Pre-Roll)")
        dialog.configure(bg=COLORS["panel_bg"])
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.geometry("400x400")

        tk.Label(
            dialog, text="Select Shadow dice to allocate as Eyes\n(before rolling)",
            bg=COLORS["panel_bg"], fg=COLORS["text"],
            font=("Arial", 11)
        ).pack(pady=10)

        # Show checkboxes for each unused Shadow die
        check_vars = []
        checks_frame = tk.Frame(dialog, bg=COLORS["panel_bg"])
        checks_frame.pack(pady=10)

        for i, die in enumerate(self.state.dice[Side.SHADOW]):
            if not die.used:
                var = tk.BooleanVar(value=False)
                check_vars.append((i, var))
                tk.Checkbutton(
                    checks_frame, text=f"Die {i+1}",
                    variable=var, bg=COLORS["panel_bg"], fg=COLORS["text"],
                    selectcolor=COLORS["shadow_secondary"]
                ).pack(anchor=tk.W)

        def confirm():
            indices = [i for i, var in check_vars if var.get()]
            if indices:
                from game_logic import allocate_eyes
                try:
                    allocate_eyes(self.state, len(indices), indices=indices)
                    self.log(f"Pre-allocated {len(indices)} eye(s) to hunt box")
                except RuntimeError as e:
                    self.log(f"Allocation failed: {e}")
            dialog.destroy()
            self.refresh()

        btn_frame = tk.Frame(dialog, bg=COLORS["panel_bg"])
        btn_frame.pack(pady=20)
        self._make_button(btn_frame, "Allocate", confirm).pack(side=tk.LEFT, padx=5)
        self._make_button(btn_frame, "Cancel", dialog.destroy).pack(side=tk.LEFT, padx=5)

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
        """Change the Fellowship guide."""
        if self.current_side != Side.FREE_PEOPLES:
            self.log("Only Free Peoples can change guide")
            return

        companions = self.state.fellowship.companions
        if not companions:
            self.log("No companions in Fellowship")
            return

        # Find highest level
        max_level = max(c.level for c in companions)
        eligible = [c for c in companions if c.level == max_level]

        if len(eligible) == 1:
            # Only one choice
            new_guide = eligible[0]
            self.state.fellowship.guide = new_guide
            self.log(f"Guide changed to {new_guide.name}")
            self.refresh()
            return

        # Multiple choices - show dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Change Guide")
        dialog.configure(bg=COLORS["panel_bg"])
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.geometry("300x300")

        tk.Label(
            dialog, text=f"Select new guide (Level {max_level}):",
            bg=COLORS["panel_bg"], fg=COLORS["text"],
            font=("Arial", 12, "bold")
        ).pack(pady=10)

        current_guide = self.state.fellowship.guide

        for comp in eligible:
            def select(c=comp):
                self.state.fellowship.guide = c
                self.log(f"Guide changed to {c.name}")
                dialog.destroy()
                self.refresh()

            text = f"{comp.name}"
            if comp == current_guide:
                text += " (current)"

            self._make_button(dialog, text, select, width=25).pack(pady=5)

        self._make_button(dialog, "Cancel", dialog.destroy).pack(pady=10)

    def _on_end_turn(self):
        """End the current turn."""
        from game_logic import next_turn

        # Use game_logic's next_turn for proper state management
        next_turn(self.state)
        # next_turn drew for both sides; latch so Draw Cards refuses a
        # duplicate draw this turn.
        self._cards_drawn_turn = self.state.turn

        self.selected_die_idx = None
        self.pending_action = None
        self.action_source = None

        # Sync hands after card draw
        self._sync_hands()

        self.log(f"=== Turn {self.state.turn} ===")
        self.log("  Cards drawn for both sides")
        self.refresh()

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
        """Newest current-generation checkpoint pair (jt > td > plain
        v30) — the old wotr_*_value*.pt names died with the 2026-07
        encoding generations (stale-path fix 2026-07-27)."""
        import os
        project_root = os.path.dirname(BASE_DIR)
        for sfx in ('v31jt', 'v31td', 'v31'):
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

    def _selfplay_next_turn(self):
        """Start a new turn: eye allocation, dice roll, then actions."""
        if not self.selfplay_running:
            return
        from ai import _safe_next_turn, _ai_eye_allocation
        import game_logic

        if self.selfplay_turn > 0:
            _safe_next_turn(self.state)

        self.selfplay_turn += 1
        if self.selfplay_turn > 25:
            self.log("Game reached turn limit (25)")
            self.selfplay_running = False
            return

        self.log(f"\n{'='*40}", "turn")
        self.log(f"=== TURN {self.selfplay_turn} ===", "turn")
        self.log(f"{'='*40}", "turn")

        # Phase 2: Fellowship Phase — FP may declare fellowship position
        from ai import _get_declare_actions, mcts_action, _execute_action, Action
        if self.state.fellowship.progress > 0 and not self.state.fellowship.mordor_entered:
            declare_actions = _get_declare_actions(self.state)
            if declare_actions:
                declare_actions.append(Action(die_idx=-1, action_type='pass', params={}))
                dec = mcts_action(self.state, Side.FREE_PEOPLES, self.fp_net, self.sh_net,
                                  num_simulations=100, temperature=0.1)
                if dec and dec.action_type == 'fellowship_declare':
                    region = dec.params.get('region', '?')
                    self.log(f"FP [Fellowship Phase]: Declares fellowship at {region}", "fp")
                    try:
                        _execute_action(self.state, dec)
                    except Exception as e:
                        self.log(f"  Declaration failed: {e}")

        # Eye allocation (always uses Shadow net)
        _ai_eye_allocation(self.state, self.sh_net)
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

        from ai import get_legal_actions, _execute_action, mcts_action
        import game_logic

        active = self.selfplay_active_side
        other = Side.SHADOW if active == Side.FREE_PEOPLES else Side.FREE_PEOPLES

        fp_left = self._count_unused_dice(Side.FREE_PEOPLES)
        sh_left = self._count_unused_dice(Side.SHADOW)

        # If both sides out of dice, next turn
        if fp_left == 0 and sh_left == 0:
            # End-of-turn Victory Check phase (rulebook p.17, ruling
            # 1.212a): military victory (Shadow-first tiebreaker) is
            # resolved here, after the turn's actions complete.
            _end_vc = game_logic.check_all_victory_conditions(self.state)
            if game_logic.should_game_end(_end_vc, end_of_turn=True):
                self.log(f"\n*** GAME OVER: {_end_vc['victory']} wins by "
                         f"{_end_vc.get('victory_type', '')}! ***")
                self.selfplay_running = False
                self._sync_hands()
                self.refresh()
                return
            if self.selfplay_paused:
                # In step mode, go to next turn immediately on this step
                self._selfplay_next_turn()
            else:
                self.root.after(self.selfplay_speed, self._selfplay_next_turn)
            return

        # Skip side if no dice left (transparent, don't consume a step)
        my_left = self._count_unused_dice(active)
        if my_left == 0:
            self.selfplay_active_side = other
            self.root.after(50, self._selfplay_one_action)
            return

        actions = get_legal_actions(self.state, active)
        if not actions:
            self.selfplay_active_side = other
            self.root.after(50, self._selfplay_one_action)
            return

        # Use MCTS for strongest play (200 sims)
        action = mcts_action(self.state, active, self.fp_net, self.sh_net,
                             num_simulations=200, temperature=0.1)
        if action is None:
            self.selfplay_active_side = other
            self.root.after(50, self._selfplay_one_action)
            return

        # Pass guard (mirrors _run_single_episode's pass gate)
        opp_left = self._count_unused_dice(other)
        is_pending = (self.state.pending_army_move_die is not None or
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

        try:
            _execute_action(self.state, action)
        except Exception as e:
            self.log(f"{side_label}: Action failed ({e})")

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
            burst_action = mcts_action(self.state, burst_side,
                                       self.fp_net, self.sh_net,
                                       num_simulations=100, temperature=0.1)
            if burst_action is None:
                burst_action = burst_actions[0]
            bl = "FP" if burst_side == Side.FREE_PEOPLES else "Shadow"
            self.log(f"  {bl} [free]: {self._describe_action(burst_action)}")
            try:
                _execute_action(self.state, burst_action)
            except Exception as e:
                self.log(f"  {bl}: Burst action failed ({e})")
                self.state.pending_combat = None
                self.state.pending_hunt = None
                self.state.pending_guide_change = None
                break
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

        # Switch sides unless pending multi-step die action
        if (self.state.pending_army_move_die is not None or
                self.state.pending_muster_die is not None):
            pass  # Same side continues (multi-step die action)
        else:
            self.selfplay_active_side = other

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
        """AI decides how many Shadow dice to pre-allocate as eyes before rolling."""
        from ai import clone_state, encode_state
        import torch, numpy as np
        from game_logic import allocate_eyes

        shadow_dice = [d for d in self.state.dice[Side.SHADOW] if not d.used]
        n_dice = len(shadow_dice)
        if n_dice == 0:
            return

        # Evaluate allocating 0 to min(3, n_dice) eyes using value net
        max_eyes = min(3, n_dice)
        best_eyes = 0
        best_val = None

        sh_net = getattr(self, 'sh_net', self.ai_net)
        sh_net.eval()
        for n_eyes in range(max_eyes + 1):
            s2 = clone_state(self.state)
            if n_eyes > 0:
                indices = [i for i, d in enumerate(s2.dice[Side.SHADOW])
                           if not d.used][:n_eyes]
                allocate_eyes(s2, n_eyes, indices=indices)
            vec = encode_state(s2, Side.SHADOW)
            with torch.no_grad():
                val = sh_net(
                    torch.from_numpy(np.array(vec, dtype=np.float32)).unsqueeze(0)
                ).item()
            if best_val is None or val > best_val:
                best_val = val
                best_eyes = n_eyes

        if best_eyes > 0:
            indices = [i for i, d in enumerate(self.state.dice[Side.SHADOW])
                       if not d.used][:best_eyes]
            allocate_eyes(self.state, best_eyes, indices=indices)
            self.log(f"Shadow AI: Pre-allocated {best_eyes} die(s) as eyes to hunt box")

    def _on_pass_turn(self):
        """Pass without using a die (only when opponent has more unused dice)."""
        if self.ai_playing:
            self.log("Wait for AI to finish its turn")
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
        human_side = Side.FREE_PEOPLES if self.ai_side == Side.SHADOW else Side.SHADOW
        ai_unused = self._count_unused_dice(self.ai_side)
        human_unused = self._count_unused_dice(human_side)
        if ai_unused == 0 and human_unused == 0:
            self.log("All dice used - action phase complete")
            return
        if ai_unused > 0:
            self.ai_playing = True
            self.root.after(500, self._ai_play_one_action)

    def _ai_play_one_action(self):
        """AI plays exactly ONE action, then returns control to the human."""
        from ai import mcts_action, get_legal_actions, _execute_action

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
            self.ai_playing = False
            return

        self.log(f"{self.ai_side.value} AI: {self._describe_action(action)}")

        try:
            _execute_action(self.state, action)
        except Exception as e:
            self.log(f"{self.ai_side.value} AI: Action failed ({e}), skipping")

        self._sync_hands()
        self.refresh()
        self.ai_playing = False

        # If human has no dice left, AI continues playing remaining dice
        human_side = Side.FREE_PEOPLES if self.ai_side == Side.SHADOW else Side.SHADOW
        human_unused = self._count_unused_dice(human_side)
        ai_unused = self._count_unused_dice(self.ai_side)
        if human_unused == 0 and ai_unused > 0:
            self.ai_playing = True
            self.root.after(800, self._ai_play_one_action)
        elif ai_unused == 0 and human_unused == 0:
            self.log("All dice used - action phase complete")

    def _describe_action(self, action):
        """Return a human-readable string for an AI action."""
        t = action.action_type
        p = action.params
        if t == 'army_move':
            return f"Moves army {p['from']} -> {p['to']}"
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
        elif t == 'pass':
            return "Passes (wastes die)"
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
        outcome = f"{meta.get('winner', '?').upper()} wins ({meta.get('victory_type', '?')}) " \
                  f"in {meta.get('turns', '?')} turns"
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
        outcome = f"{meta.get('winner', '?').upper()} wins ({meta.get('victory_type', '?')})"
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
        except BaseException as e:
            # visible even under pythonw (no console)
            try:
                messagebox.showerror('AlphaWOTR', 'net load failed: %s'
                                     % e)
            except Exception:
                pass
            raise
        session.attach_nets(fp_net, sh_net)
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
        self.live_sims_scale = tk.Scale(
            bar, from_=30, to=1000, resolution=10,
            orient=tk.HORIZONTAL, showvalue=0, length=260,
            bg=COLORS['panel_bg'], fg=COLORS['text'],
            highlightthickness=0, troughcolor=COLORS['bg'],
            command=self._live_sims_changed)
        self.live_sims_scale.set(int(init_sims))
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
            self._live_finish(error=got)
            return
        s = self.live_session
        dp = s.current()
        if dp is None:
            self._live_finish()
            return
        desc = self._ep.describe_action(s.state, dp.side,
                                        dp.actions[got])
        nm = 'FP' if dp.side is Side.FREE_PEOPLES else 'SH'
        _n9 = s.last_pick_sims  # what ai_pick actually used (slider-live)
        s.submit(got, dp.side)
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
        if error is not None:
            msg = 'game halted on AI error: %s' % error
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
        _head = ('LIVE GAME STOPPED'
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
            # USER bug 2026-07-28: a stale card-index entry launched
            # the replay GUI at a deleted trace — fail with a message,
            # not a traceback.  2026-08-11 (USER: "doesn't let me open
            # traces on batch 000"): quarantine RENAMES batch dirs, so
            # first resolve the basename across every trace root;
            # only a truly-deleted trace errors — via a dialog, since
            # the launcher usually has no visible console.
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


    # (stale sweep 2026-07-27: the legacy pay-off dialog is deleted —
    #  the engine's certified table-card-kill path is authoritative)
