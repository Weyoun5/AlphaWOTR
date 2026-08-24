#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 00:12:25 2025

@author: Weyoun
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
War of the Ring – annotated map builder
Last update : 2025-06-29
Author      : Weyoun (+ ChatGPT tweaks)

What’s new in this version
--------------------------
•  Nation-coloured region labels (North, Gondor, Rohan, Elves, Dwarves,
   Isengard, Sauron, Southrons/Easterlings).  
•  Icons for settlements & defences, following WotR rules:
      grey rectangle          = fortification
      blue rectangle          = Free Peoples city
      blue square             = Free Peoples stronghold
      red rectangle           = Shadow city
      red square              = Shadow stronghold
•  Uses the metadata already in **wotr_map.txt** (parentId, isCity,
   isStronghold, isFortification, isShadows).
•  Keeps all your previous tweaks: region shifts, hand-added and removed
   adjacencies, Mordor “star”, ONLY-rules, etc.

---------------------------------------------------------------------------
"""

import json, re, unicodedata, os
from shapely.geometry import Polygon
from PIL  import Image

_BOARD_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(os.path.dirname(_BOARD_DIR), 'data')
os.chdir(_BOARD_DIR)


# ╭───────────────────────────────────────────────────────────╮
# │  1.  CONFIGURATION                                       │
# ╰───────────────────────────────────────────────────────────╯
# --- absolute paths (edit to taste) ---
BOARD_PATH = os.path.join(_DATA_DIR, "board200.jpg")
MAP_JSON   = os.path.join(_DATA_DIR, "wotr_map.txt")
OUT_PATH   = os.path.join(_DATA_DIR, "wotr_map.png")

# --- drawing constants ---
RECT_W, RECT_H = 12, 6      # rectangle icon size
SQUARE = 8                  # square icon size
FONT_SIZE = 10

# --- nation text colours ---
NATION_COLOURS = {
    "the north"   : "#69c4ff",   # light blue
    "gondor"      : "#00008b",   # dark blue
    "rohan"       : "#008000",   # green
    "elves"       : "#048f1a",   # bright green
    "dwarves"     : "#8b4513",   # brown
    "isengard"    : "#ffd700",   # yellow
    "sauron"      : "#ff0000",   # red
    # canonical merged faction name (fid_to_name never yields the
    # split 'southrons'/'easterlings' forms)
    "southrons and easterlings": "#ff8c00",   # orange
}

# --- icon fill colours ---
GREY   = "#808080"
BLUE   = "#0099ff"
RED    = "#ff0000"



# ╭───────────────────────────────────────────────────────────╮
# │  2.  HELPER FUNCTIONS                                    │
# ╰───────────────────────────────────────────────────────────╯
_SANITIZE_CACHE: dict = {}
# Sorted-neighbour memo for Board.adjacent (see that method). Module-level
# because every Board instance shares the single static `adj` graph.
_ADJ_SORTED_CACHE: dict = {}


def sanitize(name: str) -> str:
    """ASCII-only, lower-case key used everywhere.

    Memoized: called millions of times on a small set of distinct strings
    (region/card/companion names); the two regex passes dominated
    extraction profiles before caching.
    """
    cached = _SANITIZE_CACHE.get(name)
    if cached is not None:
        return cached
    n = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    # '&' is a word, not punctuation: 'Southrons & Easterlings' must
    # sanitize to the same key as 'Southrons and Easterlings' — the
    # plain strip split ONE nation into two identities
    n = n.replace('&', ' and ')
    n = re.sub(r"[^a-z0-9\s]", " ", n.lower())
    n = re.sub(r"\s+", " ", n).strip()
    if len(_SANITIZE_CACHE) < 100_000:  # bound memory on adversarial input
        _SANITIZE_CACHE[name] = n
    return n


# WOTR_NATIVE=1 (PERF 2026-08-09): compiled hotspot port - PURE
# performance swap, byte-identical outputs (parity-certified; see
# perf/native in the lab tree).  Rebinding here, before any consumer
# executes `from board import sanitize`, routes every sanitize call
# (board, ai, game_logic, ...) through the compiled version.  Default
# OFF: without the env flag this block is inert and the module is
# byte-identical to production behavior.
if os.environ.get('WOTR_NATIVE') == '1':
    try:
        from wotr_native import sanitize as _native_sanitize
    except ImportError as _e:
        raise ImportError(
            'WOTR_NATIVE=1 but the wotr_native extension is not '
            'built/importable: %r' % (_e,))
    sanitize = _native_sanitize


def parse_polygon(path: str, W: int, H: int):
    """Convert CSS-style percentage polygon to absolute pixels."""
    points = []
    inner  = path[len("polygon("):-1]
    for tok in inner.split(','):
        tok = tok.strip()
        m   = re.match(r"([0-9.]+)%\s*([0-9.]+)%", tok)
        if m:
            points.append((float(m.group(1))*W/100, float(m.group(2))*H/100))
    return points



# ╭───────────────────────────────────────────────────────────╮
# │  3.  LOAD BOARD + JSON                                   │
# ╰───────────────────────────────────────────────────────────╯
board_img = Image.open(BOARD_PATH)
W, H      = board_img.size

with open(MAP_JSON, "r", encoding="utf-8") as f:
    factions = json.load(f)

#  quick lookup tables
# The map JSON splits ONE nation into two geographic factions
# ('Southrons' Harad + 'Easterlings' Rhun) and names the neutral
# faction in Spanish ('Regiones Libres').  Canonicalize at load time —
# there is exactly one faction "southrons and easterlings" and one
# "the north"; every consumer downstream sees only canonical names.
_FACTION_CANON = {
    'southrons': 'southrons and easterlings',
    'easterlings': 'southrons and easterlings',
    'north': 'the north',
    'regiones libres': 'free',
}


def _canon_faction(raw_name: str) -> str:
    s = sanitize(raw_name)
    return _FACTION_CANON.get(s, s)


fid_to_name  = {f["id"]: _canon_faction(f["name"]) for f in factions}
name_to_meta = {}
for f in factions:
    _ck = _canon_faction(f["name"])
    if _ck in name_to_meta:
        # merged faction (Southrons + Easterlings): combine the
        # region lists; flags (isShadows, color) agree across records
        _merged = dict(name_to_meta[_ck])
        _merged["regions"] = list(_merged.get("regions", [])) + \
            list(f.get("regions", []))
        name_to_meta[_ck] = _merged
    else:
        name_to_meta[_ck] = f



# ╭───────────────────────────────────────────────────────────╮
# │  4.  BUILD REGION POLYGONS                               │
# ╰───────────────────────────────────────────────────────────╯
regions = {}          # key → dict(orig, poly, centroid, parentId, flags...)
for fac in factions:
    for reg in fac["regions"]:
        key  = sanitize(reg["name"])
        pts  = parse_polygon(reg["path"], W, H)
        if len(pts) < 3:
            continue
        regions[key] = {
            "orig"           : reg["name"],
            "poly"           : Polygon(pts).buffer(0),
            "is_city"        : bool(reg["isCity"]),
            "is_stronghold"  : bool(reg["isStronghold"]),
            "is_fort"        : bool(reg["isFortification"]),
            'is_town'        : bool(reg["isTown"]),
            "parent_id"      : reg["parentId"],
        }

# manual centroid (after later shifts)
for info in regions.values():
    c                 = info["poly"].centroid
    info["centroid"]  = (c.x, c.y)



# ╭───────────────────────────────────────────────────────────╮
# │  5.  GEOMETRIC SHIFTS (East Rhûn, Celduin, Osgiliath)    │
# ╰───────────────────────────────────────────────────────────╯
def shift_region(key_name, dx=0, dy=0):
    k = sanitize(key_name)
    if k in regions:
        new_pts = [(x+dx, y+dy) for x,y in regions[k]["poly"].exterior.coords[:-1]]
        regions[k]["poly"] = Polygon(new_pts).buffer(0)
        c                  = regions[k]["poly"].centroid
        regions[k]["centroid"] = (c.x, c.y)

shift_region("East Rhun",           dx= 90)
shift_region("Vale of the Celduin", dy= 70)
shift_region("Osgiliath",           dx= 30)



# ╭───────────────────────────────────────────────────────────╮
# │  6.  BUILD / EDIT ADJACENCY GRAPH                        │
# │      (unchanged from your previous version)              │
# ╰───────────────────────────────────────────────────────────╯
adj = {k:set() for k in regions}
keys = list(regions)
for i,k1 in enumerate(keys):
    p1 = regions[k1]["poly"]
    for j in range(i+1, len(keys)):
        k2 = keys[j]; p2 = regions[k2]["poly"]
        if p1.distance(p2) < 1.0:
            adj[k1].add(k2); adj[k2].add(k1)

# ---------------------------------------------------------------------------
# Authoritative adjacency overrides (user-verified against the physical
# board, 2026-06-12). The polygon-derived graph connects regions whose
# artwork touches across impassable terrain (mountain walls) — remove those.
# ---------------------------------------------------------------------------
_ADJACENCY_REMOVALS = [
    ("erech", "helm s deep"),   # White Mountains block this edge
    ("erech", "westemnet"),     # White Mountains block this edge
]
for _a, _b in _ADJACENCY_REMOVALS:
    adj.get(_a, set()).discard(_b)
    adj.get(_b, set()).discard(_a)

# --- utility wrappers ---
def add_edge(a,b):
    sa,sb = sanitize(a), sanitize(b)
    if sa in adj and sb in adj:
        adj[sa].add(sb); adj[sb].add(sa)

def remove_edge(a,b):
    sa,sb = sanitize(a), sanitize(b)
    if sa in adj: adj[sa].discard(sb)
    if sb in adj: adj[sb].discard(sa)

add_pairs=[
    ("west harondor","pelargir"),
    ("khand","near harad"),
    ("druadan forest","western emyn muil"),
    ("druadan forest","dead marshes"),
    ("druadan forest","eastemnet"),
    ("old forest road","carrock"),
    ("old forest road","western mirkwood"),
    ("lamedon","pelargir"),
    ("vale of the celduin","north rhun"),
    ("northern dorwinion","north rhun"),
    ("dimrill dale","south anduin vale"),
    ("dimrill dale","north anduin vale"),
    ('cardolan','south ered luin'),
    ('druadan forest','osgiliath'),
    ('umbar','near harad'),
    ('withered heath','dale'),
    ('vale of the celduin','northern rhovanion'),
    ('vale of the celduin','southern rhovanion'),
    ('eastern emyn muil', 'dagorlad'),
    ('lossarnach','osgiliath'),
    ('pelargir','osgiliath'),
    # User-verified against the physical board (2026-07-06): Dol Guldur
    # borders Eastern Mirkwood (the polygon-derived graph missed the
    # edge; 2025-11-14 branstonpickle T1 opens with this exact move)
    ('dol guldur','eastern mirkwood'),
]
for a,b in add_pairs:
    if sanitize(a) not in adj and "druadan forest" in a:
        a="bosque de druadan"
    if sanitize(b) not in adj and "druadan forest" in b:
        b="bosque de druadan"
    add_edge(a,b)

remove_pairs=[
    ("mount gram","trollshaws"),
    ("erech","helms deep"),
    ("erech","edoras"),
    ("far harad","east harondor"),
    ("druwaith iaur","anfalas"),
    ("lamedon","edoras"),
    ("lamedon","folde"),
    ("lamedon","druadan forest"),
    ("lamedon","lossarnach"),
    ("lamedon","minas tirith"),
    ('gorgoroth','south ithilien'),
    ('gorgoroth','south rhun'),
    ('southern mirkwood','narrows of the forest')
    # ensure far harad isolation
]
for a,b in remove_pairs:
    remove_edge(a,b)

# ---------- Mordor star ----------
mordor=list(map(sanitize,["morannon","minas morgul","barad dur","nurn","gorgoroth"]))
g=sanitize("gorgoroth")
# wipe Mordor first retaining external city links we know we want
for r in mordor:
    keep=set()
    if r==sanitize("morannon"):
        keep.add(sanitize("dagorlad"))
    if r==sanitize("minas morgul"):
        keep.update([sanitize("north ithilien"), sanitize("south ithilien")])
    adj[r]=keep
# connect to gorgoroth
for r in mordor:
    if r!=g:
        adj[r].add(g); adj[g].add(r)

# ---------- ONLY rules ----------
only_rules={
 "rivendell":["trollshaws","fords of bruinen"],
 "mount gundabad":["eagles eyrie","mount gram"],
 "harlindon":["grey havens","south ered luin"],
 "forlindon":["grey havens"],
 "moria":["hollin","north dunland","dimrill dale"],
 "lorien":["dimrill dale","parth celebrant"],
 "fangorn":["parth celebrant","eastemnet","westemnet","fords of isen"],
 "orthanc":["fords of isen","gap of rohan"],
 "south dunland":["gap of rohan","enedwaith","tharbad","north dunland"],
 "north dunland":["moria","hollin","south downs","cardolan","tharbad","south dunland"],
 "far harad":["khand","near harad"],
 "minhiriath":["south ered luin","cardolan","tharbad","enedwaith"],
 "nurn":["gorgoroth"],
 "barad dur":["gorgoroth"],
 "morannon":["gorgoroth","dagorlad"],
 "minas morgul":["north ithilien","south ithilien","gorgoroth"],
 "east rhun":["north rhun","south rhun","iron hills",'vale of the carnen']}
only={sanitize(k):[sanitize(x) for x in v] for k,v in only_rules.items()}

# enforce ONLY after all adds/removes
for region,allowed in only.items():
    if region not in adj: continue
    # purge region from neighbours not in allowed
    for other in list(adj.keys()):
        if region in adj[other] and other not in allowed:
            adj[other].discard(region)
    # set region adjacency exactly
    adj[region]=set()
    for n in allowed:
        if n in adj:
            adj[region].add(n)
            adj[n].add(region)



# … everything up to section 7 is identical to the version you just tested …

# ╭───────────────────────────────────────────────────────────╮
# │  7.  DRAW BOARD + ADJACENCIES + ICONS                    │
# ╰───────────────────────────────────────────────────────────╯

if __name__ == '__main__':
    from PIL import ImageDraw, ImageFont
    import matplotlib.pyplot as plt

    canvas = board_img.copy()
    draw   = ImageDraw.Draw(canvas, "RGBA")

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", FONT_SIZE)
    except IOError:
        font = ImageFont.load_default()

    # (a) green adjacency lines
    for src, neighs in adj.items():
        cx, cy = regions[src]["centroid"]
        for n in neighs:
            nx, ny = regions[n]["centroid"]
            draw.line((cx, cy, nx, ny), fill=(0,255,0,180), width=1)

    # (b) settlement / fortification icons
    for key, info in regions.items():
        cx, cy = info["centroid"]
        is_shadow   = name_to_meta[ fid_to_name[info["parent_id"]] ]["isShadows"]
        blue_or_red = BLUE if not is_shadow else RED
        if info["is_fort"]:
            shape, colour = "rect", GREY
        elif info["is_city"]:
            shape, colour = "rect", blue_or_red
        elif info["is_stronghold"]:
            shape, colour = "square", blue_or_red
        elif info.get("is_town"):
            shape, colour = "diamond", blue_or_red
        else:
            shape = None
        if shape == "rect":
            w, h = RECT_W, RECT_H
            draw.rectangle((cx-w/2, cy-h/2, cx+w/2, cy+h/2), fill=colour, outline="black")
        elif shape == "square":
            s = SQUARE
            draw.rectangle((cx-s/2, cy-s/2, cx+s/2, cy+s/2), fill=colour, outline="black")
        elif shape == "diamond":
            d = SQUARE
            pts = [(cx, cy-d/2), (cx+d/2, cy), (cx, cy+d/2), (cx-d/2, cy)]
            draw.polygon(pts, fill=colour, outline="black")

    # (c) nation-coloured labels
    for key, info in regions.items():
        cx, cy  = info["centroid"]
        nation  = fid_to_name[ info["parent_id"] ]
        txt_col = NATION_COLOURS.get(nation, "#000000")
        draw.text((cx + 4, cy - 4), info["orig"], fill=txt_col, font=font)

    canvas.save(OUT_PATH)
    plt.figure(figsize=(12,8)); plt.imshow(canvas); plt.axis("off")
    print("Map with settlement diamonds saved to:", OUT_PATH)

# ╭───────────────────────────────────────────────────────────╮
# │  8.  OO WRAPPER  (added 2025-06-29)                       │
# ╰───────────────────────────────────────────────────────────╯
from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass
class Region:
    """Thin wrapper around one entry of the global *regions* dict."""
    key: str
    orig_name: str
    centroid: Tuple[float, float]

    # board_visual expects .screen_xy
    @property
    def screen_xy(self) -> Tuple[float, float]:
        return self.centroid


class Board:
    """Object-oriented façade over the data already loaded above."""

    def __init__(self):
        # keep reference to globals that are already prepared
        self.background_path: str = BOARD_PATH
        self._regions: Dict[str, Region] = {
            k: Region(k, info["orig"], info["centroid"])
            for k, info in regions.items()
        }

        # crude defaults for boxes/tracks – tweak to taste
        self.fellowship_box_xy = (W - 180, 80)   # top-right corner
        # add more named boxes later as needed (VP track, political, …)
        self._adj = adj

    # --------------------------------------------------------
    def region(self, name: str) -> Region:
        """Return Region object by *original* or sanitised name."""
        key = sanitize(name)
        try:
            return self._regions[key]
        except KeyError:
            # VASSAL location names carry a leading article the board
            # keys drop ('The Grey Havens' vs 'grey havens') and vice
            # versa ('Shire' vs 'the shire') — try both forms before
            # giving up (2024-07-27 Columbo: forced move to 'the grey
            # havens' crashed get_or_create_army)
            alt = key[4:] if key.startswith('the ') else 'the ' + key
            if alt in self._regions:
                return self._regions[alt]
            raise KeyError(f"Unknown region name: {name!r}") from None
            
    def adjacent(self, name: str) -> list[str]:
        key = sanitize(name)
        # SORTED: _adj holds SETS - unsorted iteration made every
        # unlogged first-legal pick (teacher retreat destinations,
        # heuristic fallbacks) PYTHONHASHSEED-dependent; canary Adobim
        # flipped CLEAN/MISMATCH between identical runs (2026-07-11,
        # amendment 1.124)
        #
        # OPT (stateless memo): the adjacency graph `adj` is built once at
        # import and never mutated at runtime (add_edge/remove_edge only
        # run during import), so the sorted neighbour list for a given key
        # is constant. Cache the sorted TUPLE keyed on the sanitized name
        # and return a fresh list copy each call — byte-identical order,
        # and callers keep a mutable list they can modify safely.
        cached = _ADJ_SORTED_CACHE.get(key)
        if cached is None:
            if key not in self._adj:
                # same leading-article fallback as region() — without
                # it an unknown alias silently cached [] forever
                alt = key[4:] if key.startswith('the ') else 'the ' + key
                if alt in self._adj:
                    key = alt
                    cached = _ADJ_SORTED_CACHE.get(key)
        if cached is None:
            cached = tuple(sorted(self._adj.get(key, [])))
            _ADJ_SORTED_CACHE[key] = cached
        return list(cached)
    
    # convenient iteration
    @property
    def regions(self):
        return self._regions.values()


# helper so other modules can just do “b = board.load_board()”
def load_board() -> Board:
    return Board()

