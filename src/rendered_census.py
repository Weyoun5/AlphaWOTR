"""RENDERED-LOG parity census DRIVER  (engine-driven-resolution directive).

Two complementary analyses per game, both anchored on the CLIENT-RENDERED
truth mirror built by rendered_replay (the authoritative gesture resolution):

  ANALYSIS 1 - RECONCILE (census backbone + raw/rendered cross-check).
    The certified pure-batch pipeline (data/parity/<stem>.parity.json) already
    drove the real engine through every logged decision under forced RNG and
    recorded each die-batch where the ENGINE state diverged from the RAW
    (LogReplayer) truth.  For every such divergence we recompute the field's
    value from the INDEPENDENT rendered mirror and classify:
      PERSIST  - engine value never occurs in rendered truth, but the raw
                 truth does  ->  a real engine bug (survives the rendered pipe).
      VANISH   - the engine value DOES occur in rendered truth (engine was
                 right; the raw parser mis-resolved a gesture/timing)  -> a
                 raw-parser artifact that disappears under rendered truth.
      DISAGREE - neither engine nor raw truth matches rendered truth  ->
                 rendered-parser gap / deeper issue (flag for inspection).

  ANALYSIS 2 - ENGINE-DRIVEN HUNT confirmation (directive method, self-
    contained).  For each hunt batch we resync a real GameState to the
    rendered truth-before, force the batch's hunt roll + drawn tile, run the
    engine's own hunt resolution, and compare the engine-computed corruption /
    reveal against the rendered truth-after.  Independent corroboration of the
    RNG-computed corruption surface (where the reroll-drop family lives).

Deliverables: data/rendered_parity/<stem>.json (per game) + CENSUS.md +
resync-rate table (Analysis-1 persist rate is the <1% training-gate view).

No engine files are modified.  Certified primitives (_board_snapshot,
_snapshot_diff, sanitize) are reused by import; the raw pipeline stays intact
for A/B.
"""
import ast
import json
import os
import re
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('PYTHONHASHSEED', '0')

import board  # noqa: E402  (chdir to src on import)
from board import sanitize  # noqa: E402
import rendered_replay as rr  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RLOG_DIR = os.path.join(REPO, 'data', 'rendered_logs')
PARITY_DIR = os.path.join(REPO, 'data', 'parity')
OUT_DIR = os.path.join(REPO, 'data', 'rendered_parity')

# GAMExx stem  ->  raw .log stem (== parity json stem)
_STEM_RE = re.compile(r'^GAME\d+_(.*)\.rendered\.txt$')


def rendered_files():
    fs = sorted(glob.glob(os.path.join(RLOG_DIR, 'GAME*.rendered.txt')))
    # branstonpickle_Compote: client-vs-rulebook casualty quirk game
    # (per-type forcing saga, campaign_survey game 16) -- its rendered
    # log is excluded from every census aggregate
    return [f for f in fs if 'branstonpickle_Compote.rendered' not in f]


def raw_stem(rendered_path):
    m = _STEM_RE.match(os.path.basename(rendered_path))
    return m.group(1) if m else None


def _norm_stem(s):
    # raw .log filenames keep spaces in player names ('Stella Rossa');
    # rendered filenames use underscores -> collapse [ _]+ for matching
    return re.sub(r'[ _]+', '_', s).lower()


_PARITY_BY_STEM = None


def find_parity(stem):
    """Locate the parity json for a rendered stem, tolerating space/underscore
    differences in player-name segments."""
    global _PARITY_BY_STEM
    if _PARITY_BY_STEM is None:
        _PARITY_BY_STEM = {}
        for p in glob.glob(os.path.join(PARITY_DIR, '*.parity.json')):
            base = os.path.basename(p)[:-len('.parity.json')]
            _PARITY_BY_STEM[_norm_stem(base)] = p
    return _PARITY_BY_STEM.get(_norm_stem(stem))


# --------------------------------------------------------------------------
# div-string parsing  ("units@dol amroth eng=[...] truth=[...]")
# --------------------------------------------------------------------------
_DIFF_RE = re.compile(
    r'^(units|leaders|pol|fel|ctrl)[@.](.+?)\s+eng=(.+?)\s+truth=(.+)$')


def parse_diff(s):
    m = _DIFF_RE.match(s)
    if not m:
        return None
    kind, key, eng, truth = m.groups()
    try:
        eng_v = ast.literal_eval(eng)
    except Exception:
        eng_v = eng
    try:
        truth_v = ast.literal_eval(truth)
    except Exception:
        truth_v = truth
    return {'kind': kind, 'key': key, 'eng': eng_v, 'truth': truth_v,
            'raw': s}


def _norm_val(kind, val):
    """Hashable canonical form matching _snapshot_diff's comparison."""
    if kind in ('units', 'leaders', 'ctrl'):
        try:
            return tuple(sorted(tuple(x) if isinstance(x, (list, tuple))
                                else x for x in val))
        except Exception:
            return tuple(val) if isinstance(val, (list, tuple)) else (val,)
    if kind == 'pol':
        return tuple(val) if isinstance(val, (list, tuple)) else (val,)
    # fel: scalar, or comps list -> sorted tuple
    if isinstance(val, list):
        return tuple(sorted(val))
    return val


# --------------------------------------------------------------------------
# rendered-mirror field extraction (matches _snapshot_diff normalization)
# --------------------------------------------------------------------------
def _snap_units(snap, region):
    """sorted merged (nation,kind) list at sanitized `region` (matches
    _snapshot_diff._norm_armies)."""
    out = []
    for k, (units, _l) in snap['armies'].items():
        if sanitize(k) == region:
            out.extend(tuple(u) for u in units)
    return tuple(sorted(out))


def _snap_leaders(snap, region):
    out = []
    for k, (_u, leaders) in snap['armies'].items():
        if sanitize(k) == region:
            out.extend(tuple(l) for l in leaders)
    return tuple(sorted(out))


def _snap_pol(snap, nation):
    v = snap['pol'].get(nation)
    return tuple(v) if v else None


def _snap_fel(snap, attr):
    v = snap['fel'].get(attr)
    if attr == 'region' and v:
        return sanitize(str(v))
    if attr == 'comps' and isinstance(v, list):
        return tuple(sorted(v))
    return v


def _snap_ctrl(snap, side):
    return tuple(sorted(sanitize(x) for x in snap['ctrl'].get(side, [])))


def field_value(snap, kind, key):
    if kind == 'units':
        return _snap_units(snap, sanitize(key))
    if kind == 'leaders':
        return _snap_leaders(snap, sanitize(key))
    if kind == 'pol':
        return _snap_pol(snap, key)
    if kind == 'fel':
        return _snap_fel(snap, key)
    if kind == 'ctrl':
        return _snap_ctrl(snap, key)
    return None


def rendered_values(batches_by_turn, turn, kind, key):
    """Every value the field took across batches of `turn` (+/-1 turn for
    boundary housekeeping lag).  Uses both truth_before and truth."""
    vals = set()
    for t in (turn - 1, turn, turn + 1):
        for b in batches_by_turn.get(t, []):
            for snap_key in ('truth_before', 'truth'):
                snap = b.get(snap_key)
                if snap is None:
                    continue
                try:
                    vals.add(field_value(snap, kind, key))
                except Exception:
                    pass
    return vals


# --------------------------------------------------------------------------
# ANALYSIS 1 - reconcile parity divs against rendered truth
# --------------------------------------------------------------------------
def reconcile(game):
    rendered_path = os.path.join(RLOG_DIR, game) if not os.path.isabs(game) \
        else game
    stem = raw_stem(rendered_path)
    parity_path = find_parity(stem)
    if not parity_path or not os.path.exists(parity_path):
        return None
    parity = json.load(open(parity_path, encoding='utf-8'))
    g = rr.parse_game(rendered_path)
    by_turn = {}
    for b in g['batches']:
        by_turn.setdefault(b['turn'], []).append(b)

    persist = []
    vanish = []
    disagree = []
    for div in parity['divs']:
        turn = div['turn']
        for s in div['diffs']:
            pd = parse_diff(s)
            if pd is None:
                disagree.append({'batch': div['batch'], 'turn': turn,
                                 'diff': s, 'why': 'unparsed'})
                continue
            rvals = rendered_values(by_turn, turn, pd['kind'], pd['key'])
            eng_n = _norm_val(pd['kind'], pd['eng'])
            truth_n = _norm_val(pd['kind'], pd['truth'])
            entry = {'batch': div['batch'], 'turn': turn, 'alog': div['alog'],
                     'kind': pd['kind'], 'key': pd['key'], 'diff': s}
            # pol is monotonic & marker-lagged: engine AHEAD of a lagging
            # rendered marker is not a bug (same doctrine _snapshot_diff uses)
            if eng_n in rvals:
                entry['class'] = 'VANISH'
                vanish.append(entry)
            elif truth_n in rvals:
                entry['class'] = 'PERSIST'
                entry['rendered'] = 'corroborates-rawtruth'
                persist.append(entry)
            elif pd['kind'] == 'pol':
                # _snapshot_diff only emits a pol diff when the engine is
                # BEHIND the truth marker (missed advance/activation) - a
                # real gap per the marker-lag doctrine.  The rendered mirror
                # is a pure-gesture LOWER BOUND (it does not model the
                # rule-implicit attack advance), so it cannot corroborate the
                # exact advanced value - but engine-behind is real either way.
                entry['class'] = 'PERSIST'
                entry['rendered'] = 'pol-engine-behind'
                persist.append(entry)
            else:
                # rendered truth agrees with NEITHER side
                entry['class'] = 'DISAGREE'
                entry['rendered_vals'] = _short_vals(rvals)
                disagree.append(entry)

    # per-batch rollups
    def _batches(lst):
        return len({e['batch'] for e in lst})
    total_batches = parity['batches']
    persist_b = _batches(persist)
    out = {
        'game': stem, 'rendered_log': os.path.basename(rendered_path),
        'total_batches': total_batches,
        'raw_diverged_batches': parity['diverged'],
        'raw_echoes': parity.get('echoes', 0),
        'rendered_batches': len(g['batches']),
        'rendered_turns': g['final_turn'],
        'parser_gaps': len(g['gaps']),
        'persist_fields': len(persist), 'persist_batches': persist_b,
        'vanish_fields': len(vanish), 'vanish_batches': _batches(vanish),
        'disagree_fields': len(disagree),
        'disagree_batches': _batches(disagree),
        'resync_rate_raw': round(parity['diverged'] / total_batches, 4)
        if total_batches else 0,
        'resync_rate_rendered': round(persist_b / total_batches, 4)
        if total_batches else 0,
        'persist': persist, 'vanish': vanish, 'disagree': disagree,
    }
    return out


def _short_vals(vals):
    out = []
    for v in list(vals)[:4]:
        out.append(str(v)[:60])
    return out


# --------------------------------------------------------------------------
# ANALYSIS 2 - engine-driven hunt confirmation (self-contained)
# --------------------------------------------------------------------------
def hunt_confirm(game):
    """Drive the real engine's hunt resolution for each rendered hunt batch
    under forced roll+tile; compare engine-computed corruption delta against
    the rendered truth delta.  Returns list of hunt-computation divergences.
    """
    import random
    import game_state as gs
    import game_logic as gl
    rendered_path = os.path.join(RLOG_DIR, game) if not os.path.isabs(game) \
        else game
    g = rr.parse_game(rendered_path)
    divs = []
    b = board.load_board()
    for bt in g['batches']:
        if bt['kind'] != 'hunt':
            continue
        # need: a forced hunt roll chunk + a drawn tile in this batch
        roll = next((r for r in bt['rng'] if r['kind'] == 'roll_num'), None)
        tile = next((r for r in bt['rng'] if r['kind'] == 'draw_tile'), None)
        tb = bt['truth_before']['fel']
        ta = bt['truth']['fel']
        corr_delta = ta['corruption'] - tb['corruption']
        # a companion sacrificed to the Hunt absorbs its LEVEL of hunt
        # damage (the rule _drive_hunt applies via _COMP_LEVEL); total
        # truth damage = corruption taken + absorbed levels this batch
        _lost9 = [n9 for n9 in (tb.get('comps') or [])
                  if n9 not in (ta.get('comps') or [])]
        comps_absorb = sum(_COMP_LEVEL.get(n9, 1) for n9 in _lost9)
        truth_damage = corr_delta + comps_absorb
        # only measure ordinary (non-Mordor) hunts with a numeric roll + tile
        if roll is None or tile is None or tb['mordor']:
            continue
        tile_name = str(tile['value'])
        # build a minimal engine state carrying just the hunt inputs
        st = gs.setup_initial_game(b)
        st.fellowship.corruption = tb['corruption']
        st.fellowship.hidden = tb['hidden']
        st.fellowship.progress = tb['progress']
        # hunt level = number of dice in the roll chunk (Shadow hunt dice)
        n_dice = len(roll['ints'])
        st.hunt_box = [gs.ActionDie(gs.SH_FACES, gs.Side.SHADOW)
                       for _ in range(n_dice)]
        # arrange the drawn tile on top of the pool
        forced_tile = _tile_obj(gs, tile_name)
        if forced_tile is None:
            continue
        st.hunt_pool = [forced_tile]
        # force the roll: patch random.randint to yield the logged ints
        seq = list(roll['ints'])
        orig_randint = random.randint
        orig_choice = random.choice
        it = iter(seq)

        def fake_randint(a, bb, _it=it, _f=orig_randint):
            try:
                return next(_it)
            except StopIteration:
                return _f(a, bb)

        random.randint = fake_randint
        random.choice = lambda seq2: seq2[-1]
        try:
            gl.move_fellowship(st, skip_auto_sacrifice=True)
        except Exception as e:
            divs.append({'turn': bt['turn'], 'batch_line': bt['start_line'],
                         'why': 'engine-exception', 'err': str(e)[:80]})
            continue
        finally:
            random.randint = orig_randint
            random.choice = orig_choice
        # damage the engine computed = corruption applied + damage still
        # pending the sacrifice choice (skip_auto_sacrifice left it there)
        eng_damage = (st.fellowship.corruption - tb['corruption']) + \
            int(getattr(st, '_pending_hunt_damage', 0) or 0)
        if eng_damage != truth_damage:
            divs.append({
                'turn': bt['turn'], 'line': bt['start_line'],
                'tile': tile_name, 'roll': seq,
                'eng_damage': eng_damage, 'truth_damage': truth_damage,
                'why': 'hunt-damage-mismatch'})
    return divs


def _tile_obj(gs, name):
    import copy
    n = name.strip().lower()
    if n == 'eye':
        return gs.HuntTile(is_eye=True)
    m = re.match(r'^(\d+)(r?)$', n)
    if m:
        return gs.HuntTile(damage=int(m.group(1)),
                           reveal=bool(m.group(2)))
    # Named blue (Fellowship) / red (Shadow) SPECIAL hunt tiles - Elven
    # Rope, Elven Cloaks, Shelob's Lair, Smeagol Helps Nice Master, On On
    # They Went, etc.  Return a COPY of the engine's own tile definition so
    # its modeled properties (damage / is_dice_roll / stops_movement /
    # reveal / is_eye) drive resolution.  rb p.39 (data/_rulebook.txt
    # L3576-3608 SPECIAL HUNT TILES): "The special Hunt tiles ... enter play
    # only by the use of Event cards ... A Die icon means that the Hunt
    # damage is equal to the subsequent roll of a die ... A negative value
    # (-2 or -1) means that there is no Hunt damage and instead the
    # indicated number is actually subtracted from the current Corruption".
    for key, tile in getattr(gs, 'SPECIAL_HUNT_TILES', {}).items():
        # accent-fold: engine keys 'Sm\xe9agol...', logs spell ASCII
        k9 = key.strip().lower().replace('\xe9', 'e')
        if k9 == n or key.strip().lower() == n:
            return copy.deepcopy(tile)
    return None


# --------------------------------------------------------------------------
# ANALYSIS 3 - RENDERED-DRIVEN engine census (directive next-step).
#   Instead of reconciling the RAW pipeline's engine drive, drive the REAL
#   engine directly from the rendered mirror: reconstruct a GameState from
#   each batch's rendered truth-BEFORE, run the engine's own rule resolver
#   for that batch's decision under FORCED RNG (hunt: move_fellowship; combat:
#   game_logic.battle), then diff the engine-after board against the rendered
#   mirror's truth-AFTER with the local snapshot diff.  No
#   raw LogReplayer, no parity json - the rendered truth is the sole anchor.
#   Bookkeeping batches (muster/move/political) carry no engine DISCRETION, so
#   the engine reproduces them by construction; only the rule resolvers can
#   diverge, and those divergences are the honest engine-handler signal.
# --------------------------------------------------------------------------
_LDR_META = {'Witch King': ('SAURON', 2), 'Saruman': ('SAURON', 1),
             'Mouth of Sauron': ('SAURON', 1)}


def _state_from_snapshot(snap):
    """Reconstruct a real engine GameState from a rendered mirror snapshot
    (armies, political, fellowship, control, siege).  Named leaders become
    real Leader objects carrying .name/.leadership so _board_snapshot round-
    trips and combat leadership is modelled."""
    import game_state as gs
    import game_logic as gl
    b = board.load_board()
    st = gs.setup_initial_game(b)
    fresh = gs.setup_initial_game(b)
    comp_by_name = {c.name: c for c in fresh.fellowship.companions}
    Nation, UnitKind = gs.Nation, gs.UnitKind
    st.armies = {}

    def mk_leader(desc):
        kind, val = desc[0], desc[1]
        if kind == 'NAZGUL':
            return gs.Nazgul(Nation[val])
        if kind == 'LEADER':
            return gs.Leader(Nation[val])
        if kind == 'NAMED' and val == 'Saruman':
            # W2-033 (COORD #2 2026-09-03, USER order 22:30): Saruman is
            # a MINION - the engine identifies him by the board object
            # muster_minion places (game_logic.Minion, _is_saruman keys
            # on the type name) and excludes him from every transport
            # (army moves, battle advances, retreats).  A plain
            # gs.Leader named Saruman failed that test, so the engine
            # offered / executed advances that carried him out of
            # Orthanc (2023-12-25 L761-770: the human dragged him with
            # the advance and the opponent put him back - the real
            # advance is [5]Isengard Elite + 1 Regular, Saruman stays).
            # Narrow fix: Saruman only (Witch-king / Mouth stay as the
            # _LDR_META leaders they were - a separate lead).
            st.saruman.in_play = True
            st.saruman.was_ever_in_play = True
            return gl.Minion(nation=Nation.SAURON, name='Saruman')
        if kind == 'NAMED':
            if val in _LDR_META:
                nat, ldr = _LDR_META[val]
                lo = gs.Leader(Nation[nat])
            else:
                c = comp_by_name.get(val)
                ldr = getattr(c, 'leadership', 1) if c else 1
                lo = gs.Leader(Nation.GONDOR)
            lo.name = val
            lo.leadership = ldr
            return lo
        return gs.Leader(Nation.SAURON)

    for reg, (units, leaders) in snap['armies'].items():
        # defensive: an off-board pseudo-region (a mirror leak into a
        # casualties/reinforcement zone) is not a board region - skip it
        # rather than crash reconstruction (off-board figures aren't on the
        # board anyway, so dropping them is the faithful thing)
        if sanitize(reg) not in board.regions:
            continue
        a = gl.get_or_create_army(st, sanitize(reg))
        for (nat, ki) in units:
            a.units.append(gs.Unit(Nation[nat], UnitKind[ki]))
        for desc in leaders:
            a.leaders.append(mk_leader(tuple(desc)))
    for nat, pv in snap['pol'].items():
        st.political.boxes[Nation[nat]] = pv[0]
        st.political.active[Nation[nat]] = pv[1]
    fel = snap['fel']
    st.fellowship.region = fel['region']
    st.fellowship.progress = fel['progress']
    st.fellowship.corruption = fel['corruption']
    st.fellowship.hidden = fel['hidden']
    st.fellowship.mordor_entered = fel['mordor']
    st.fellowship.mordor_step = fel['step']
    st.fellowship.companions = [comp_by_name[n] for n in fel['comps']
                                if n in comp_by_name]
    # W3-018: the Guide is the log's own declaration carried by the mirror
    # snapshot ('guide': initial Gandalf the Grey, every 'moves <Companion>
    # ... to Guide Of The Fellowship', Gollum for an empty company, the
    # highest-Level rule between a guide's death and the next declaration).
    # companions[0] of the alphabetically sorted set was Boromir at turn 1
    # and Gandalf the Grey (level 3, palantir draw) for a Gollum-led company:
    # Strider's hide_any_die, the hobbit guide's hunt_guide_separate,
    # Gollum's reveal-to-reduce and the Athelas/Candles thresholds all key
    # on the guide identity.
    _gname = fel.get('guide')
    _gobj = comp_by_name.get(_gname) if _gname else None
    if _gobj is not None and _gobj in st.fellowship.companions:
        st.fellowship.guide = _gobj
    elif not st.fellowship.companions:
        gl.change_guide(st)          # Gollum guides an empty company (rb p.39)
    else:
        # snapshot without a usable guide name: engine rule (highest Level,
        # ties in setup order = fresh.fellowship.companions order)
        _best = max(c.level for c in st.fellowship.companions)
        st.fellowship.guide = next(
            c for c in fresh.fellowship.companions
            if c.level == _best and c in st.fellowship.companions)
    st.enemy_controlled = {
        gs.Side.FREE_PEOPLES: set(snap['ctrl']['FREE_PEOPLES']),
        gs.Side.SHADOW: set(snap['ctrl']['SHADOW'])}
    st.besieged_strongholds = set(snap['besieged'])
    # Permanent supply is part of the snapshot, even when the lost figure
    # is absent from every board army. Keep independent enum-keyed copies.
    st.fp_casualties = {}
    for nation, counts in snap.get('fpc', {}).items():
        if nation not in ('ELVES', 'DWARVES', 'GONDOR', 'THE_NORTH', 'ROHAN'):
            raise ValueError('Non-FP nation in permanent casualty ledger: ' + str(nation))
        if any(kind not in ('regular', 'elite', 'leader') or type(count) is not int or count < 0
               for kind, count in counts.items()):
            raise ValueError('Invalid permanent casualty ledger')
        st.fp_casualties[Nation[nation]] = dict(counts)
    from rendered_replay import Mirror
    present = {Mirror._minion_attr(desc[1]) for reg, (_, leaders) in snap['armies'].items()
               if sanitize(reg) in board.regions
               for desc in leaders if desc[0] == 'NAMED'}
    for attr in ('saruman', 'witch_king', 'mouth_of_sauron'):
        history = snap.get('minion_history', {}).get(attr, {})
        special = getattr(st, attr)
        special.in_play = attr in present
        special.was_ever_in_play = bool(history.get('entered') or special.in_play)
    return st


def _first_roll(batch):
    return next((r for r in batch['rng'] if r['kind'] == 'roll_num'), None)


def _all_roll_ints(batch):
    """W2-035: every logged die of every roll_num chunk of the batch, in
    line order (first roll, then the re-rolls) - the forced sequence the
    engine's hunt resolver consumes through random.randint."""
    out = []
    for r in batch['rng']:
        if r['kind'] == 'roll_num':
            out.extend(int(v) for v in r['ints'])
    return out


def _batch_has_combat_card(batch):
    """A combat card is played this batch (a '/'-titled card moved to a
    Current FP/SA Card slot, or killed off one) - the crude single-round
    forced-dice drive cannot model its effect, so such a battle is not
    faithfully drivable."""
    for r in batch['records']:
        if r.get('op') == 'move' and '/' in str(r.get('obj', '')) and \
                'current' in str(r.get('to', '')).lower() and \
                'card' in str(r.get('to', '')).lower():
            return True
        if r.get('op') == 'kill' and 'card' in str(r.get('zone', '')).lower():
            for p in r.get('pieces', []):
                if '/' in p:
                    return True
    return False


_SHADOW_NATS = ('SAURON', 'ISENGARD', 'SOUTHRON_EASTERLINGS')


def _nat_side(nat):
    return 'SHADOW' if nat in _SHADOW_NATS else 'FREE_PEOPLES'


def _unit_side(u):
    return _nat_side(u.nation.name)


def _other_side(side):
    return 'FREE_PEOPLES' if side == 'SHADOW' else 'SHADOW'


_CARD_LOOKUP_CACHE = None


def _card_lookup():
    global _CARD_LOOKUP_CACHE
    if _CARD_LOOKUP_CACHE is None:
        import ai
        _CARD_LOOKUP_CACHE = ai._build_card_name_lookup()
    return _CARD_LOOKUP_CACHE


def _eng_side_units(st, region, side):
    """Sorted (nation,kind) of `side`'s units at `region` in a live state."""
    out = []
    for k, a in st.armies.items():
        if sanitize(k) != region:
            continue
        for u in a.units:
            if _unit_side(u) == side:
                out.append((u.nation.name, u.kind.name))
    return sorted(out)


def _snap_side_units(snap, region, side):
    out = []
    for k, (units, _l) in snap['armies'].items():
        if sanitize(k) == region:
            for (nat, ki) in units:
                if _nat_side(nat) == side:
                    out.append((nat, ki))
    return sorted(out)




def _snap_sides_at(snap, region):
    """Sides with units present at `region` in a snapshot."""
    s = set()
    for k, (units, _l) in snap['armies'].items():
        if sanitize(k) == region:
            for (nat, _ki) in units:
                s.add(_nat_side(nat))
    return s


def _atk_units_at(st, region, atk_side):
    a = st.armies.get(sanitize(region))
    if not a:
        for k, aa in st.armies.items():
            if sanitize(k) == sanitize(region):
                a = aa
                break
    if not a:
        return 0
    return sum(1 for u in a.units if _unit_side(u) == atk_side)


def _attacker_advance_plan(st, batch, atk_side, bregs):
    """The post-combat ADVANCE - the STRONGEST attacker/defender pin.  The
    winning attacker moves his own Army units from the attacker region into the
    adjacent embattled (defender) region.  rb p.30 Note / p.31 (data/_rulebook
    .txt L2538-2544 / L2832-2834): "Only if the battle is over and won, the
    attacker may move his attacking units into the embattled region" / "If the
    defending Army is eliminated or retreats, the attacker may immediately move
    ... into the embattled region."  This pins BOTH regions even when the
    kill-only battle-region set collapses onto the attacker (a defender that
    fully retreated, or whose loss rendered as a move to a Current Card /
    Casualties slot).  Returns (attacker_region, defender_region) when exactly
    one such advance is present, else None (attacker lost / did not advance)."""
    hits = []
    for r in batch['records']:
        if r.get('op') != 'move':
            continue
        fk = rr.zone_kind(r.get('from', ''))
        tk = rr.zone_kind(r.get('to', ''))
        if fk[0] not in ('region', 'stronghold') \
                or tk[0] not in ('region', 'stronghold') or fk[1] == tk[1]:
            continue
        obj = str(r.get('obj', ''))
        toks = (rr.split_group(obj.replace('group (', '').rstrip(')'))
                if 'group' in obj else [obj])
        us = set()
        for t in toks:
            p = rr.parse_piece(t)
            if p and p[0] == 'unit':
                us.add(_nat_side(p[1]))
        if us != {atk_side}:                 # attacker-side Army units only
            continue
        # tie the advance to the battle: it leaves OR enters a battle region
        if fk[1] not in bregs and tk[1] not in bregs:
            continue
        try:
            if fk[1] in [sanitize(x) for x in st.board.adjacent(tk[1])]:
                hits.append((fk[1], tk[1]))
        except Exception:
            pass
    return hits[0] if len(hits) == 1 else None


def _plan_combat(st, batch, expected_dice=None, chats=None):
    """Identify (attacker_region, defender_region, is_siege) for a combat
    batch from the rendered kill regions + truth-before occupation.  Handles
    co-located sieges (attacker==defender key), one-region field assaults and
    two-region field battles (mutual casualties).  When `expected_dice` (the
    first roll-chunk size = the attacker's true combat strength) is given, the
    adjacent-attacker search PREFERS the region whose unit count matches it -
    disambiguating multiple adjacent Shadow armies and rejecting a mis-ID.
    When `chats` (the batch's chat texts) is given, an explicit chat naming
    exactly ONE candidate source overrides tie-order (gesture-primary: T4
    Lossarnach 'from Pel' - both MT and Pelargir matched 5 dice and the
    adjacency-order pick poisoned the whole family drive)."""
    bregs = set(rr._battle_regions(batch))
    atk_side = batch.get('die_side')
    # A combat CASUALTY renders as the losing unit leaving its region to an
    # OFF-BOARD sink - Reinforcements (kills-as-returns dialect, GAME19), the
    # Casualties box, or (the client's field-battle dialect) a Current FP/SA
    # Card slot.  Every such region is a battle region; _battle_regions counts
    # only op=='kill', so without this a field battle whose defender loss
    # rendered as a move-to-Card/Casualties collapsed onto the ATTACKER's own
    # casualty region (COMBAT-1 root: no-plan + friendly-army). rb p.30 Attack.
    for r in batch['records']:
        if r.get('op') != 'move':
            continue
        fk = rr.zone_kind(r.get('from', ''))
        tk = rr.zone_kind(r.get('to', ''))
        if fk[0] not in ('region', 'stronghold') \
                or tk[0] not in ('reinf', 'cas', 'card'):
            continue
        p = rr.parse_piece(str(r.get('obj', '')).replace(
            'group (', '').rstrip(')').split(',')[0].strip())
        if p and p[0] == 'unit':
            bregs.add(fk[1])
    if not atk_side or not bregs:
        return None
    tb = batch['truth_before']
    besieged = set(sanitize(x) for x in (tb.get('besieged') or []))
    # STRONGEST signal first (rb p.31 advance): a single winning-attacker
    # advance from the attacker region into the adjacent embattled region pins
    # both unambiguously.  A cross-region advance is a FIELD attack (a besieged
    # garrison can only be attacked co-located, rb p.32).
    _adv = _attacker_advance_plan(st, batch, atk_side, bregs)
    if _adv is not None:
        return (_adv[0], _adv[1], False)
    dfn = None
    is_siege = False
    # the DEFENDER is where the NON-attacking side took casualties -
    # decisive when several battle regions coexist (GAME19 T8: the
    # attacker's own losses at Minas Morgul made MM outrank the real
    # defender region Gorgoroth via the besieged-precedence)
    _dstr = ('FREE_PEOPLES' if atk_side == 'SHADOW' else 'SHADOW')
    _ecas = set()
    for r in batch['records']:
        toks9, reg9 = [], None
        if r.get('op') == 'kill':
            zk9 = rr.zone_kind(str(r.get('zone', '') or ''))
            if zk9[0] in ('region', 'stronghold') and zk9[1] in bregs:
                reg9 = zk9[1]
                toks9 = (r.get('pieces') or [])
        elif r.get('op') == 'move':
            fk9 = rr.zone_kind(r.get('from', ''))
            tk9 = rr.zone_kind(r.get('to', ''))
            if fk9[0] in ('region', 'stronghold') and fk9[1] in bregs \
                    and tk9[0] in ('cas', 'reinf', 'card'):
                reg9 = fk9[1]
                toks9 = [r.get('obj', '')]
        for t9 in toks9:
            for t8 in (rr.split_group(
                    str(t9).replace('group (', '').rstrip(')'))
                    if 'group' in str(t9) else [str(t9)]):
                p9 = rr.parse_piece(t8)
                if p9 and p9[0] == 'unit':
                    if _nat_side(p9[1]) == _dstr and reg9:
                        _ecas.add(reg9)
    if len(_ecas) == 1:
        dfn = next(iter(_ecas))
        is_siege = dfn in besieged
    for reg in ([] if dfn else sorted(bregs)):
        if reg in besieged:
            dfn = reg
            is_siege = True
            break
    if dfn is None:
        # a valid defender holds an army in truth-before (a battle() pre-
        # condition) and is NOT the destination of an enemy retreat from
        # another battle region (rb p.31 retreat - Stella/Jaymanen no-def-army).
        _retreat_dests = set()
        for r in batch['records']:
            if r.get('op') != 'move':
                continue
            fk0 = rr.zone_kind(r.get('from', ''))
            tk0 = rr.zone_kind(r.get('to', ''))
            if fk0[0] in ('region', 'stronghold') and fk0[1] in bregs \
                    and tk0[0] in ('region', 'stronghold') \
                    and tk0[1] != fk0[1]:
                _retreat_dests.add(tk0[1])
        cand = [r for r in sorted(bregs)
                if atk_side not in _snap_sides_at(tb, r)
                and _snap_sides_at(tb, r) and r not in _retreat_dests]
        if len(cand) == 1:
            dfn = cand[0]
        elif len(bregs) == 1:
            dfn = next(iter(bregs))
        elif cand:
            dfn = cand[0]
    if dfn is None:
        return None
    sides_at_dfn = _snap_sides_at(tb, dfn)
    # A co-located SIEGE requires the attacker to be PHYSICALLY present in the
    # defender's region in truth-before (rb p.32: a besieged Army "can only be
    # attacked by an Army in the same region").  The `besieged` truth-flag can
    # be stale/aspirational (a garrison holding in the OPEN with 10 units - it
    # cannot be inside the 5-cap stronghold, rb p.31 - and zero co-located
    # enemy); in that case the co-located split has an empty besieger side, so
    # fall through to the field-battle search.  COMBAT-1 coloc-missing root.
    if atk_side in sides_at_dfn and _other_side(atk_side) in sides_at_dfn:
        # co-located siege UNLESS the ATTACKER's own casualties sit in
        # an adjacent battle region - then an OUTER army fought (GAME19
        # T8 sortie: 'Minas Morgul from Gorgoroth', FP GR killed AT
        # Gorgoroth while the FP garrison also holds MM)
        _acas9 = set()
        for r in batch['records']:
            reg9 = None
            if r.get('op') == 'kill':
                zk9 = rr.zone_kind(str(r.get('zone', '') or ''))
                if zk9[0] in ('region', 'stronghold') \
                        and zk9[1] in bregs and zk9[1] != dfn:
                    for pc9 in (r.get('pieces') or []):
                        p9 = rr.parse_piece(str(pc9))
                        if p9 and p9[0] == 'unit':
                            if _nat_side(p9[1]) == atk_side:
                                _acas9.add(zk9[1])
        if len(_acas9) == 1:
            _src9 = next(iter(_acas9))
            try:
                if _src9 in [sanitize(x)
                             for x in st.board.adjacent(dfn)]:
                    return (_src9, dfn, False)
            except Exception:
                pass
        return (dfn, dfn, True)         # co-located siege (shared region key)
    # field battle: attacker is an adjacent region carrying attacker units;
    # prefer the one whose combat strength matches the observed roll size
    try:
        adj = [sanitize(r) for r in st.board.adjacent(dfn)]
    except Exception:
        adj = []
    cands = [r for r in adj if _atk_units_at(st, r, atk_side) > 0]
    for r in sorted(bregs):             # the other kill region also qualifies
        if r != dfn and atk_side in _snap_sides_at(tb, r) and r not in cands:
            cands.append(r)
    if not cands:
        return None
    # USER 2026-07-24: the attack source is COMPUTABLE from the battle's
    # own gestures - in a two-region field battle the ATTACKER's
    # casualties land in the attacker's region (kills there / parked
    # moves out of there), the strongest evidence, ahead of dice-count
    # and chat naming.
    atk_cas = set()
    for r in batch['records']:
        if r.get('op') == 'kill':
            zk = rr.zone_kind(str(r.get('zone', '') or ''))
            if zk[0] in ('region', 'stronghold') and zk[1] != dfn:
                for tok in (r.get('pieces') or []):
                    p = rr.parse_piece(tok)
                    if p and p[0] == 'unit':
                        if _nat_side(p[1]) == atk_side:
                            atk_cas.add(zk[1])
        elif r.get('op') == 'move':
            fk = rr.zone_kind(r.get('from', ''))
            tk = rr.zone_kind(r.get('to', ''))
            # W2-030: same dialects as the defender scan (_ecas) - a
            # GROUP drag and the cas / reinf / card destinations
            # ('group ([2]S&E Regular) from West Harondor to SA
            # Casualties', 2023-12-25 L1183: attack source fell to
            # pool[0] = lossarnach with the bare-unit scan)
            if fk[0] in ('region', 'stronghold') and fk[1] != dfn \
                    and tk[0] in ('cas', 'reinf', 'card'):
                _o = str(r.get('obj', '') or '')
                for tok in (rr.split_group(
                        _o.replace('group (', '').rstrip(')'))
                        if 'group' in _o else [_o]):
                    p = rr.parse_piece(tok)
                    if p and p[0] == 'unit':
                        if _nat_side(p[1]) == atk_side:
                            atk_cas.add(fk[1])
    pool = cands
    if expected_dice is not None:
        exact = [r for r in cands
                 if min(5, _atk_units_at(st, r, atk_side)) == expected_dice]
        if len(exact) == 1:
            return (exact[0], dfn, False)
        if exact:
            pool = exact
    if chats and len(pool) > 1:
        try:
            from rendered_extract import _chat_names_region
            named = [r for r in pool
                     if r != dfn and _chat_names_region(r, chats)]
            if len(named) == 1:
                return (named[0], dfn, False)
        except Exception:
            pass
    # casualty evidence decides among dice-consistent candidates when no
    # chat names one (USER 2026-07-24: the source is computable without
    # chat); it ranks AFTER an explicit chat naming (region-to-casualties
    # gestures also arise from table-card payments and unrelated
    # cleanup) - it serves the chat-less games.
    _cas_named = [r for r in pool if r != dfn and r in atk_cas]
    if len(_cas_named) == 1:
        return (_cas_named[0], dfn, False)
    return (pool[0], dfn, False)


def _combat_cards(batch):
    """(attacker_card_id, defender_card_id) from the batch's combat-card
    plays (moves of a '/'-titled card into a Current FP/SA Card slot),
    resolved to engine card ids via the closed_loop lookup (imported, never
    edited).  Cards that don't resolve stay None (best-effort drive)."""
    lk = _card_lookup()
    fp_cid = sa_cid = None
    for r in batch['records']:
        if r.get('op') != 'move':
            continue
        to = str(r.get('to', '')).lower()
        if 'card' not in to or 'current' not in to:
            continue
        title = r.get('obj', '')
        if '/' not in str(title):
            continue
        cid = _lookup_cid(lk, title)
        if cid is None:
            continue
        if 'fp' in to or 'free' in to:
            fp_cid = cid
        elif 'sa' in to or 'shadow' in to:
            sa_cid = cid
    if batch.get('die_side') == 'SHADOW':
        return (sa_cid, fp_cid)
    return (fp_cid, sa_cid)






def _combat_atk_player(batch):
    """The player who STAGED the action die = the attacker (roll chunks by
    this player are attacker rolls; the other player's are the defender's)."""
    for r in batch['records']:
        if r.get('op') == 'move' and r.get('die_stage'):
            return r.get('player')
    return None


# --- combat ROUND segmentation (W2-008) -----------------------------------
# The rendered log rolls in HUMAN order and humans also roll their combat
# CARD strikes (Sudden Strike / Charge / Durin's Bane before the volleys,
# Onslaught / We Come to Kill after the casualties), split a leader
# re-roll over several chunks, and sometimes let the defender roll first.
# The old rule ("an attacker chunk after a defender chunk opens a round")
# minted a phantom round out of every one-sided card roll and every
# defender-first round (~90 phantom combat_card/press/retreat rows over
# the 24 gate games).  Rounds are now segmented from the dice COUNTS and
# the casualty/card gestures between the chunks; every chunk is classed
# pre / main / re / post and re-serialised in the engine's own randint
# order (pre strikes defender-first, atk main, def main, atk re-rolls,
# def re-rolls, post strikes).
_PRE_ROLL_CARDS = {'sudden strike': None, 'charge': None, "durin's bane": 3}
_POST_ROLL_CARDS = ('onslaught', 'we come to kill')
_CUR_CARD_ZONES = ('current fp card', 'current sa card')


def _roll_tokens(obj):
    mg = re.match(r'^group\s*\((.*)\)$', str(obj or ''), re.I)
    return rr.split_group(mg.group(1)) if mg else [str(obj or '')]


def _card_title_side(title, zone):
    """Owning side of a rendered combat-card title (card identity first,
    the Current FP/SA slot it was committed to as the fallback)."""
    import ai
    cid = ai._lookup_card_id(_card_lookup(), str(title))
    if not cid:
        head = str(title).split('/')[0].strip()
        cid = ai._lookup_card_id(_card_lookup(), head) if head else None
    if cid:
        return 'FREE_PEOPLES' if str(cid).startswith('FREE_PEOPLES') else 'SHADOW'
    return 'FREE_PEOPLES' if 'fp' in str(zone).lower() else 'SHADOW'


def _card_discard_line(batch, title, after):
    """Line of the first discard of a committed combat card after its
    commit line (a kill in a card/dice zone, or a move out of the Current
    slot), or None while it is never discarded inside the batch."""
    tl = str(title or '').strip().lower()
    for r in batch['records']:
        ln = r.get('line', 0)
        if ln <= after:
            continue
        op = r.get('op')
        if op == 'kill':
            zk = rr.zone_kind(str(r.get('zone', '') or ''))
            if zk[0] not in ('card', 'dice'):
                continue
            toks = []
            for pc in (r.get('pieces') or []):
                toks.extend(_roll_tokens(pc))
            if any(str(t).strip().lower() == tl for t in toks):
                return ln
        elif op == 'move':
            if (str(r.get('obj', '') or '').strip().lower() == tl
                    and str(r.get('from', '')).strip().lower() in _CUR_CARD_ZONES
                    and str(r.get('to', '')).strip().lower() not in _CUR_CARD_ZONES):
                return ln
    return None


def _combat_card_commits(batch):
    """[(line, side_str, cls, fixed_dice, discard_line)] for every combat
    card commit (a '/'-titled card moved INTO a Current FP/SA Card slot);
    cls 'pre' = strike rolled before the combat roll, 'post' = strike
    rolled after the casualties, 'none' otherwise.  Table event plays
    that transit a Current slot on their way to a Board* slot (siege-round
    counters) are never combat-card commits."""
    out = []
    table_titles = set()
    for r in batch['records']:
        if r.get('op') == 'move' and rr.zone_kind(str(r.get('to', '')))[0] == 'table':
            table_titles.add(str(r.get('obj', '') or '').strip().lower())
    for r in batch['records']:
        if r.get('op') != 'move':
            continue
        if str(r.get('to', '')).strip().lower() not in _CUR_CARD_ZONES:
            continue
        obj = str(r.get('obj', '') or '')
        if '/' not in obj or obj.strip().lower() in table_titles:
            continue
        ct = obj.split('/')[-1].strip().lower()
        if ct in _PRE_ROLL_CARDS:
            cls, fixed = 'pre', _PRE_ROLL_CARDS[ct]
        elif ct in _POST_ROLL_CARDS:
            cls, fixed = 'post', None
        else:
            cls, fixed = 'none', None
        ln = r.get('line', 0)
        out.append((ln, _card_title_side(obj, r.get('to', '')), cls, fixed,
                    _card_discard_line(batch, obj, ln)))
    return out


def _combat_gestures(batch, lo, hi):
    """Casualty-phase gestures strictly inside (lo, hi): -> (n_any,
    {side: net unit losses}, n_card) where losses = units of that side
    leaving a board region (kill / park to casualties, reinforcements or a
    card slot) minus replacement units returned from casualties or
    reinforcements (an elite downgrade nets to zero); n_card counts the
    combat-card commits and discards inside the window."""
    n = 0
    ncard = 0
    losses = {'FREE_PEOPLES': 0, 'SHADOW': 0}
    for r in batch['records']:
        ln = r.get('line', 0)
        if ln <= lo or ln >= hi:
            continue
        op = r.get('op')
        if op == 'kill':
            zk = rr.zone_kind(str(r.get('zone', '') or ''))
            toks = []
            for pc in (r.get('pieces') or []):
                toks.extend(_roll_tokens(pc))
            if zk[0] in ('region', 'stronghold'):
                for t in toks:
                    p = rr.parse_piece(t)
                    if p and p[0] == 'unit':
                        losses[_nat_side(p[1])] += p[3]
                        n += 1
            elif zk[0] in ('card', 'dice') and any('/' in t for t in toks):
                n += 1                         # round-end card discard
                ncard += 1
        elif op == 'move':
            fk = rr.zone_kind(str(r.get('from', '')))
            tk = rr.zone_kind(str(r.get('to', '')))
            if fk[0] in ('region', 'stronghold') and tk[0] in ('cas', 'reinf', 'card'):
                for t in _roll_tokens(r.get('obj')):
                    p = rr.parse_piece(t)
                    if p and p[0] == 'unit':
                        losses[_nat_side(p[1])] += p[3]
                        n += 1
            elif fk[0] in ('cas', 'reinf') and tk[0] in ('region', 'stronghold'):
                for t in _roll_tokens(r.get('obj')):
                    p = rr.parse_piece(t)
                    if p and p[0] == 'unit':
                        losses[_nat_side(p[1])] -= p[3]
                        n += 1
            elif (str(r.get('to', '')).strip().lower() in _CUR_CARD_ZONES
                  and '/' in str(r.get('obj', ''))):
                n += 1                         # combat-card commit
                ncard += 1
    return n, losses, ncard


def _combat_chunk_rounds(batch, leadership=None):
    """Segment the batch's numeric roll chunks into combat ROUNDS and class
    every chunk: -> [[(chunk, 'A'|'D', cls), ...], ...] with cls in 'pre'
    (Sudden Strike / Charge / Durin's Bane strike before the combat roll),
    'main', 're' (leader re-roll, possibly split over several chunks) and
    'post' (Onslaught / We Come to Kill after the casualties).

    `leadership` (optional) = {'A': n, 'D': n} upper bound of each side's
    re-roll leadership at the battle start; a chunk longer than that can
    never be a re-roll (a leaderless army that rolls again after both main
    volleys is opening the next round)."""
    ap = _combat_atk_player(batch)
    chunks = [r for r in batch['rng'] if r['kind'] == 'roll_num']
    if not chunks:
        return []
    atk = batch.get('die_side') or 'SHADOW'
    sstr = {'A': atk, 'D': _other_side(atk)}
    sides = ['A' if c['player'] == ap else 'D' for c in chunks]
    commits = _combat_card_commits(batch)
    used = set()

    def commit(kind, s, lo, hi, n=None, at=None):
        for k, (cl, sd, cls, fixed, dl) in enumerate(commits):
            if k in used or cls != kind or sd != sstr[s]:
                continue
            if not (lo < cl < hi):
                continue
            if n is not None and fixed is not None and n != fixed:
                continue
            if at is not None and dl is not None and dl < at:
                continue                       # already discarded: unrollable
            return k
        return None

    def later(i, s):
        return any(sides[j] == s for j in range(i + 1, len(sides)))

    def can_reroll(s, n):
        cap = leadership.get(s) if leadership else None
        if cap is None:
            return True                    # unknown army: no cap
        return n <= int(cap)

    rounds = []

    def open_round(lo):
        r = {'items': [], 'main': {}, 'main_ln': {}, 'last_ln': {}, 're': {}, 'lo': lo}
        rounds.append(r)
        return r

    def round_closed(cur, ln):
        # every combat card committed in this round was discarded after
        # the volleys rolled so far and before line ln: the round is over
        mains = list(cur['main_ln'].values())
        if not mains:
            return False
        after = max(mains)
        cs = [(cl, dl) for (cl, sd, cls, fixed, dl) in commits if cur['lo'] < cl < after]
        if not cs:
            return False
        return all(dl is not None and after < dl < ln for cl, dl in cs)

    def pre_ok(i, s, n, lo, hi):
        k = commit('pre', s, lo, hi, n)
        if k is None:
            return None
        o = 'D' if s == 'A' else 'A'
        if later(i, s) or not later(i, o):
            return k
        return None

    def start_round(cur_lo, i, s, ln):
        r = open_round(cur_lo)
        r['main'][s] = i
        r['main_ln'][s] = ln
        return r

    prev_ln = batch['start_line'] - 1
    cur = open_round(prev_ln)
    for i, (c, s) in enumerate(zip(chunks, sides)):
        o = 'D' if s == 'A' else 'A'
        n = len(c['ints'])
        ln = c['line']
        if s not in cur['main']:
            k = pre_ok(i, s, n, cur['lo'], ln)
            if k is not None:
                used.add(k)
                cls = 'pre'
            else:
                cls = 'main'
                cur['main'][s] = i
                cur['main_ln'][s] = ln
        elif o not in cur['main']:
            if round_closed(cur, ln):
                cur = start_round(prev_ln, i, s, ln)
                cls = 'main'
            else:
                cls = 're'
                cur['re'][s] = cur['re'].get(s, 0) + n
        else:
            # both main volleys rolled: a leader re-roll (possibly split), a
            # post-casualty card strike, or the first roll of the NEXT round
            k = pre_ok(i, s, n, cur['last_ln'][s], ln)
            if k is not None:
                used.add(k)
                cur = open_round(prev_ln)
                cls = 'pre'
            elif round_closed(cur, ln):
                cur = start_round(prev_ln, i, s, ln)
                cls = 'main'
            else:
                misses = sum(1 for d in chunks[cur['main'][s]]['ints'] if d < 6)
                cum = cur['re'].get(s, 0) + n
                fits = cum <= misses and can_reroll(s, cum)
                kpost = commit('post', s, cur['lo'], cur['main_ln'][s], at=ln)
                ev, losses, ncard = _combat_gestures(batch, prev_ln, ln)
                if kpost is not None and (not fits or cur['re'].get(s, 0) or ev):
                    used.add(kpost)
                    cls = 'post'
                elif fits and not ev:
                    cls = 're'
                    cur['re'][s] = cum
                elif fits:
                    base = len(chunks[cur['main'][s]]['ints'])
                    _, lo_s, _ = _combat_gestures(batch, cur['main_ln'][s], ln)
                    if base < 5:
                        exp = base - lo_s[sstr[s]]
                        is_re = (n < exp or exp <= 0)
                    else:
                        # a 5-die volley hides the unit count: a re-roll
                        # only while the opponent has not rolled since
                        is_re = not any(sides[j] == o
                                        for j in range(cur['main'][s] + 1, i)
                                        if chunks[j]['line'] > cur['last_ln'][s])
                    if is_re:
                        cls = 're'
                        cur['re'][s] = cum
                    else:
                        cur = start_round(prev_ln, i, s, ln)
                        cls = 'main'
                elif sides[i - 1] == s and losses[sstr[s]] == 0 and ncard == 0:
                    # over-count re-roll straight after this side's own
                    # volley: no casualty of this side and no card gesture
                    # separates them, so the round cannot have ended
                    cls = 're'
                    cur['re'][s] = cum
                else:
                    cur = start_round(prev_ln, i, s, ln)
                    cls = 'main'
        cur['items'].append((c, s, cls))
        cur['last_ln'][s] = ln
        prev_ln = ln
    return [r['items'] for r in rounds]


def _combat_engine_order(rounds):
    """Flatten classed rounds into the engine's randint consumption order:
    per round the pre-combat strikes (defender first - combat.py
    _resolve_pre_combat ties go to the defender), attacker main and
    re-rolls (all chunks), defender main and re-rolls, then the
    post-casualty strikes in log order."""
    flat = []
    for items in rounds:
        for c, s, cls in sorted((it for it in items if it[2] == 'pre'),
                                key=lambda it: (0 if it[1] == 'D' else 1)):
            flat += c['ints']
        for tag in ('A', 'D'):
            for stage in ('main', 're'):
                for c, s, cls in items:
                    if s == tag and cls == stage:
                        flat += c['ints']
        for c, s, cls in items:
            if cls == 'post':
                flat += c['ints']
    return flat


def _combat_rounds_and_order(batch, leadership=None):
    """Segment the numeric roll chunks into combat ROUNDS and re-serialise
    them into the engine's OWN randint call-order.

    The current engine uses the requested attacker main + re-roll, then
    defender main + re-roll convention. Human logs can use that order or
    interleave the sides. Classify chunks by side and stage before arranging
    them; neither the raw printed order nor the former engine order is a
    valid general substitute.

    Rounds come from _combat_chunk_rounds (dice counts + casualty/card
    gestures; one-sided card strikes and defender-first rounds no longer
    open phantom rounds - W2-008).  Returns (n_rounds,
    engine_order_flat_ints)."""
    rounds = _combat_chunk_rounds(batch, leadership)
    if not rounds:
        return 1, []
    return max(1, len(rounds)), _combat_engine_order(rounds)


_OFFBOARD_CAS_ZONES = ('cas', 'reinf', 'card', 'dice', 'hunt')


def _human_def_casualties(batch, regs):
    """The DEFENDER's chosen combat casualties from the rendered kill/removal
    gestures = every DEFENDER-side unit figure that LEFT the battle region(s)
    during the batch, whether by an explicit 'kills' line OR by a move to an
    off-board zone (casualties / reinforcements / a card slot / dice / hunt -
    the client sometimes renders a destroyed garrison as a move to the Current
    FP/SA Card instead of a kill).  A move to ANOTHER board region is a RETREAT
    (survivor, not a casualty) and is excluded - which is exactly why the
    casualty COUNT is the location-independent combat signal (retreat / muster
    into the region no longer pollute the comparison).  Leaders/Nazgul are not
    hit-absorbers and are not counted."""
    atk_side = batch['die_side']
    d = 0
    for r in batch['records']:
        if r['op'] == 'kill':
            zk = rr.zone_kind(r.get('zone', ''))
            if zk[0] in ('region', 'stronghold') and zk[1] in regs:
                for tok in r['pieces']:
                    p = rr.parse_piece(tok)
                    if p and p[0] == 'unit' and _nat_side(p[1]) != atk_side:
                        d += p[3]
        elif r['op'] == 'move':
            fk = rr.zone_kind(r.get('from', ''))
            tk = rr.zone_kind(r.get('to', ''))
            if fk[0] in ('region', 'stronghold') and fk[1] in regs \
                    and tk[0] in _OFFBOARD_CAS_ZONES:
                obj = r.get('obj', '')
                mg = re.match(r'^group\s*\((.*)\)$', obj, re.I)
                toks = rr.split_group(mg.group(1)) if mg else [obj]
                for tok in toks:
                    p = rr.parse_piece(tok)
                    if p and p[0] == 'unit' and _nat_side(p[1]) != atk_side:
                        d += p[3]
            # NET casualties (Columbo GAME05 b95, user-verified in the client):
            # a defender unit sent to the Casualties box and then pulled BACK
            # onto the board OUTSIDE the battle within the SAME batch is the
            # human's SELF-CORRECTION (a mis-click, then a retreat out of
            # casualties - L908 'kills [2]Gondor Regular' then L910 'Gondor
            # Regular from FP Casualties to Lamedon'), NOT a real casualty:
            # a casualty is a unit that ENDS out of play, so tally the NET.
            # RESTRICTED to a NON-battle destination so a legitimate Elite->
            # Regular DOWNGRADE (rb p.30: the replacement Regular is taken from
            # the casualties INTO the battle region) still counts as the one
            # hit it absorbs - that Regular returns to a battle region (in regs)
            # and is left alone; only a retreat/undo OUT of the battle nets.
            elif fk[0] == 'cas' and tk[0] in ('region', 'stronghold') \
                    and tk[1] not in regs:
                obj = r.get('obj', '')
                mg = re.match(r'^group\s*\((.*)\)$', obj, re.I)
                toks = rr.split_group(mg.group(1)) if mg else [obj]
                for tok in toks:
                    p = rr.parse_piece(tok)
                    if p and p[0] == 'unit' and _nat_side(p[1]) != atk_side:
                        d -= p[3]
    return max(0, d)


def _equivalent_defender_hit_capacity(batch, regs, casualties):
    """Sufficient proof for different legal figure counts at equal damage.

    Only compare an isolated defender army's loss: unrelated defender-unit
    actions invalidate this witness. A full Elite absorbs two hits; replacing
    it with a Regular absorbs one. This never infers a learned casualty choice.
    """
    from collections import Counter
    side = 'FREE_PEOPLES' if batch['die_side'] == 'SHADOW' else 'SHADOW'
    removed_elites = Counter()
    for record in batch['records']:
        if record.get('line', 0) < batch['start_line']:
            continue
        op = record.get('op')
        if op not in ('move', 'kill'):
            continue
        obj = record.get('obj', '')
        tokens = (record.get('pieces', ()) if op == 'kill' else
                  rr.split_group(obj[7:-1]) if obj.startswith('group (') and obj.endswith(')') else [obj])
        units = [p for token in tokens for p in [rr.parse_piece(token)]
                 if p and p[0] == 'unit' and _nat_side(p[1]) == side]
        if not units:
            continue
        source = rr.zone_kind(record.get('zone' if op == 'kill' else 'from', ''))
        target = rr.zone_kind(record.get('to', '')) if op == 'move' else None
        if source[0] in ('region', 'stronghold'):
            if source[1] not in regs:
                return False
            if target is None or target[0] in _OFFBOARD_CAS_ZONES:
                for _, nation, kind, count in units:
                    if kind == 'ELITE':
                        removed_elites[nation] += count
        elif target and target[0] in ('region', 'stronghold'):
            if target[1] not in regs or source[0] not in ('cas', 'reinf'):
                return False
            for _, nation, kind, count in units:
                if kind != 'REGULAR' or removed_elites[nation] < count:
                    return False
                removed_elites[nation] -= count
    def capacity(snapshot):
        return sum(2 if kind == 'ELITE' else 1
                   for units, _ in snapshot['armies'].values()
                   for nation, kind in units if _nat_side(nation) == side)
    observed = capacity(batch['truth_before']) - capacity(batch['truth'])
    simulated = sum(2 if unit.kind.name == 'ELITE' and not getattr(unit, '_downgraded', False)
                    else 1 for unit in casualties)
    return observed >= 0 and observed == simulated


def _card_combat_target(batch):
    """(target_side, affected_regs): the side that LOST army units net per
    region (truth_before -> truth_after) and the affected region set.  A card-
    effect combat (Ents Awake, Return to Valinor) removes units on ONE side -
    the target - so this identifies the victim from the board delta itself
    (the batch die_side is unreliable: the card can be an FP event glued behind
    a Shadow die)."""
    import collections
    tb = batch['truth_before']['armies']
    ta = batch['truth']['armies']
    affected = set()
    lost = collections.Counter()
    for reg in set(tb) | set(ta):
        rs = sanitize(reg)
        cb = collections.Counter(_nat_side(n) for (n, _k) in
                                 tb.get(reg, ([], []))[0])
        ca = collections.Counter(_nat_side(n) for (n, _k) in
                                 ta.get(reg, ([], []))[0])
        for side in ('SHADOW', 'FREE_PEOPLES'):
            if ca[side] < cb[side]:
                affected.add(rs)
                lost[side] += cb[side] - ca[side]
    target = lost.most_common(1)[0][0] if lost else None
    return target, affected


def _card_combat_human_cas(batch, regs, target_side):
    """HITS the target side absorbed at regs = NET unit loss (truth-snapshot
    delta, immune to transient UI parking gestures) PLUS downgrade REPLACEMENTS
    (target-side units brought in from reinforcements/casualties, each pairing a
    downgraded-elite kill).  A downgrade is 1 hit / net 0 (kill elite + add
    regular); an elimination is 1 hit / net -1; so hits = net_loss + repl.
    rb p.30 (casualty removal: remove a Regular OR replace an Elite with a
    Regular from casualties/reinforcements)."""
    import collections
    tb, ta = batch['truth_before']['armies'], batch['truth']['armies']

    def side_ct(snap):
        c = collections.Counter()
        for reg, (units, _l) in snap.items():
            if sanitize(reg) in regs:
                for (n, _k) in units:
                    if _nat_side(n) == target_side:
                        c[sanitize(reg)] += 1
        return c
    cb, ca = side_ct(tb), side_ct(ta)
    net_loss = sum(max(0, cb[r] - ca[r]) for r in regs)
    repl = 0
    for r in batch['records']:
        if r.get('op') != 'move':
            continue
        fk = rr.zone_kind(r.get('from', ''))
        tk = rr.zone_kind(r.get('to', ''))
        if fk[0] in ('reinf', 'cas') and tk[0] in ('region', 'stronghold') \
                and tk[1] in regs:
            obj = r.get('obj', '')
            mg = re.match(r'^group\s*\((.*)\)$', obj, re.I)
            toks = rr.split_group(mg.group(1)) if mg else [obj]
            for tok in toks:
                p = rr.parse_piece(tok)
                if p and p[0] == 'unit' and _nat_side(p[1]) == target_side:
                    repl += p[3]
    return net_loss + repl


def _drive_card_effect_combat(st, batch):
    """A NO-army-plan combat that is actually an EVENT-CARD effect (Ents Awake
    hits Orthanc, Return to Valinor hits Elven strongholds): the card's own
    engine handler inflicts the unit removals, not game_logic.battle.  Dispatch
    the registered handler under the LOGGED dice (served to each of the
    handler's d6 roll requests by matching the requested count) and compare the
    CASUALTY COUNT the engine inflicted to the human's absorbed hits - the same
    count-level signal the certified _drive_combat gate uses (casualty
    composition = the loser's legal allocation choice, rb p.30, never a bug).

    Reliability (anti-fluke): a handler roll request that the LOGGED chunks
    cannot serve (dice-count mismatch = a human dice anomaly or a wrong card
    model) makes the drive UNRELIABLE -> honest skip, never a random-fallback
    match.  Skip-on-mismatch throughout (no resync divergence)."""
    import random
    import types
    import event_effects as ee
    import game_logic as gl
    import combat as cmb
    cards = _played_event_cards(batch)
    # ALL co-played cards with a registered handler, in log order: a card-effect
    # combat can be a COMPOUND play (the Ents Awake attack + a free-chained
    # Bilbo's Song corruption heal, MrM0th b49).  Dispatch each in order under
    # the forced dice; skip-on-mismatch still guards a wrong compound.
    handlers = [(ee.CARD_EFFECT_HANDLERS.get(_card_key(c)), c) for c in cards]
    handlers = [(h, c) for (h, c) in handlers if h is not None]
    if not handlers:
        return ('skip', [], 'combat:no-plan')
    cardname = handlers[0][1]
    chunks = [list(r['ints']) for r in batch['rng'] if r['kind'] == 'roll_num']
    meter = {'fallback': False, 'cas': 0}

    def _served(n, ctx=''):
        for i, ch in enumerate(chunks):
            if len(ch) == n:
                return chunks.pop(i)
        meter['fallback'] = True
        return [random.randint(1, 6) for _ in range(n)]

    orig_ac = cmb.apply_casualties

    def _ac(army, hits, policy='preserve_elites'):
        r = orig_ac(army, hits, policy)
        meter['cas'] += len(r)
        return r
    orig_roll, orig_ch = gl.roll_d6_batch, random.choice
    gl.roll_d6_batch = _served
    cmb.apply_casualties = _ac
    random.choice = lambda s2: s2[-1]
    try:
        for h, c in handlers:
            h(st, types.SimpleNamespace(card_name=c, card_id=c), {})
    except Exception as e:
        return ('skip', [], 'combat:card-effect-error(%s)' % str(e)[:24])
    finally:
        gl.roll_d6_batch, cmb.apply_casualties, random.choice = \
            orig_roll, orig_ac, orig_ch
    if meter['fallback']:
        return ('skip', [], 'combat:card-effect-dice(%s)' % cardname[:12])
    ta = batch['truth']['fel']
    if st.fellowship.corruption != ta['corruption']:
        return ('skip', [], 'combat:card-effect-fel(%s)' % cardname[:12])
    target, affected = _card_combat_target(batch)
    if not target or not affected:
        return ('skip', [], 'combat:card-effect-noregion(%s)' % cardname[:8])
    hum = _card_combat_human_cas(batch, affected, target)
    if meter['cas'] > 0 and meter['cas'] == hum:
        return ('drove', [], None)
    if meter['cas'] == 0 and hum == 0:
        # anti-fluke: a 0>0 match is required to count as driven, but
        # 0==0 is not a mismatch either
        return ('skip', [], 'combat:card-effect-nocas')
    return ('skip', [], 'combat:card-effect-cas(%d!=%d)' % (meter['cas'], hum))


def _drive_combat(st, batch):
    """Drive the engine's own combat resolver (game_logic.battle) from the
    rendered mirror under FORCED, RE-ORDERED dice, and compare the ENGINE's
    computed DEFENDER casualty COUNT against the human's chosen defender
    casualties (kills + off-board removals).

    Wiring (W-F, directive - 'why can't the engine do carded combats and
    retreats'): every player DECISION is fed as the engine's own input and only
    the RNG is forced:
      * combat-card plays - card ids resolved through the imported closed_loop
        lookup and passed to battle();
      * round continuation / press - the round count is INFERRED from the
        attacker/defender main-volley pairing in the roll stream (max_rounds);
      * casualty ALLOCATION - not fed at all: the engine applies its own hits,
        and we compare only the COUNT (allocation composition is the loser's
        legal choice, never a bug), so nothing is clamped;
      * retreat / withdraw / muster - handled implicitly by the casualty-COUNT
        signal, which is location-independent (a retreated survivor is not a
        casualty; a mustered reinforcement is not one either).

    RELIABILITY GATE.  A defender-count divergence gate-counts (a real engine
    finding) ONLY when the drive is provably clean: a single unambiguous
    attacker->defender split (defender region one-sided; field battles need a
    one-sided distinct attacker region, sieges use the co-located garrison
    split), UN-carded, and the engine consumed the RE-ORDERED forced stream
    EXACTLY (every int, no miss-fallback).  Otherwise the divergence is
    reported but NOT gate-counted (carded choices not supplied / dice structure
    diverged / plan ambiguous)."""
    import random
    import game_logic as gl
    # A combat batch whose board is UNCHANGED end-to-end is an UNDONE / aborted
    # attack (GAME21 b226: 'Polijuna undoes ... kills Elven Elite' -> net-zero
    # no-op; there was no dice-resolved battle to drive).  Reproduce it as
    # bookkeeping (the mirror already holds the correct = unchanged board).
    if batch['truth_before'].get('armies') == batch['truth'].get('armies') \
            and batch['truth_before'].get('fel') == batch['truth'].get('fel'):
        return ('drove', [], None)
    chunks = [r for r in batch['rng'] if r['kind'] == 'roll_num']
    if not chunks:
        return ('skip', [], 'combat:no-dice(card-effect/automatic)')
    # The attacker's Combat Strength = his FIRST main-volley chunk size, which
    # must be the ATTACKER's roll, not chunks[0]: the two players roll
    # simultaneously (rb p.30) and the client sometimes prints the DEFENDER's
    # volley first (COMBAT-2 Family-B), so chunks[0] would grab the wrong
    # side's die count.  Pick the first chunk rolled by the staging player.
    _ap = _combat_atk_player(batch)
    _atk_chunks = [c for c in chunks if c.get('player') == _ap] if _ap else []
    first_dice = len((_atk_chunks or chunks)[0]['ints'])
    carded = _batch_has_combat_card(batch)
    plan = _plan_combat(st, batch,
                        expected_dice=None if carded else first_dice)
    if plan is None:
        # No army-attack plan: the batch may be an EVENT-CARD effect combat
        # (Ents Awake / Return to Valinor) whose own handler inflicts the
        # removals.  Route it there (returns the same 'combat:no-plan' skip if
        # no registered card handler applies).
        return _drive_card_effect_combat(st, batch)
    atk, dfn, is_siege = plan
    if not carded:
        # The attacker's PHYSICAL first throw = Combat Strength volley, possibly
        # with the Leader re-roll merged into the same throw (rb p.30 L2714-2727:
        # Leadership re-rolls up to Leadership failed dice).  Combat Strength =
        # Army units capped at 5, +1 if Strider "Captain of the West" leads
        # (rb Strider card).  So accept first_dice in [strength, strength +
        # min(5, Leadership)] and let the engine split volley/re-roll itself;
        # only a TRUE strength mismatch (wrong attacker region) is skipped.
        _units = _atk_units_at(st, atk, batch['die_side'])
        _strider = _leadership = 0
        for _kk, _aa in st.armies.items():
            if sanitize(_kk) != atk:
                continue
            for _l in _aa.leaders:
                _nm = str(getattr(_l, 'name', '') or '')
                if 'Strider' in _nm or 'Aragorn' in _nm:
                    _strider = 1
                _leadership += int(getattr(_l, 'leadership', 1) or 1)
        strength = min(5, _units + _strider)
        if not (strength <= first_dice <= strength + min(5, _leadership)):
            return ('skip', [],
                    'combat:atk-strength-mismatch(%d!=%d)'
                    % (strength, first_dice))
    # count the human's DEFENDER casualties over the kill-based battle regions
    # PLUS the planned defender region: when the defender's loss rendered as a
    # move to a Current Card / Casualties slot (not a 'kills' line), its region
    # is absent from _battle_regions, so without it the count was 0 while the
    # engine (now correctly planned) computed the real casualties.
    regs = set(rr._battle_regions(batch)) | {dfn}
    nrounds, rolls = _combat_rounds_and_order(batch)
    acid = dcid = None
    if carded:
        acid, dcid = _combat_cards(batch)
    hum_def = _human_def_casualties(batch, regs)
    st_seq = iter(rolls)
    meter = {'n': 0, 'fb': False}

    def _forced(a, b2, _s=st_seq, _m=meter):
        try:
            v = next(_s)
            _m['n'] += 1
            return v
        except StopIteration:
            _m['fb'] = True
            return 1
    orig_ri = random.randint
    orig_ch = random.choice
    random.randint = _forced
    random.choice = lambda s2: s2[-1]
    try:
        res = gl.battle(st, atk, dfn, attacker_card_id=acid,
                        defender_card_id=dcid, is_siege=is_siege,
                        max_rounds=nrounds, auto_press=False,
                        # Onslaught's sacrifice count is a learned
                        # post-casualty DECISION (spec 1.216): without
                        # defer_onslaught the engine auto-applies the
                        # MAX sacrifice (combat.py ~L1737) - killing
                        # units the human never chose to lose, failing
                        # the settled-totals gate, and the pending the
                        # back-calc emitter waits on never opens (USER
                        # 2026-07-24)
                        choices={'defer_onslaught': True})
    except Exception as e:
        random.randint, random.choice = orig_ri, orig_ch
        # co-located / friendly-army plan failures are AMBIGUOUS-plan skips,
        # not engine errors (the standalone plan could not split the stack)
        return ('skip', [], 'combat:plan-error(%s)' % str(e)[:40])
    finally:
        random.randint, random.choice = orig_ri, orig_ch
    # ONSLAUGHT RESUME (USER 2026-07-24): with defer_onslaught the engine
    # returns deferred_effect UNRESOLVED (the pending translation lives in
    # ai's self-play loop, not battle()).  Resume with the back-calculated
    # sacrifice K = engine's surviving own units MINUS the client's settled
    # own units at the battle region (the human's chosen self-hits), capped
    # to the legal menu.  Enemy destroyed => onslaught is MOOT (user
    # ruling; combat.py L1720 already skips, deferred_effect stays None).
    _role = getattr(res, 'deferred_effect', None)
    if _role in ('attacker', 'defender'):
        try:
            import combat as _cmb
            import ai as _ai9
            import game_state as gs
            _own_r = atk if _role == 'attacker' else dfn
            _enemy_r = dfn if _role == 'attacker' else atk
            # sides from the drive's own PLAN (die_side = the attacker;
            # occupancy is useless in co-located sieges where both
            # stacks share the region key - Erebor class)
            _atk_enum = (gs.Side.SHADOW if str(batch['die_side'])
                         .upper().startswith('SH')
                         else gs.Side.FREE_PEOPLES)
            _dfn_enum = (gs.Side.FREE_PEOPLES
                         if _atk_enum == gs.Side.SHADOW
                         else gs.Side.SHADOW)
            _own_side = (_atk_enum if _role == 'attacker'
                         else _dfn_enum)
            _enemy_side = (_dfn_enum if _role == 'attacker'
                           else _atk_enum)
            _mx = _cmb.onslaught_max_sacrifice(st, _own_r, _own_side)
            # human's chosen self-hits = engine's surviving own units
            # minus the client's settled own units at the region
            _, _a3 = _cmb._army_entry(st, _own_r)
            _eng_own = (len([u for u in _a3.units
                             if _ai9._side_of_nation(u.nation)
                             == _own_side]) if _a3 else 0)
            _tru = (batch.get('truth') or {}).get('armies', {})
            _tru_own = 0
            for _rk, (_us, _ls) in _tru.items():
                if sanitize(str(_rk)) == sanitize(_own_r):
                    _tru_own = len([1 for (n2, k2) in _us
                                    if _ai9._side_of_nation(
                                        gs.Nation[n2]) == _own_side])
                    break
            _k = max(0, min(_mx, _eng_own - _tru_own))
            # PHANTOM GUARD (USER truth-table fix #10, GAME01 T7):
            # a real Onslaught sacrifice renders as the owner's
            # units PARKED on a card/action zone mid-battle (T3
            # L325 '[4]Sauron Regular -> Current SA Card') or
            # killed in the cleanup group.  Without that gesture,
            # a drive-divergence delta must NOT mint a magnitude
            # decision (T7: FUh Board6->Current cycling produced a
            # phantom K=1).
            _park = False
            for _r9 in batch.get('records') or []:
                if _r9.get('op') not in ('move', 'kill'):
                    continue
                _ob = str(_r9.get('obj', '') or '')
                _to9 = str(_r9.get('to', '') or '')
                if (('Current' in _to9 or _r9.get('op') == 'kill')
                        and ('Regular' in _ob or 'Elite' in _ob)
                        and 'Action Die' not in _ob
                        and sanitize(str(_r9.get('from', '') or '')
                                     ) == sanitize(_own_r)):
                    _park = True
                    break
            if _mx > 0 and _park:
                # surface the DECISION POINT for the emitter (a
                # clone of the state BEFORE the sacrifice + the
                # legal menu bounds + the back-calced K) - the
                # sample itself packs in rendered_extract, which
                # owns encode/emit (spec 1.216: onslaught magnitude
                # is a learned decision, engine back-calculated)
                try:
                    import ai as _ai10
                    st._onslaught_decision = {
                        'state': _ai10.clone_state(st),
                        'own_r': _own_r, 'enemy_r': _enemy_r,
                        'own_side': _own_side,
                        'enemy_side': _enemy_side,
                        'k': _k, 'mx': _mx,
                        'role': _role,
                        'card_id': (acid if _role == 'attacker'
                                    else dcid) or ''}
                except Exception as _es:
                    st._onslaught_stash_err = str(_es)[:60]
                _cmb.apply_onslaught_choice(
                    st, _own_r, _enemy_r, _own_side, _enemy_side, _k,
                    enemy_is_defender=(_role == 'attacker'))
        except Exception as _eo:
            st._onslaught_stash_err = getattr(
                st, '_onslaught_stash_err', None) or str(_eo)[:60]
    dice_exact = (meter['n'] == len(rolls)) and not meter['fb']
    eng_def = len(res.defender_casualties)
    if eng_def == hum_def:
        return ('drove', [], None)                       # MATCHED
    # mismatch: is the split unambiguous enough to gate-count?
    atk_sides = _snap_sides_at(batch['truth_before'], atk)
    dfn_sides = _snap_sides_at(batch['truth_before'], dfn)
    clean = (len(dfn_sides) == 1 and
             (is_siege or (len(atk_sides) == 1 and atk != dfn)))
    reliable = clean and dice_exact and not carded
    if reliable and _equivalent_defender_hit_capacity(batch, regs, res.defender_casualties):
        cls = 'combat-casualty-policy'
        reliable = False
    elif reliable:
        cls = 'combat-hits'
    elif carded:
        cls = 'combat-complex(carded)'
    elif not dice_exact:
        cls = 'combat-complex(dice-inexact:%d/%d%s)' % (
            meter['n'], len(rolls), '+fb' if meter['fb'] else '')
    else:
        cls = 'combat-complex(plan-ambiguous)'
    return ('drove', [{'field': 'def_cas@' + dfn, 'eng': eng_def,
                       'truth': hum_def, 'class': cls, 'carded': carded,
                       'resync': reliable}], None)


# Companion LEVEL (hunt damage absorbed when eliminated to the Hunt).  A
# Companion sacrificed to the Hunt reduces the tile damage by its Level; the
# remaining damage becomes corruption (rulebook hunt-resolution).  Crowned /
# transformed figures keep their level; Gollum has none.
_COMP_LEVEL = {'Gandalf the Grey': 3, 'Strider': 3, 'Aragorn': 3,
               'Gandalf the White': 3, 'Legolas': 2, 'Gimli': 2,
               'Boromir': 2, 'Meriadoc': 1, 'Peregrin': 1, 'Gollum': 0}


def _has_fsp_move(batch):
    """The batch physically advances the Fellowship pawn (an 'FSP' feltrack
    gesture) - the signature of a genuine FREE-PEOPLES fellowship MOVE (vs a
    Shadow hunt-tile card play or an FP corruption-heal card that also draws
    a tile but does NOT move the pawn)."""
    for r in batch['records']:
        if r.get('op') == 'move' and str(r.get('obj', '')).strip().lower() \
                == 'fsp':
            return True
    return False


def _played_event_cards(batch):
    """List the EVENT names (substring before '/') of every event card played
    this batch - each '/'-titled card moved into a Current FP/SA Card slot.
    Distinguishes an event-card hunt-effect PLAY from a Fellowship MOVE (a
    genuine move never plays a card into that slot).  More than one = an
    interpose (a mitigation/redraw card such as Mithril Coat and Sting or
    Wizard's Staff joining the hunt), a compound resolution."""
    out = []
    for r in batch['records']:
        if r.get('op') != 'move':
            continue
        to = str(r.get('to', '')).lower()
        obj = str(r.get('obj', ''))
        if 'current' in to and 'card' in to and '/' in obj:
            out.append(obj.split('/')[0].strip())
    return out


def _played_event_card(batch):
    """The first event card played this batch, or None."""
    cs = _played_event_cards(batch)
    return cs[0] if cs else None


def _card_key(name):
    """Normalise a rendered card name for handler lookup: the client renders
    some titles with hyphens for spaces (Foul-Thing-From-the-Deep), others
    with spaces (Orc Patrol)."""
    import re as _re2
    return _re2.sub(r'\s+', ' ', str(name).replace('-', ' ')).strip().lower()


# Event cards whose hunt-effect is the STANDARD tile-hunt-damage mechanic:
# the card draws one Hunt tile and the FP takes its value as Hunt damage -
# absorbed as a Companion casualty (its Level) with the excess as Corruption,
# or entirely as Corruption.  rb p.41 (data/_rulebook.txt L3691-3700): "If the
# tile is numbered, its value represents the Hunt damage inflicted to the
# Fellowship ... If the tile shows an Eye ... (due to an Event card, the Eye
# is considered to have a value of 0)"; rb p.41-42 (L3780-3790) casualty-Level
# absorption + Using the Ring.  All three cards share this exact card text, so
# ONE general driver - the engine's own game_logic._apply_fellowship_damage
# primitive fed the logged tile + logged companion casualty - resolves them
# with no per-card branching.  Dice-roll cards (Candles of Corpses / Athelas -
# guide-threshold dependent), companion-SEPARATION cards (The Breaking of the
# Fellowship), and reveal-CHAIN cards (Nazgul Search / Strike) are a different
# mechanic and keep a narrower per-card skip until their own driver lands.
_TILE_HUNT_DAMAGE_CARDS = {
    "orc patrol", "isildur's bane", "foul thing from the deep"}

# Free-Peoples HUNT-PROTECTION cards played on the table ("Play on the table"):
# these are the ONLY cards that turn a single-tile hunt into a genuine COMPOUND
# resolution - Axe and Bow / Horn of Gondor reduce the Hunt damage (one per
# tile), Mithril Coat and Sting redraws the tile, Wizard's Staff prevents it.
# alm p.3/p.11 (scratchpad hunt-protection order).  A card co-played into a
# hunt batch that is NOT one of these (a Shadow attack event, a card-discard
# event, a taken-back muster) never touches the Hunt and is deduped.
_MITIGATION_CARDS = {
    "axe and bow", "horn of gondor", "mithril coat and sting",
    "wizard's staff"}

# Event cards whose hunt-effect is a 3-DICE roll (no tile): Athelas heals
# Corruption per die >= threshold (3+ with Strider guiding, else 5+); Candles
# of Corpses adds Corruption per die >= threshold (6+ with Gollum guiding, else
# 4+).  The engine's own registered handlers encode each card's rule, so we
# force the logged dice and dispatch the handler (no threshold hardcoded here).
_DICE_HUNT_CARDS = {"athelas", "candles of corpses"}


def _drive_dice_card_hunt(st, batch, key, cardname):
    """Drive a 3-dice hunt-effect card (Athelas / Candles of Corpses) through
    the engine's own registered handler under the FORCED logged dice + the
    reconstructed Guide (batch['_guide'] sets the threshold).  Compare fel
    corruption vs client truth.  Gated to an exactly-3-dice roll (a different
    count = a combat/other roll mixed into the batch); a residual corruption
    mismatch means another effect ran in the batch, so it is an HONEST skip -
    never a masked policy note (Mordor-Eye lesson 2026-07-25)."""
    import random
    import types
    import event_effects as ee
    rolls = [r for r in batch['rng'] if r['kind'] == 'roll_num']
    if not rolls or len(rolls[0]['ints']) != 3:
        return ('skip', [], 'hunt:card-dice(%s)' % cardname[:10])
    handler = ee.CARD_EFFECT_HANDLERS.get(key)
    if handler is None:
        return ('skip', [], 'hunt:card(%s)' % cardname[:16])
    gname = batch.get('_guide')
    if gname:
        gobj = next((c for c in st.fellowship.companions
                     if c.name == gname), None)
        if gobj is None:                     # e.g. Gollum (not reconstructed)
            gobj = types.SimpleNamespace(name=gname, level=0,
                                         guide_ability=None)
        st.fellowship.guide = gobj
    ta = batch['truth']['fel']
    seq = iter(list(rolls[0]['ints']))
    o_ri = random.randint
    random.randint = lambda a, b2, _i=seq, _f=o_ri: next(_i, _f(a, b2))
    try:
        handler(st, None, {})
    except Exception as e:
        return ('error', [{'why': 'dice-card-exception', 'err': str(e)[:80]}],
                'hunt:card-error')
    finally:
        random.randint = o_ri
    if st.fellowship.corruption != ta['corruption']:
        return ('skip', [], 'hunt:card-dice-other(%s)' % cardname[:8])
    return ('drove', [], None)


# 'The Breaking of the Fellowship' (Shadow) is a companion-SEPARATION card:
# draw one Hunt tile and the FP separates that many Companions to the Ring-
# bearers' region (Eye / Fellowship-special tile = discard, no effect; on the
# Mordor Track separation = removal; Gollum guide = +1 Corruption instead).
# rb p.38-39 (Separating Companions) + rb p.43 (Mordor).
_SEPARATION_CARDS = {"the breaking of the fellowship"}

# 'Challenge of the King' (Shadow) is a hunt-POOL card: draw 3 tiles; if all 3
# show Eyes put them back and eliminate Strider/Aragorn, otherwise permanently
# remove the drawn Eye tiles.  It has NO fellowship effect except the 3-Eyes
# elimination, so driving the engine's own handler over the 3 logged tiles and
# comparing fel state is an honest empty-diff check (the pool bookkeeping the
# mirror does not model is invisible either way).  rb card #15.
_POOL_HUNT_CARDS = {"challenge of the king"}


def _drive_challenge(st, batch, cardname):
    """Drive 'Challenge of the King' through its engine handler over the 3
    logged Hunt tiles; compare fel comps (Gollum-normalized) + corruption +
    hidden.  Skip-on-mismatch (honest, never masked)."""
    import event_effects as ee
    import game_state as gs
    ta = batch['truth']['fel']
    tb = batch['truth_before']['fel']
    tiles = [r for r in batch['rng'] if r['kind'] == 'draw_tile']
    if len(tiles) != 3:                       # a clean Challenge draws 3
        return ('skip', [], 'hunt:card(Challenge of the)')
    pool = []
    for t in tiles:
        fo = _tile_obj(gs, str(t['value']).strip().lower())
        if fo is None:
            return ('skip', [], 'hunt:card(Challenge of the)')
        pool.append(fo)
    st.hunt_pool = pool
    handler = ee.CARD_EFFECT_HANDLERS.get('challenge of the king')
    if handler is None:
        return ('skip', [], 'hunt:card(Challenge of the)')
    try:
        handler(st, None, {})
    except Exception as e:
        return ('error', [{'why': 'challenge-exception', 'err': str(e)[:80]}],
                'hunt:card-error')
    eng_comps = sorted(c.name for c in st.fellowship.companions
                       if c.name != 'Gollum')
    ok = (st.fellowship.corruption == ta['corruption']
          and eng_comps == sorted(c for c in ta['comps'] if c != 'Gollum')
          and (tb['mordor'] or st.fellowship.hidden == ta['hidden']))
    if not ok:
        return ('skip', [], 'hunt:card-challenge-complex')
    return ('drove', [], None)


def _drive_breaking(st, batch, cardname):
    """Drive 'The Breaking of the Fellowship' through the engine's own handler:
    force the logged Hunt tile, separate the logged Companions (the FP's learned
    choice, taken from the snapshot comps-diff), and compare fel comps
    (Gollum-normalized) + corruption + hidden vs client truth-after.  A residual
    mismatch = a compound play (a second card, a concurrent effect) or a batch-
    segmentation artifact = HONEST skip, never masked (Mordor-Eye doctrine)."""
    import types
    import event_effects as ee
    import game_state as gs
    tb = batch['truth_before']['fel']
    ta = batch['truth']['fel']
    tiles = [r for r in batch['rng'] if r['kind'] == 'draw_tile']
    if len(tiles) != 1:
        return ('skip', [], 'hunt:card-multi-tile(Breaking)')
    fo = _tile_obj(gs, str(tiles[0]['value']).strip().lower())
    if fo is None:
        return ('skip', [], 'hunt:card-special-tile(Breaking)')
    # a die-icon tile (Shelob's Lair) carries its separate-count in the d6 roll
    if getattr(fo, 'is_dice_roll', False):
        rn = [r for r in batch['rng'] if r['kind'] == 'roll_num']
        if rn and rn[0]['ints']:
            fo.damage = int(rn[0]['ints'][0])
    # Gollum-normalise: VASSAL lists a 'Gollum' counter in the Fellowship box
    # once he is the sole occupant; the engine never lists Gollum in companions.
    cb = [c for c in tb['comps'] if c != 'Gollum']
    ca = [c for c in ta['comps'] if c != 'Gollum']
    separated = [c for c in cb if c not in ca]
    # Gollum is the Guide once every real Companion has left (rb p.39): then the
    # card adds +1 Corruption instead of separating.
    if ('Gollum' in tb['comps']) and not cb:
        st.fellowship.guide = types.SimpleNamespace(
            name='Gollum', level=0, guide_ability=None)
    st.hunt_pool = [fo]
    handler = ee.CARD_EFFECT_HANDLERS.get('the breaking of the fellowship')
    if handler is None:
        return ('skip', [], 'hunt:card(The Breaking of )')
    try:
        handler(st, None, {'separate': separated})
    except Exception as e:
        return ('error', [{'why': 'breaking-exception', 'err': str(e)[:80]}],
                'hunt:card-error')
    eng_comps = sorted(c.name for c in st.fellowship.companions
                       if c.name != 'Gollum')
    ok = (st.fellowship.corruption == ta['corruption']
          and eng_comps == sorted(ca)
          and (tb['mordor'] or st.fellowship.hidden == ta['hidden']))
    if not ok:
        return ('skip', [], 'hunt:card-breaking-complex(%s)' % cardname[:6])
    return ('drove', [], None)


def _nazgul_flights(batch):
    """Reconstruct the logged Nazgul flights as the engine's own
    {'from','to','n','nmin'} list (n = plain Nazgul, nmin = named minions:
    Witch-king / Mouth of Sauron / the Black Captain) from the batch's
    region->region move records.  rb p.23: one Character die / event moves any
    or all Nazgul to any region."""
    moves = []
    for r in batch['records']:
        if r.get('op') != 'move':
            continue
        fk = rr.zone_kind(r.get('from', ''))
        tk = rr.zone_kind(r.get('to', ''))
        if fk[0] not in ('region', 'stronghold') \
                or tk[0] not in ('region', 'stronghold'):
            continue
        obj = str(r.get('obj', ''))
        toks = (rr.split_group(obj.replace('group (', '').rstrip(')'))
                if 'group' in obj.lower() else [obj])
        n = nmin = 0
        for t in toks:
            p = rr.parse_piece(t)
            if not p:
                continue
            if p[0] == 'nazgul':
                n += int(p[3] or 1)
            elif p[0] == 'named' and str(p[1]) in ('Witch King',
                                                   'Mouth of Sauron'):
                nmin += 1
            elif p[0] == 'unknown' and 'black captain' in str(p[1]).lower():
                nmin += 1
        if n or nmin:
            moves.append({'from': fk[1], 'to': tk[1], 'n': n, 'nmin': nmin})
    return moves


def _drive_nazgul_reveal_chain(st, batch, key, cardname):
    """Drive 'Nazgul Search' / 'The Nazgul Strike!' through the engine's own
    handler on the TEACHER path: fly the logged Nazgul, let the handler's
    co-location reveal-check fire, then apply the logged Hunt tile.  rb p.23
    (Nazgul move) + p.37 (reveal on co-location / event) + p.41 (hunt damage;
    a card-drawn Eye = 0, a Strike-hunt Eye = successes).  Compare corruption +
    comps; fel.hidden is compared for SEARCH (co-location is deterministic) but
    NOT for STRIKE - the rendered log collapses an 'Eye' and an 'Eye+Reveal'
    tile into 'Eye', so the reveal bit is a representation gap (like Mordor).
    Skip-on-mismatch (honest, never masked)."""
    import combat as _cmb
    import event_effects as ee
    import game_logic as gl
    tb = batch['truth_before']['fel']
    ta = batch['truth']['fel']
    is_mordor = tb['mordor']
    is_strike = 'strike' in key
    handler = (ee._effect_nazgul_strike if is_strike
               else ee._effect_nazgul_search)
    moves = _nazgul_flights(batch)
    tiles = [r for r in batch['rng'] if r['kind'] == 'draw_tile']
    if len(tiles) > 1:                         # multi-tile reveal chain
        return ('skip', [], 'hunt:card-multi-tile(Nazgul)')
    damage, reveal_tile = 0, False
    if tiles:
        tn = str(tiles[0]['value']).strip().lower()
        m = re.match(r'^(\d+)(r?)$', tn)
        if m:
            damage, reveal_tile = int(m.group(1)), bool(m.group(2))
        elif tn == 'eye':
            if is_strike:
                rn = [r for r in batch['rng'] if r['kind'] == 'roll_num']
                damage = sum(1 for v in (rn[0]['ints'] if rn else [])
                             if v >= 6)          # successes (rb p.41)
            else:
                damage = 0                       # card-drawn Eye = 0
        else:
            return ('skip', [], 'hunt:card-special-tile(Nazgul)')
    cb = [c for c in tb['comps'] if c != 'Gollum']
    ca = [c for c in ta['comps'] if c != 'Gollum']
    left = [c for c in cb if c not in ca]
    if len(left) > 1:
        return ('skip', [], 'hunt:card-multi-companion(Nazgul)')
    victim = left[0] if (left and damage > 0) else None
    _saved = _cmb.FINAL_HITS_HOOK
    _cmb.FINAL_HITS_HOOK = (lambda *a, **k: ([], []))   # activate teacher path
    try:
        ch = {'nazgul_moves': moves}
        if is_strike:
            ch.update({'action': 'hunt', '_log_forced_then': True,
                       '_pure_open_hunt': True})
        handler(st, None, ch)
        if damage > 0:
            gl._apply_fellowship_damage(st, damage, kill_name=victim)
        if reveal_tile:
            st.fellowship.hidden = False
    except Exception as e:
        _cmb.FINAL_HITS_HOOK = _saved
        return ('error', [{'why': 'nazgul-exception', 'err': str(e)[:80]}],
                'hunt:card-error')
    finally:
        _cmb.FINAL_HITS_HOOK = _saved
    if st.fellowship.corruption != ta['corruption']:
        return ('skip', [], 'hunt:card-nazgul-other(%s)' % cardname[:6])
    eng_comps = sorted(c.name for c in st.fellowship.companions
                       if c.name != 'Gollum')
    if eng_comps != sorted(ca):
        return ('skip', [], 'hunt:card-nazgul-comps(%s)' % cardname[:6])
    if (not is_mordor) and (not is_strike) \
            and st.fellowship.hidden != ta['hidden']:
        return ('skip', [], 'hunt:card-nazgul-hidden(%s)' % cardname[:6])
    return ('drove', [], None)


def _card_discarded(batch, cardkey):
    """True if the named card was DISCARDED (killed to a Current/discard zone)
    this batch - i.e. it was USED (a Mithril redraw / an Axe-Horn reduction that
    ends in the discard pile), as opposed to returned to the table/hand."""
    for r in batch['records']:
        if r.get('op') == 'kill':
            for p in (r.get('pieces') or []):
                if _card_key(str(p).split('/')[0]) == cardkey:
                    return True
    return False


def _drive_mithril_redraw(st, batch, key, cardname):
    """Mithril Coat and Sting redraw on a tile-hunt-damage card: the first tile
    is set aside and returned to the pool, the SECOND (redrawn) tile is applied
    as a normal hunt (alm p.11; rb p.41 tile-hunt / p.42 casualty absorption;
    card-drawn Eye = 0).  Skip-on-mismatch (never a resync divergence)."""
    import game_logic as gl
    tb = batch['truth_before']['fel']
    ta = batch['truth']['fel']
    is_mordor = tb['mordor']
    tiles = [r for r in batch['rng'] if r['kind'] == 'draw_tile']
    if len(tiles) < 2:
        return ('skip', [], 'hunt:card-interpose(%s)' % cardname[:12])
    tv = str(tiles[-1]['value']).strip().lower()      # the redrawn tile
    if tv == 'eye':
        dmg, rev = 0, False
    else:
        m = re.match(r'^(\d+)(r?)$', tv)
        if not m:
            return ('skip', [], 'hunt:card-mithril-sp(%s)' % tv[:6])
        dmg, rev = int(m.group(1)), bool(m.group(2))
    cb = [c for c in tb['comps'] if c != 'Gollum']
    ca = [c for c in ta['comps'] if c != 'Gollum']
    left = [c for c in cb if c not in ca]
    if len(left) > 1:
        return ('skip', [], 'hunt:card-mithril-comp(%s)' % cardname[:8])
    victim = left[0] if (left and dmg > 0) else None
    try:
        gl._apply_fellowship_damage(st, dmg, kill_name=victim)
    except Exception as e:
        return ('error', [{'why': 'mithril-exc', 'err': str(e)[:60]}],
                'hunt:card-error')
    if rev:
        st.fellowship.hidden = False
    if st.fellowship.corruption != ta['corruption']:
        return ('skip', [], 'hunt:card-mithril-corr(%s)' % cardname[:8])
    if sorted(c for c in cb if c != victim) != sorted(ca):
        return ('skip', [], 'hunt:card-mithril-comp(%s)' % cardname[:8])
    if (not is_mordor) and st.fellowship.hidden != ta['hidden']:
        return ('skip', [], 'hunt:card-mithril-hid(%s)' % cardname[:6])
    return ('drove', [], None)


def _drive_multi_tile_hunt(st, batch, key, cardname, tiles):
    """SEQUENTIAL multi-tile hunt resolution (rb p.41 L3726-3743: "if the Free
    Peoples player is forced to draw multiple Hunt tiles ... each tile effect is
    resolved completely before applying the following tile effect").  A reveal
    in a Shadow Stronghold forces an EXTRA tile.  The FIRST tile is the CARD's
    own tile - for Isildur's Bane it is taken ENTIRELY as Corruption (card text:
    the FP "must take all Hunt damage as Corruption", no casualty); the other
    tile-hunt cards resolve the first tile as a normal hunt.  The remaining
    tiles are the stronghold/event reveals, resolved as NORMAL hunts (the drawn
    Companion absorbs by its Level, the excess becomes Corruption; rb p.42).
    Skip-on-mismatch (the general per-card multi-tile logic is unproven, so it
    only claims a MATCH on an exact reproduction, never a resync divergence)."""
    import game_logic as gl
    tb = batch['truth_before']['fel']
    ta = batch['truth']['fel']
    is_mordor = tb['mordor']
    comps_before = [c for c in tb['comps'] if c != 'Gollum']
    comps_after = [c for c in ta['comps'] if c != 'Gollum']
    left = [c for c in comps_before if c not in comps_after]
    if len(left) > 1:
        return ('skip', [], 'hunt:card-multi-tile-comp(%s)' % cardname[:8])
    victim = left[0] if left else None
    isildur = (key == "isildur's bane")
    victim_used = False
    reveal_any = False
    for i, t in enumerate(tiles):
        tv = str(t['value']).strip().lower()
        if tv == 'eye':
            dmg, rev = 0, False
        else:
            m = re.match(r'^(\d+)(r?)$', tv)
            if not m:
                return ('skip', [], 'hunt:card-multi-tile-sp(%s)' % tv[:6])
            dmg, rev = int(m.group(1)), bool(m.group(2))
        reveal_any = reveal_any or rev
        if i == 0 and isildur:
            v = None                          # Isildur: ALL damage as Corruption
        else:
            v = victim if (not victim_used and dmg > 0) else None
            if v:
                victim_used = True
        try:
            gl._apply_fellowship_damage(st, dmg, kill_name=v)
        except Exception as e:
            return ('error', [{'why': 'multi-tile-exc', 'err': str(e)[:60]}],
                    'hunt:card-error')
    if reveal_any:
        st.fellowship.hidden = False
    if st.fellowship.corruption != ta['corruption']:
        return ('skip', [], 'hunt:card-multi-tile-corr(%s)' % cardname[:8])
    expected = sorted(c for c in comps_before if c != victim)
    if expected != sorted(comps_after):
        return ('skip', [], 'hunt:card-multi-tile-comp(%s)' % cardname[:8])
    if (not is_mordor) and st.fellowship.hidden != ta['hidden']:
        return ('skip', [], 'hunt:card-multi-tile-hid(%s)' % cardname[:6])
    return ('drove', [], None)


def _drive_event_card_hunt(st, batch):
    """Drive an EVENT-CARD hunt-effect play (Shadow OR FP) through the engine's
    own hunt-damage resolver under FORCED RNG (the drawn tile + the randomly
    drawn Companion casualty, both printed in the log) and compare fel
    corruption + companions + hidden vs client truth-after.  These batches are
    event-card PLAYS (rb Ch.V), not Fellowship MOVES, so move_fellowship is the
    wrong resolver.  Scoped to the tile-hunt-damage cards (_TILE_HUNT_DAMAGE_
    CARDS); every other played card returns a narrower per-card skip, kept
    honestly UN-covered until its own mechanic driver lands."""
    import game_logic as gl
    cards = _played_event_cards(batch)
    if not cards:
        return ('skip', [], 'hunt:card-none')
    tb = batch['truth_before']['fel']
    ta = batch['truth']['fel']
    is_mordor = tb['mordor']

    def _is_hunt_key(k):
        return (k in _TILE_HUNT_DAMAGE_CARDS or k in _DICE_HUNT_CARDS
                or k in _SEPARATION_CARDS or k in _POOL_HUNT_CARDS
                or ('nazgul' in k and ('search' in k or 'strike' in k)))
    # The PRIMARY hunt card = the first played card matching a known Hunt
    # mechanic.  Co-batched NON-hunt cards (a Shadow attack event like Durin's
    # Bane / Devilry of Orthanc, a card-discard event, a taken-back muster) are
    # DEDUPED - they never touch the Hunt, so they must not force an interpose
    # skip.
    primary = next((c for c in cards if _is_hunt_key(_card_key(c))), None)
    if primary is None:
        return ('skip', [], 'hunt:card(%s)' % cards[0][:16])
    cardname = primary
    key = _card_key(primary)
    # A genuine COMPOUND hunt = a Hunt-PROTECTION card co-played (Axe / Horn /
    # Mithril redraw / Wizard's Staff) OR a SECOND distinct hunt card; the
    # single-mechanic drivers do not model those, so keep the interpose skip.
    mitig = [c for c in cards if _card_key(c) in _MITIGATION_CARDS]
    other_hunt = [c for c in cards if _is_hunt_key(_card_key(c))
                  and _card_key(c) != key]
    _board_plays = sum(
        1 for r in batch['records']
        if r.get('op') == 'move' and '/' in str(r.get('obj', ''))
        and 'board' in str(r.get('to', '')).lower()
        and 'board' not in str(r.get('from', '')).lower())
    if mitig or other_hunt:
        # Mithril Coat and Sting REDRAW is the one mitigation whose net effect
        # is deterministic from the log: it sets the drawn tile aside and draws
        # a SECOND tile which is applied (alm p.11 "discard this card to set
        # aside the drawn tile and draw a second tile to replace it ... Apply
        # the results of the second tile").  When it is the sole mitigation on a
        # tile-hunt-damage card AND ends discarded (used), resolve the SECOND
        # (redrawn) tile as a normal hunt.
        if (len(mitig) == 1 and not other_hunt
                and _card_key(mitig[0]) == 'mithril coat and sting'
                and key in _TILE_HUNT_DAMAGE_CARDS
                and _card_discarded(batch, 'mithril coat and sting')):
            return _drive_mithril_redraw(st, batch, key, cardname)
        return ('skip', [], 'hunt:card-interpose(%s)' % cardname[:12])
    # SKIP-ON-MISMATCH dispatchers (dice / separation / pool / nazgul) each
    # reconstruct their own inputs from the log and skip (never emit a resync
    # divergence) on any residual, so routing the primary while ignoring a
    # deduped non-hunt co-card or a card's own table-staging is SAFE: a misroute
    # is an honest re-skip, never a false divergence or a regression.
    if key in _DICE_HUNT_CARDS:
        return _drive_dice_card_hunt(st, batch, key, cardname)
    if key in _SEPARATION_CARDS:
        return _drive_breaking(st, batch, cardname)
    if key in _POOL_HUNT_CARDS:
        return _drive_challenge(st, batch, cardname)
    if 'nazgul' in key and ('search' in key or 'strike' in key):
        return _drive_nazgul_reveal_chain(st, batch, key, cardname)
    # The tile-hunt-damage path REPORTS resync divergences on mismatch, so keep
    # it gated on a CLEAN single-card hunt with no table leg: a concession-
    # truncated Orc Patrol (GAME09 b109: the FP conceded with the marker frozen
    # mid-defeat, +N corruption never walked) must stay an HONEST skip, never a
    # false divergence.
    if _board_plays:
        return ('skip', [], 'hunt:card-interpose(%s)' % cardname[:12])
    if key not in _TILE_HUNT_DAMAGE_CARDS:
        # Candles/Athelas (dice), Breaking (separate), Nazgul (reveal chain),
        # FP heal, or an unmodeled card - honestly named, still uncovered.
        return ('skip', [], 'hunt:card(%s)' % cardname[:16])
    # Exactly one Hunt tile: a reveal in a Shadow Stronghold draws an EXTRA
    # tile (rb p.41 "Hunt Effects When ... Revealing"); that multi-tile chain
    # is beyond the single-tile model.
    tiles = [r for r in batch['rng'] if r['kind'] == 'draw_tile']
    if len(tiles) != 1:
        if len(tiles) > 1:
            return _drive_multi_tile_hunt(st, batch, key, cardname, tiles)
        return ('skip', [], 'hunt:card-no-tile(%s)' % cardname[:12])
    tile = tiles[0]
    tname = str(tile['value']).strip().lower()
    # A card-drawn Eye has value 0 and does NOT reveal (rb p.41, quoted above);
    # a numbered tile's value is the Hunt damage, with an 'r' suffix = Reveal.
    if tname == 'eye':
        damage, reveal = 0, False
    else:
        m = re.match(r'^(\d+)(r?)$', tname)
        if not m:
            return ('skip', [], 'hunt:card-special-tile(%s)' % tname[:8])
        damage, reveal = int(m.group(1)), bool(m.group(2))
    # the Companion casualty is the ONE companion that left the fellowship
    # this batch (the Shadow's logged random draw, or the FP's chosen guide
    # sacrifice); a tile-hunt takes at most one casualty (rb p.41 "losing one
    # Companion"), so >1 leaving is a different mechanic (Breaking separation).
    # Gollum-normalise (as the nazgul/challenge/breaking drivers do): VASSAL
    # lists a 'Gollum' counter once he is the sole guide (all real Companions
    # gone), but the engine never tracks Gollum in companions, so comparing the
    # last real Companion leaving to a Gollum guide (Foul Thing -> [Gollum])
    # must strip Gollum from both sides.
    comps_before = [c for c in tb['comps'] if c != 'Gollum']
    comps_after = [c for c in ta['comps'] if c != 'Gollum']
    left = [c for c in comps_before if c not in comps_after]
    if len(left) > 1:
        return ('skip', [], 'hunt:card-multi-companion(%s)' % cardname[:12])
    # A random Companion was drawn (draw_companion) but NONE left the
    # fellowship: a drawn HOBBIT absorbs the damage by his Level and SEPARATES
    # instead of dying (Foul Thing rule), and the mirror keeps him tracked -
    # the single-casualty removal model can't reconcile it, so skip.
    _drew_comp = any(r.get('kind') == 'draw_companion' for r in batch['rng'])
    if _drew_comp and not left:
        return ('skip', [], 'hunt:card-companion-stays(%s)' % cardname[:8])
    # W3-031 (2026-09-02): the FP hunt MITIGATIONS the engine itself models
    # on the card-hunt path - event_effects._effect_orc_patrol (Gollum reveal
    # gate) / _effect_foul_thing + _foul_confront (Axe-Horn interpose, guide
    # -1) and the SHARED settle bodies game_logic._offer_hunt_reduction /
    # _gollum_reveal_available / _settle_hunt_damage reached by the play_event
    # tail (ai.py) exactly as by the die-hunt tail - were ABSENT here: the
    # printed tile value was applied raw, so every logged mitigation became a
    # 'card-hunt-damage' / 'card-hunt-reveal' resync divergence (engine =
    # truth + 1) and gate5 quarantined the whole game (6 of 8 in the 300-game
    # probe).  Model them from the log the way the die driver does:
    #  (a) Gollum guide: a standard numbered Reveal tile does NOT reveal
    #      (rb 'Gollum as Guide': 'Standard numbered Hunt tiles with a Reveal
    #      icon do not reveal the Fellowship'; engine 1.115).
    #  (b) Gollum guide, not revealed by the tile: the FP may reveal to
    #      reduce the damage by one - the batch's own Fellowship flip IS that
    #      gesture (a Shadow card batch holds no FP hide).  Outside Mordor
    #      the flip must start from hidden; on the Mordor track the mirror's
    #      hidden flag is a representation gap (see the fel.hidden compare
    #      below), so the flip alone is the gesture.
    #  (c) Axe and Bow / Horn of Gondor DISCARDED this batch (a table-card
    #      kill, not a play - _played_event_cards never sees it) = -1, one
    #      card per tile, on Orc Patrol / Foul Thing only: Isildur's Bane
    #      'must take all Hunt damage as Corruption' (card_list.csv:55; USER
    #      1.217 / CL-287 - the engine offers no reduction there), so a
    #      logged Axe discard on it stays an HONEST divergence (humans vs
    #      the ruling), never absorbed.
    _guide = str(tb.get('guide') or batch.get('_guide') or '').strip().lower()
    _gollum = (_guide == 'gollum')
    _fel_flip = any(
        r.get('op') == 'flip'
        and 'fellowship' in str(r.get('obj', '')).lower()
        for r in batch['records'])
    if _gollum and reveal:
        reveal = False                                          # (a)
    if (_gollum and not reveal and _fel_flip and damage > 0
            and (is_mordor or tb['hidden'])):
        damage -= 1                                             # (b)
        reveal = True
    if key != "isildur's bane" and damage > 0:
        for _mk in ('axe and bow', 'horn of gondor'):
            if _card_discarded(batch, _mk):
                damage -= 1                                     # (c)
                break
    victim = left[0] if (left and damage > 0) else None
    try:
        gl._apply_fellowship_damage(st, damage, kill_name=victim)
    except Exception as e:
        return ('error', [{'why': 'card-hunt-exception', 'err': str(e)[:80]}],
                'hunt:card-error')
    if reveal:
        st.fellowship.hidden = False
    diffs = []
    eng_corr = st.fellowship.corruption
    if eng_corr != ta['corruption']:
        diffs.append({'field': 'fel.corruption', 'eng': eng_corr,
                      'truth': ta['corruption'], 'class': 'card-hunt-damage',
                      'resync': True})
    # Companion check is LOG-BASED (reconstruction-independent - _state_from_
    # snapshot does not rebuild Gollum): removing the single logged casualty
    # from the before-set must reproduce the after-set.
    expected_after = sorted(c for c in comps_before if c != victim)
    if expected_after != sorted(comps_after):
        diffs.append({'field': 'fel.comps', 'eng': expected_after,
                      'truth': sorted(comps_after),
                      'class': 'card-hunt-comps', 'resync': True})
    # fel.hidden is a representation gap on the Mordor track (like _drive_hunt),
    # so only compare a reveal outside Mordor.
    if not is_mordor and st.fellowship.hidden != ta['hidden']:
        diffs.append({'field': 'fel.hidden', 'eng': st.fellowship.hidden,
                      'truth': ta['hidden'], 'class': 'card-hunt-reveal',
                      'resync': True})
    return ('drove', diffs, None)


def _drive_hunt(st, batch):
    """Drive the engine's own hunt resolver (game_logic.move_fellowship) from
    the rendered mirror under FORCED roll + drawn tile.  The SIGNAL is
    (corruption + pending damage, hidden) vs rendered truth.

    Only a GENUINE Free-Peoples fellowship MOVE is faithfully drivable: it
    stages an FP die, advances the FSP pawn, and draws an ordinary fixed-
    damage tile (or none, on a 0-success move).  COMPANION-CASUALTY hunts ARE
    now driven (W-F): the engine still computes the total tile damage, and we
    reconcile it against the human's corruption walk PLUS the Level of every
    Companion the FP sacrificed to the Hunt (each absorbs its Level) - the
    directive's corruption-walk-vs-companion-elimination split, wired as a
    damage INVARIANT rather than skipped.  Still excluded (wrong resolver or
    genuinely-missing information, each named):
      * Shadow-side hunt-tile card plays (Nazgul Strike/Search) - a Shadow
        die moves & reveals the pawn via a card, not an FP move.
      * FP corruption-heal/effect cards with no pawn move (Athelas).
      * Eye / Shelob / dynamic-damage special tiles (the damage is the hunt-
        box success count at that instant, not a fixed tile number - needs the
        full hunt-box-modifier forcing machinery) and Mordor steps.
    fel.progress is NOT compared: the VASSAL Fellowship-track marker resets
    to 0 on every reveal while the engine's progress counter does not, a
    representation gap (like the Mordor-region alias), not an engine bug."""
    import random
    import re as _re
    import game_state as gs
    import game_logic as gl
    tb = batch['truth_before']['fel']
    ta = batch['truth']['fel']
    is_mordor = tb['mordor']
    # EVENT-CARD hunt-effect play (Shadow OR FP): an event card moved to a
    # Current Card slot resolves via the card's text (rb Ch.V), NOT
    # move_fellowship.  Route it to the event-card hunt driver - but ONLY when
    # the batch is not a genuine Fellowship PAWN advance (a real move never
    # plays a card into that slot), so this is purely additive over the batches
    # that otherwise skip and cannot regress a currently-driven move.
    _step_adv = is_mordor and (ta['step'] > tb['step'])
    _is_fp_move = (batch.get('die_side') == 'FREE_PEOPLES'
                   and (getattr(st, '_fellowship_move_die_pending', None) is not None
                        or _step_adv or (not is_mordor and _has_fsp_move(batch))))
    if _played_event_card(batch) and not _is_fp_move:
        return _drive_event_card_hunt(st, batch)
    if batch.get('die_side') != 'FREE_PEOPLES':
        return ('skip', [],
                'hunt:mordor-shadow' if is_mordor else 'hunt:shadow-tile-card')
    # non-Mordor requires an FSP pawn-move; a Mordor STEP is signalled by a
    # step advance / corruption move instead (the move_fellowship Mordor
    # branch draws a tile directly, no success roll)
    if not is_mordor and not _has_fsp_move(batch):
        return ('skip', [], 'hunt:no-pawn-move')
    if is_mordor and (ta['step'] < tb['step'] or tb['step'] >= 5):
        return ('skip', [], 'hunt:mordor-step')
    roll = _first_roll(batch)
    tile = next((r for r in batch['rng'] if r['kind'] == 'draw_tile'), None)
    tname = str(tile['value']).strip().lower() if tile else None
    # The Eye tile's damage is DYNAMIC (= the hunt-box success count that
    # instant), but for an ordinary (non-Mordor) FP move the successes are
    # exactly the forced FP hunt roll, so the engine CAN compute it (directive:
    # Eye dynamic tiles = engine-computed from the hunt box).  Let it through;
    # the damage invariant then reconciles vs the corruption walk (a residual
    # mismatch = the Shadow's pre-allocated hunt-box Eyes, which the mirror
    # does not track - named missing info, driver-limited).  Shelob / Mordor-
    # Eye special tiles stay skipped (they need the full hunt-box machinery).
    fo = _tile_obj(gs, tname) if tname else None
    # A tile whose name the engine does not model at all -> genuinely
    # un-drivable (rare, unknown special); keep it skipped as a named tile.
    if tname is not None and fo is None:
        return ('skip', [], 'hunt:special(%s)' % tname[:6])
    # An Eye tile on the MORDOR TRACK deals DYNAMIC damage = the number of dice
    # in the Hunt Box (rb p.42-43, data/_rulebook.txt L3905-3910: "the Hunt
    # damage is equal to the number of dice in the Hunt Box (including Free
    # Peoples dice previously used for moving the Fellowship during the same
    # turn)").  The count is reconstructed by _prescan_mordor_box (line-ordered
    # box simulator to the step-move anchor); seed the engine's Hunt Box with it
    # and _apply_mordor_hunt_tile computes damage = len(hunt_box).  Without a
    # count (skip_probe / non-census caller), it stays skipped.
    _eye_mordor = (fo is not None and getattr(fo, 'is_eye', False)
                   and is_mordor)
    _mbox = batch.get('_mordor_box')
    if _eye_mordor and _mbox is None:
        return ('skip', [], 'hunt:special(eye)')
    if is_mordor and tname is None:
        return ('skip', [], 'hunt:mordor-no-tile')
    if roll is None and tile is None:
        return ('skip', [], 'hunt:no-rng')
    st.hunt_pool = [fo] if fo is not None else []
    physical_move = getattr(st, '_fellowship_move_die_pending', None) is not None
    if is_mordor:
        # seed the Hunt Box for a Mordor Eye (len = the dynamic damage);
        # any other Mordor tile is drawn directly with no roll.
        if not physical_move:
            st.hunt_box = ([gs.ActionDie(gs.SH_FACES, gs.Side.SHADOW)
                            for _ in range(int(_mbox or 0))]
                           if _eye_mordor else [])
        elif _eye_mordor and len(st.hunt_box) != int(_mbox):
            return ('skip', [], 'hunt:physical-mordor-box-mismatch')
        # A die-icon special tile (Shelob's Lair) rolls a d6 for its Hunt
        # damage (rb p.39, data/_rulebook.txt L3599-3603: "A Die icon means
        # that the Hunt damage is equal to the subsequent roll of a die
        # (roll after the tile has been revealed)").  Force that d6 from the
        # batch's logged roll_num so the engine's roll_d6_batch consumes the
        # human's die rather than rolling live.
        if fo is not None and getattr(fo, 'is_dice_roll', False):
            seq = iter(_all_roll_ints(batch))
        else:
            seq = iter(())
    else:
        n_dice = len(roll['ints']) if roll else 0
        # PRESERVE pre-seeded FREE-PEOPLES dice (the caller seeds one
        # per earlier fellowship move this turn - each is +1 to the
        # hunt roll, p.40; without them the engine scored GAME02 T2's
        # third-move 4+2 hit as a miss and skipped the resolution)
        _fp_prev = [d for d in (getattr(st, 'hunt_box', None) or [])
                    if d.side is gs.Side.FREE_PEOPLES]
        if not physical_move:
            st.hunt_box = [gs.ActionDie(gs.SH_FACES, gs.Side.SHADOW)
                           for _ in range(n_dice)] + _fp_prev
        elif n_dice != min(5, sum(d.side is gs.Side.SHADOW for d in st.hunt_box)):
            return ('skip', [], 'hunt:physical-shadow-roll-count-mismatch')
        # W2-035 (COORD #2 2026-09-03, USER order 22:30): feed EVERY
        # logged roll in line order - the first chunk is the hunt roll
        # (n_dice above), the later chunks are the Shadow's RE-ROLLS
        # (Nazgul / Shadow stronghold / Fellowship in a Shadow region,
        # rb p.40).  Feeding only the first chunk left the engine's
        # re-roll on the LIVE random module -> hunt damage / casualty
        # rows that changed with PYTHONHASHSEED (2023-12-25 L340: rolls
        # '2 3 1' then '5'; 2023-11-26 L998: '3 5' then '3 6' = one
        # success, tile 3, corruption 8 -> 11).
        seq = iter(_all_roll_ints(batch))
    orig_ri, orig_ch = random.randint, random.choice
    random.randint = lambda a, b2, _i=seq, _f=orig_ri: next(_i, _f(a, b2))
    random.choice = lambda s2: s2[-1]
    try:
        gl.move_fellowship(st, skip_auto_sacrifice=True)
    except Exception as e:
        return ('error', [{'why': 'hunt-exception', 'err': str(e)[:80]}],
                'hunt:error')
    finally:
        random.randint, random.choice = orig_ri, orig_ch
    pend = int(getattr(st, '_pending_hunt_damage', 0) or 0)
    eng_corr = st.fellowship.corruption + pend
    tr_corr = ta['corruption']
    diffs = []
    # HUNT-DAMAGE invariant (directive corruption-walk-vs-companion split): the
    # engine's TOTAL computed hunt damage = the corruption the FP actually took
    # PLUS the Level of every Companion it sacrificed to the Hunt (each absorbs
    # its Level, remainder -> corruption).  So eng_damage == corr_gained +
    # sum(levels) exactly, unless the FP OVER-absorbed (took a Companion whose
    # Level exceeds the residual, corruption then 0 and eng_damage <= levels).
    # A residual mismatch is DRIVER-LIMITED (never a gate bug): the rendered
    # text does not carry the FP hunt-box dice modifier (+1/die lowers the
    # success threshold) nor FP damage-reduction cards (Athelas, Elven Cloak).
    eng_damage = eng_corr - tb['corruption']
    corr_gained = tr_corr - tb['corruption']
    elim = [c for c in tb.get('comps', []) if c not in ta.get('comps', [])]
    sumlev = sum(_COMP_LEVEL.get(c, 1) for c in elim)
    if sumlev == 0:
        consistent = (eng_damage == corr_gained)
    else:
        consistent = (eng_damage == corr_gained + sumlev) or \
                     (corr_gained == 0 and 0 < eng_damage <= sumlev)
    # TERMINAL hunt (rb p.44 L3852-3854: at 12 Corruption the Ring-bearers are
    # immediately lost): when the engine's damage brings Corruption to >= 12 the
    # game ends on the spot, so the client's Corruption walk is truncated /
    # split across the final cross-player gestures and the batch-snapshot delta
    # under-reads it.  The engine's total (which reaches defeat) is correct -
    # accept it (Mordor-Eye terminal residuals TheLastNumeno T8 / ppoma T6).
    if not consistent and tb['corruption'] + eng_damage >= 12:
        consistent = True
    if not consistent:
        cls = ('hunt-damage(companion-split)' if sumlev else
               'hunt-damage(fp-mitigation/hunt-box-mod)')
        diffs.append({'field': 'fel.corruption', 'eng': eng_corr,
                      'truth': tr_corr, 'class': cls,
                      'eng_damage': eng_damage, 'corr_gained': corr_gained,
                      'companion_levels': sumlev, 'resync': False})
    # fel.hidden is a REPRESENTATION GAP on the Mordor track: there is no
    # hidden/revealed fellowship-MARKER there (the mirror sets hidden=True on
    # Mordor entry and never re-derives it from a step tile, while the engine's
    # Mordor branch tracks it independently), so the flag is not comparable in
    # Mordor - exactly like fel.progress.  Only an ordinary-hunt reveal is a
    # real signal.
    # With a staged reveal the Guide/casualty decisions still precede the
    # marker flip (and choosing Gollum may suppress a numeric icon). This
    # driver has not applied those choices, so its hidden flag is not yet
    # comparable with the batch's final physical snapshot.
    if not is_mordor and not getattr(st, '_hunt_reveal_pending', 0):
        eng_hidden = st.fellowship.hidden
        if eng_hidden != ta['hidden']:
            diffs.append({'field': 'fel.hidden', 'eng': eng_hidden,
                          'truth': ta['hidden'], 'class': 'hunt-reveal',
                          'resync': False})
    return ('drove', diffs, None)


def _die_count(obj):
    """Count Action dice named in a move/kill token or group, honouring the
    '[N]' multiplier (e.g. '[2]Shadow Action Die (E)' = 2)."""
    s = str(obj)
    if s.strip().lower().startswith('group'):
        toks = s[s.find('(') + 1:s.rfind(')')].split(',')
    else:
        toks = [s]
    n = 0
    for t in toks:
        if 'action die' in t.lower():
            m = re.match(r'^\s*\[(\d+)\]', t)
            n += int(m.group(1)) if m else 1
    return n


def _prescan_mordor_box(g):
    """Stash on each Mordor-track Eye hunt batch the Hunt Box die count at the
    moment the Eye is drawn (batch['_mordor_box']).  rb p.42-43 (data/_rulebook
    .txt L3905-3910): a Mordor Eye's Hunt damage = the number of dice in the
    Hunt Box (including FP dice PREVIOUSLY used for moving the Fellowship this
    turn).  Reconstructed by simulating the VASSAL 'The Hunt For The Ring' box
    over the TURN's net records IN LINE ORDER, up to the ANCHOR = this batch's
    fellowship-marker move whose Mordor destination step == the after-step (the
    current move's own die is placed AFTER that marker line, so line<anchor
    excludes it, matching the engine which never appends the current die).
    Two allocation rendering eras both count (alloc_eye 'allocates an eye'; and
    'Shadow Action Die -> The Hunt' SA-die moves), plus rolled eyes 'adds N
    eyes'; mid-turn recovery/kills in the box subtract.  Verified 29/29 box-
    correct (25/29 full empty-diff; the 4 are same-batch card/terminal snapshot
    limits, not box gaps) - Mordor-eye forensic 2026-07-25."""
    # index records by turn once
    by_turn = {}
    for b in g['batches']:
        by_turn.setdefault(b['turn'], []).extend(b['records'])
    for b in g['batches']:
        ta = b['truth']['fel']
        tb = b['truth_before']['fel']
        if not (tb.get('mordor') and ta.get('step', 0) >= tb.get('step', 0)):
            continue
        tile = next((r for r in b['rng'] if r.get('kind') == 'draw_tile'), None)
        if not tile or str(tile.get('value', '')).strip().lower() != 'eye':
            continue
        step = ta.get('step', 0)
        anchor = None
        _fallback = None
        for r in b['records']:
            if r.get('op') == 'move' and \
                    str(r.get('obj', '')).strip().lower() == 'the fellowship' \
                    and 'mordor' in str(r.get('to', '')).lower():
                _fallback = _fallback or r.get('line')
                if re.search(r'mordor\s+%d\b' % step,
                             str(r.get('to', '')).lower()):
                    anchor = r.get('line')
                    break
        anchor = anchor if anchor is not None else _fallback
        if anchor is None:
            continue
        box = 0
        for r in by_turn.get(b['turn'], []):
            if int(r.get('line', 0) or 0) >= anchor:
                continue
            op = r.get('op')
            obj = str(r.get('obj', ''))
            to = str(r.get('to', '')).lower()
            frm = str(r.get('from', '')).lower()
            if op == 'alloc_eye':
                box += 1
            elif op == 'add_eye':
                box += int(r.get('n', 0) or 0)
            elif op == 'move' and 'action die' in obj.lower():
                if 'hunt for the ring' in to:
                    box += _die_count(obj)
                elif 'hunt for the ring' in frm:
                    box -= _die_count(obj)
            elif op == 'kill' and 'hunt for the ring' in \
                    str(r.get('zone', '')).lower():
                box -= sum(_die_count(p) for p in (r.get('pieces') or []))
        b['_mordor_box'] = max(0, box)


def _prescan_guide(g):
    """Stash on each batch the Guide of the Fellowship at its START
    (batch['_guide']).  The guide is set by a "moves <Companion> ... to Guide
    Of The Fellowship" gesture and starts as Gandalf the Grey; it governs two
    Hunt thresholds (Athelas heals on 3+ with Strider guiding else 5+; Candles
    of Corpses corrupts on 6+ with Gollum guiding else 4+).  Fully determined
    by the log's explicit guide-box moves (verified: 14/14 Athelas drives
    reproduce the logged corruption)."""
    guide = 'Gandalf the Grey'
    for b in g['batches']:
        b['_guide'] = guide
        for r in b['records']:
            if r.get('op') == 'move' and 'guide of the fellowship' in \
                    str(r.get('to', '')).lower():
                nm = str(r.get('obj', '')).strip()
                if nm:
                    guide = nm


def rendered_driven_census(game):
    """Drive the engine directly from the rendered mirror (no raw pipeline)
    and diff engine-after vs rendered truth-after per batch.

    Coverage accounting (honest anti-gaming guard, USER 2026-07-22):
      * BOOKKEEPING batches (muster/move/political/fellowship/other) carry no
        engine DISCRETION - the mirror IS the client-resolved truth and the
        engine reproduces them by construction - so they are FAITHFULLY
        covered.
      * DRIVEN batches (hunt/combat) exercise a real engine resolver.
      * SKIPPED hunt/combat batches are the UN-covered surface.
    faithful-coverage = (bookkeeping + driven) / total; a low resync at low
    coverage is not a pass (the skip surface hides untested decisions)."""
    import collections
    rendered_path = resolve_rendered(game)
    g = rr.parse_game(rendered_path)
    total = len(g['batches'])
    _prescan_guide(g)
    _prescan_mordor_box(g)
    status_ct = collections.Counter()
    skip_reasons = collections.Counter()
    diverged = []
    policy_notes = []
    matched = 0
    for b in g['batches']:
        kind = b['kind']
        if kind not in ('hunt', 'combat', 'combat?'):
            status_ct['bookkeeping'] += 1
            continue
        try:
            st = _state_from_snapshot(b['truth_before'])
        except Exception as e:
            status_ct['recon-error'] += 1
            skip_reasons['recon-error'] += 1
            continue
        if kind == 'hunt':
            status, diffs, reason = _drive_hunt(st, b)
        else:
            status, diffs, reason = _drive_combat(st, b)
        status_ct[status] += 1
        if reason:
            skip_reasons[reason] += 1
        if status != 'drove':
            continue
        if not diffs:
            matched += 1
        rediffs = [d for d in diffs if d.get('resync')]
        pols = [d for d in diffs if not d.get('resync') and d.get('class')]
        rec = {'batch': b['idx'], 'turn': b['turn'], 'kind': kind,
               'diffs': [_fmt_diff(d) for d in rediffs],
               'carded': any(d.get('carded') for d in diffs)}
        if rediffs:
            diverged.append(rec)
        if pols:
            policy_notes.append({'batch': b['idx'], 'turn': b['turn'],
                                 'kind': kind,
                                 'diffs': [_fmt_diff(d) for d in pols]})
    driven_n = status_ct.get('drove', 0)
    book_n = status_ct.get('bookkeeping', 0)
    skip_n = status_ct.get('skip', 0)
    err_n = status_ct.get('error', 0) + status_ct.get('recon-error', 0)
    faithful = book_n + driven_n
    drivable = driven_n + skip_n + status_ct.get('error', 0)
    out = {'game': raw_stem(rendered_path),
           'rendered_log': os.path.basename(rendered_path),
           'total_batches': total,
           'bookkeeping_batches': book_n,
           'engine_driven_batches': driven_n,
           'driven_matched_batches': matched,
           'skipped_batches': skip_n,
           'error_batches': err_n,
           'driven_diverged_batches': len(diverged),
           'policy_note_batches': len(policy_notes),
           # CONCLUSIVE = bookkeeping (no resolver, faithful by construction)
           # + driven-and-matched + reliable gate-divergence (a real finding).
           # driven-but-complex-divergent batches are driver-inconclusive.
           'conclusive_pct': round(100 * (book_n + matched + len(diverged))
                                   / total, 1) if total else 0,
           'coverage_pct': round(100 * faithful / total, 1) if total else 0,
           'driven_coverage_pct': round(100 * driven_n / drivable, 1)
           if drivable else 100.0,
           'resync_rate_driven': round(len(diverged) / total, 4)
           if total else 0,
           'gate_pass': (len(diverged) / total < 0.01 if total else True)
           and (100 * faithful / total >= 90.0 if total else True),
           'status_counts': dict(status_ct),
           'skip_reasons': dict(skip_reasons),
           'divergences': diverged,
           'policy_notes': policy_notes}
    return out


def _fmt_diff(d):
    """closed_loop _snapshot_diff-style string for a driven diff dict."""
    return '%s eng=%s truth=%s [%s]' % (
        d['field'], d.get('eng'), d.get('truth'), d.get('class'))


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def resolve_rendered(game):
    """Accept a full path, a GAMEnn_*.rendered.txt basename, or a bare
    stem/substring, and return the rendered-log path."""
    if os.path.isabs(game) and os.path.exists(game):
        return game
    cand = os.path.join(RLOG_DIR, game)
    if os.path.exists(cand):
        return cand
    for f in rendered_files():
        base = os.path.basename(f)
        if game in base or _norm_stem(raw_stem(f) or '') == _norm_stem(game):
            return f
    return game


def run_game(game, do_hunt=True):
    game = resolve_rendered(game)
    rec = reconcile(game)
    if rec is None:
        print('SKIP (no parity json):', game)
        return None
    if do_hunt:
        try:
            rec['hunt_confirm'] = hunt_confirm(game)
        except Exception as e:
            rec['hunt_confirm'] = [{'why': 'pass-error', 'err': str(e)[:120]}]
    os.makedirs(OUT_DIR, exist_ok=True)
    op = os.path.join(OUT_DIR, rec['game'] + '.json')
    with open(op, 'w', encoding='utf-8') as f:
        json.dump(rec, f, indent=1)
    print("%-42s batches=%3d raw_div=%3d -> PERSIST=%3d VANISH=%3d "
          "DISAGREE=%3d  raw=%.1f%% rendered=%.1f%%  huntdiv=%d"
          % (rec['game'][:42], rec['total_batches'],
             rec['raw_diverged_batches'], rec['persist_batches'],
             rec['vanish_batches'], rec['disagree_batches'],
             100 * rec['resync_rate_raw'], 100 * rec['resync_rate_rendered'],
             len(rec.get('hunt_confirm', []))))
    return rec


def run_driven(game):
    """Run + persist the rendered-DRIVEN engine census for one game."""
    rec = rendered_driven_census(game)
    os.makedirs(OUT_DIR, exist_ok=True)
    op = os.path.join(OUT_DIR, (rec['game'] or 'game') + '.driven.json')
    with open(op, 'w', encoding='utf-8') as f:
        json.dump(rec, f, indent=1)
    print("%-38s bat=%3d cov=%5.1f%% drv=%3d(dcov=%5.1f%%) skip=%3d "
          "resync=%5.2f%% div=%2d pol=%2d %s"
          % (rec['game'][:38], rec['total_batches'], rec['coverage_pct'],
             rec['engine_driven_batches'], rec['driven_coverage_pct'],
             rec['skipped_batches'], 100 * rec['resync_rate_driven'],
             rec['driven_diverged_batches'], rec['policy_note_batches'],
             'PASS' if rec['gate_pass'] else 'FAIL'))
    return rec


# CARD_NAME_ALIASES: manual overrides for app card titles the fuzzy
# ai._lookup_card_id cannot resolve: unresolved titles land in
# _LOOKUP_FAILS (diagnostic set, inspect interactively or dump in a
# census tail) and get mapped here; extend when extraction reports
# new misses.
CARD_NAME_ALIASES = {}
_LOOKUP_FAILS = set()


def _lookup_cid(card_lookup, name):
    import ai
    nm = str(name or '')
    cid = ai._lookup_card_id(card_lookup, nm)
    if cid is None:
        cid = CARD_NAME_ALIASES.get(ai._normalize_card_name(nm))
    # the app logs full names with '/combat-title' suffixes, commas and
    # trailing spaces ('Grimbeorn the Old, Son of Beorn',
    # 'Olog-Hai/We Come to Kill ') -- retry on progressively shorter
    # prefixes
    if cid is None and '/' in nm:
        return _lookup_cid(card_lookup, nm.split('/')[0].strip())
    if cid is None and ',' in nm:
        return _lookup_cid(card_lookup, nm.split(',')[0].strip())
    if cid is None and nm != nm.strip():
        return _lookup_cid(card_lookup, nm.strip())
    if cid is None and hasattr(card_lookup, 'keys'):
        # prefix match: the parser DB truncates some names the card CSV
        # spells out ('Grimbeorn the Old' vs 'Grimbeorn the Old Son of
        # Beorn')
        want = ai._normalize_card_name(nm)
        if want and len(want) >= 8:
            for k in card_lookup.keys():
                kn = ai._normalize_card_name(str(k))
                if kn.startswith(want) or want.startswith(kn):
                    return card_lookup[k]
    if cid is None and nm.strip():
        # only a FINAL miss is a lookup failure (fallbacks above
        # return directly on success)
        _LOOKUP_FAILS.add(nm.strip())
    return cid


if __name__ == '__main__':
    args = sys.argv[1:]
    if args and args[0] == '--driven':
        rec = run_driven(args[1] if len(args) > 1 else 'GAME01')
        print('  total=%d cov=%.1f%% driven=%d skip=%d resync=%.2f%% '
              'gate=%s' % (rec['total_batches'], rec['coverage_pct'],
                           rec['engine_driven_batches'],
                           rec['skipped_batches'],
                           100 * rec['resync_rate_driven'],
                           'PASS' if rec['gate_pass'] else 'FAIL'))
        print('  skip_reasons: %s' % rec['skip_reasons'])
        for d in rec['divergences'][:15]:
            print('   DIV b%-3d T%-2d %-8s %s'
                  % (d['batch'], d['turn'], d['kind'], d['diffs'][:2]))
    elif args and args[0] == '--driven-all':
        recs = []
        for f in rendered_files():
            try:
                recs.append(run_driven(f))
            except Exception as e:
                print('DRIVEN-ERROR', os.path.basename(f), str(e)[:100])
        tb = sum(r['total_batches'] for r in recs)
        cov = sum(r['bookkeeping_batches'] + r['engine_driven_batches']
                  for r in recs)
        drv = sum(r['engine_driven_batches'] for r in recs)
        skp = sum(r['skipped_batches'] for r in recs)
        dv = sum(r['driven_diverged_batches'] for r in recs)
        npass = sum(1 for r in recs if r['gate_pass'])
        print('\n=== DRIVEN AGGREGATE (%d games) ===' % len(recs))
        print('batches=%d faithful-coverage=%.1f%% driven=%d skip=%d '
              'diverged=%d resync=%.2f%%  games<1%%&>=90%%cov: %d/%d'
              % (tb, 100 * cov / tb if tb else 0, drv, skp, dv,
                 100 * dv / tb if tb else 0, npass, len(recs)))
    elif args and args[0] == '--game':
        run_game(args[1])
    elif args and args[0] == '--all':
        recs = []
        for f in rendered_files():
            r = run_game(f)
            if r:
                recs.append(r)
        # aggregate summary
        tb = sum(r['total_batches'] for r in recs)
        pp = sum(r['persist_batches'] for r in recs)
        rd = sum(r['raw_diverged_batches'] for r in recs)
        print('\n=== AGGREGATE (%d games) ===' % len(recs))
        print('batches=%d  raw_diverged=%d (%.1f%%)  PERSIST=%d (%.2f%%)  '
              'VANISH_fields=%d  DISAGREE_fields=%d'
              % (tb, rd, 100 * rd / tb, pp, 100 * pp / tb,
                 sum(r['vanish_fields'] for r in recs),
                 sum(r['disagree_fields'] for r in recs)))
    else:
        print(__doc__)
        print('usage: rendered_census.py --driven <stem|path> | --driven-all | --game <stem|path> | --all')
