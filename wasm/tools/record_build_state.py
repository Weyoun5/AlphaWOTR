"""Phase 3's data and references: the protocol adapter's state building (protocol/wotr_protocol_bot.py build_state).

    venv/bin/python -I wasm/tools/record_build_state.py --alpha ALPHAWOTR_CHECKOUT --out wasm/fixtures
        [--every K] TRANSCRIPT.protocol.jsonl ...

Writes into --out:
  adapter.json        the adapter's name tables (Names: regions, nations, faces, cards, the cards of each deck in
                      the engine's order), CHAR_IN, MINION_KEYS, DECKS, ai._region_nation of every board region,
                      game_logic._vp_settlement_tables, the board's canonical region keys (get_or_create_army)
  prototypes.awg      one graph: a dict of the objects build_state creates, as their constructors make them
                      ('initial' gs.setup_initial_game(BOARD) under random.seed(0), 'army', 'unit', 'leader',
                      'nazgul', 'die_fp', 'die_sh', 'tile', 'special:<key>', 'figure:<char key>',
                      'persistent:<card id>', 'deck_status', 'persistent_state'); one graph, so the enums and the
                      registered global lists are shared nodes as in a state
  build_state.awb     every K-th build_state of the adapter run offline over the transcripts (as
                      harvest_decisions.py runs it): the view (JSON), the memo before (side, its Random's state,
                      the per-turn flags), the state built (graph)

awb: b'AWB5' (AWB1: without the set-up part; AWB2: without the menus; AWB3: without the matching; AWB4: without the
     stages), u32 count, per case: u32 json bytes, json, u32 side, rng (624 u32, u32 index, i32 has_gauss, u64 bits),
     u8 messenger_used, u8 fellowship_acted, u8 in_mordor, graph (awg record): the state,
     u32 searched (1 when the adapter set the decision up and searched this state), then when 1: u32 + the decide
     message's decision (JSON), u32 + the set-up's label, u32 + its phase flag ('' none), graph: the state as set up
     (before the flag is set; the reference for the set-up's port), then two menus (get_legal_actions under the
     flag; project_ai_actions of it): u32 rows, each u32 + its canonical text (canon_action), then the matching of
     every legal row (u32 rows; per row i64 option or -1, u8 has_plan, u8 chain), then the adapter's answer
     (i64 option or -1, u32 + how) and its search (u64 seat seed, u32 search index, u64 search seed, i64 sims or -1),
     then u32 + the stages of LeagueSearcher.choose as JSON: the search's act, the net's raw act, the certain line
     (outright, favoured, certain_line.LAST), ai_units' act, the pick, the search's how (canon_action texts)
Run it with PYTHONHASHSEED=0 in the environment and without -I (which ignores it): the AI server's launcher sets it,
and a Python set's order depends on it.
"""
import argparse
import copy
import json
import os
import pickle
import random
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--every', type=int, default=5)
    ap.add_argument('transcripts', nargs='*')
    a = ap.parse_args()
    alpha, out = os.path.abspath(a.alpha), os.path.abspath(a.out)
    transcripts = [os.path.abspath(t) for t in a.transcripts]
    sys.path.insert(0, HERE)
    import export_fixtures as ef
    from record_searches import rng_bytes
    sys.path[:0] = [os.path.join(alpha, 'protocol'), os.path.join(alpha, 'wotr_bridge'),
                    os.path.join(alpha, 'native_backend')]
    import wotr_protocol_bot as wpb
    wpb.import_engine(os.path.join(alpha, 'src'), True)
    ai, gs, gl = wpb.ai, wpb.gs, wpb.gl
    names = wpb.Names()
    import snapshot
    import board as board_mod

    import cards as cards_mod0
    import combat as combat_mod

    def cards_mod_title(cid):
        return cards_mod0.get_card(cid).card_name

    def combat_kind(cid):
        return combat_mod.combat_effect_kind(getattr(cards_mod0.get_card(cid), 'combat_name', '') or '') or None

    # 1. the tables
    FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW
    fp_vp, sh_vp = gl._vp_settlement_tables()
    tables = dict(
        region_in=names.region_in, nation_in={k: v.name for k, v in names.nation_in.items()},
        face_in={k: v.name for k, v in names.face_in.items()}, card_in=names.card_in,
        cards_of={'%s|%s' % (s.name, d): list(v) for (s, d), v in names.cards_of.items()},
        char_in=wpb.CHAR_IN, minion_keys=list(wpb.MINION_KEYS), decks=list(wpb.DECKS),
        region_nation={k: (ai._region_nation(k).name if ai._region_nation(k) is not None else None)
                       for k in wpb.BOARD._regions},
        vp_fp_settlements=[list(x) for x in fp_vp], vp_sh_settlements=[list(x) for x in sh_vp],
        board_keys={k: getattr(r, 'key', k) for k, r in wpb.BOARD._regions.items()},
        special_tiles=sorted(names.special_tile),
        # for the decision set-ups (SETUPS): the board's strongholds and adjacency, each card's printed title and
        # combat effect kind (combat.combat_effect_kind of its combat name), ai._HIDDEN_CC
        stronghold={k: bool(m.get('is_stronghold')) for k, m in board_mod.regions.items()},
        adjacent={k: sorted(wpb.BOARD.adjacent(k)) for k in wpb.BOARD._regions},
        card_title={cid: cards_mod_title(cid) for cid in names.card_out},
        combat_effect_kind={cid: combat_kind(cid) for cid in names.card_out},
        hidden_cc=ai._HIDDEN_CC,
        fp_region={k: bool(gl.is_fp_region(k)) for k in wpb.BOARD._regions},
        face_by_value={f.value: f.name for f in gs.ActionDieFace},
        nation_value={n.name: n.value for n in gs.Nation})
    with open(os.path.join(out, 'adapter.json'), 'w', encoding='utf-8') as f:
        json.dump(tables, f, sort_keys=True, ensure_ascii=False)

    # 2. the prototypes, one graph
    random.seed(0)
    initial = gs.setup_initial_game(wpb.BOARD)
    protos = {'initial': initial,
              'army': gs.Army(region=wpb.BOARD.region('rivendell')),
              'unit': gs.Unit(gs.Nation.ELVES, gs.UnitKind.REGULAR),
              'leader': gs.Leader(gs.Nation.ELVES), 'nazgul': gs.Nazgul(gs.Nation.SAURON),
              'die_fp': gs.ActionDie(gs.FP_FACES, FP), 'die_sh': gs.ActionDie(gs.SH_FACES, SH),
              'tile': gs.HuntTile(0), 'deck_status': gs.DeckStatus(0, 0, 0, 0),
              'persistent_state': gs.PersistentCardState()}
    for key in names.special_tile:
        protos['special:' + key] = names.tile(key)
    for key in ('gandalf-the-white', 'aragorn') + tuple(wpb.MINION_KEYS):
        protos['figure:' + key] = wpb.character_figure(key, {})
    from persistent_effects import create_persistent_card
    import cards as cards_mod
    for cid in names.card_out:
        card = cards_mod.get_card(cid)
        side = FP if names.card_out[cid].startswith('fp-') else SH
        try:
            pc = create_persistent_card(cid, card.card_name, side)
        except Exception:
            pc = gs.PersistentCard(card_id=cid, card_name=card.card_name, side=side, card_type=None)
        protos['persistent:' + cid] = pc
    ef.write_awg(os.path.join(out, 'prototypes.awg'), [('prototypes', snapshot.encode_graph(ai, protos))])
    print('tables and %d prototypes written' % len(protos), file=sys.stderr)

    # a menu row as a canonical text (the C++ side writes the box's rows the same way, adapter/menu.cpp)
    import enum

    def canon(v):
        if v is None:
            return 'N'
        if v is True:
            return 'T'
        if v is False:
            return 'F'
        if isinstance(v, enum.Enum):
            return 'e' + v.name
        if isinstance(v, int):
            return 'i%d' % v
        if isinstance(v, float):
            return 'f' + repr(v)
        if isinstance(v, str):
            return 's%d:%s' % (len(v.encode('utf-8')), v)
        if isinstance(v, (list, tuple)):
            return '[' + ','.join(canon(x) for x in v) + ']'
        if isinstance(v, dict):
            return '{' + ','.join(sorted(canon(k) + ':' + canon(x) for k, x in v.items())) + '}'
        if isinstance(v, (set, frozenset)):
            return '<' + ','.join(sorted(canon(x) for x in v)) + '>'
        return 'o' + str(getattr(v, 'name', type(v).__name__))

    def canon_action(a):
        return '%d|%s|%s' % (a.die_idx, a.action_type, canon(a.params or {}))

    # 3. build_state cases from the transcripts
    cases, count = [], [0]
    build0 = wpb.build_state

    def build_state(view, memo, names_):
        count[0] += 1
        take = count[0] % a.every == 0
        if take:
            before = (memo.side, rng_bytes(memo.rng.getstate()), memo.messenger_used, memo.fellowship_acted,
                      memo.in_mordor)
        st = build0(view, memo, names_)
        if take:
            text = json.dumps(view, sort_keys=False).encode('utf-8')
            side, rb, mu, fa, im = before
            b = struct.pack('<I', len(text)) + text + struct.pack('<I', 0 if side is FP else 1) + rb
            b += struct.pack('<BBB', int(mu), int(fa), int(im))
            b += ef.graph_bytes('state', snapshot.encode_graph(ai, st))
            cases.append([b, b''])
            last['st'] = st
        else:
            last['st'] = None
        return st
    last = {'st': None}
    search_map0 = wpb.AlphaWotrBot._search_and_map

    def search_and_map(self, st, dec, label, flag, left):
        if last['st'] is st and cases:
            text = json.dumps(dec, sort_keys=False).encode('utf-8')
            lb, fb = str(label).encode('utf-8'), (flag or '').encode('utf-8')
            setup = (struct.pack('<I', len(text)) + text + struct.pack('<I', len(lb)) + lb +
                     struct.pack('<I', len(fb)) + fb + ef.graph_bytes('setup', snapshot.encode_graph(ai, st)))
            # the engine's menu as the adapter takes it: get_legal_actions under the phase flag, then
            # project_ai_actions (LeagueSearcher.choose's menu); each row as canon_action
            from ai_policies import project_ai_actions
            rs = random.getstate()
            try:
                with wpb._Phase(st, flag):
                    legal = list(ai.get_legal_actions(st, self.memo.side))
                    projected = list(project_ai_actions(st, self.memo.side, legal).actions) or legal
                rows = [legal, projected]
            except Exception as error:
                rows = [[], []]
                print('menu failed: %s' % error, file=sys.stderr)
            random.setstate(rs)
            for menu in rows:
                setup += struct.pack('<I', len(menu))
                for act in menu:
                    c = canon_action(act).encode('utf-8')
                    setup += struct.pack('<I', len(c)) + c
            # the option matching of every row of the menu (map_action): the option, a plan, a chain
            setup += struct.pack('<I', len(rows[0]))
            for act in rows[0]:
                try:
                    mt = self.map_action(st, dec, act)
                except Exception:
                    mt = None
                o = -1 if mt is None or mt.o is None else int(mt.o)
                setup += struct.pack('<qBB', o, int(mt is not None and mt.has_plan()),
                                     int(mt is not None and bool(mt.chain)))
            random.setstate(rs)
            cases[-1][1] = setup
            last['searched'] = True
        last['st'] = None
        return search_map0(self, st, dec, label, flag, left)
    wpb.AlphaWotrBot._search_and_map = search_and_map
    decide0 = wpb.AlphaWotrBot._decide

    def _decide(self, dec, view, left):
        last['searched'] = False
        last['search'] = None
        last['stages'] = None
        n = len(cases)
        o, how, info = decide0(self, dec, view, left)
        if last.get('searched') and len(cases) == n + 1 and cases[-1][1]:
            # the adapter's answer and its search (seat seed, search index, search seed, simulations; none: -1 ...)
            sr = last.get('search') or (0, 0, 0, -1)
            hb = str(how).encode('utf-8')
            cases[-1][1] += struct.pack('<q', -1 if o is None else int(o)) + struct.pack('<I', len(hb)) + hb
            cases[-1][1] += struct.pack('<QIQq', *sr)
            stages = json.dumps(last.get('stages') or {}, sort_keys=True, default=str).encode('utf-8')
            cases[-1][1] += struct.pack('<I', len(stages)) + stages
            cases[-1].append(True)
        return o, how, info
    wpb.AlphaWotrBot._decide = _decide
    wpb.build_state = build_state
    from core import own_model
    choose0 = own_model.OwnModelSearch.choose

    def choose(self, st, side, *, num_simulations=300):
        # a seeded 8-simulation search (the harness's convention: random.seed(search seed) before it), recorded
        sims = min(8, num_simulations)
        seed = 9000000 + len(cases) * 16 + (last.get('nsearch') or 0)
        last['nsearch'] = (last.get('nsearch') or 0) + 1
        last['search'] = (int(self.seed), int(self.index), seed, sims)
        random.seed(seed)
        result = choose0(self, st, side, num_simulations=sims)
        stage('search', result[0] if result else None)
        return result

    # the stages of the adapter's decision after the search (LeagueSearcher.choose), recorded as canon_action texts
    def stage(name, act):
        if last.get('stages') is not None:
            last['stages'][name] = None if act is None else canon_action(act)
    raw0 = own_model.OwnModelSearch.raw

    def raw(self, st, side, menu):
        act = raw0(self, st, side, menu)
        stage('raw', act)
        return act
    own_model.OwnModelSearch.raw = raw
    from core import turn_search as ts_mod, ai_units as units_mod
    certain0, units0 = ts_mod.certain, units_mod.pick

    def certain(ai_, seats, st, side, owner, menu, own):
        out = certain0(ai_, seats, st, side, owner, menu, own)
        stage('certain_outright', out[0]); stage('certain_favoured', out[1])
        from core import certain_line as cl
        if last.get('stages') is not None:
            last['stages']['certain_last'] = {k: v for k, v in cl.LAST.items() if k != 'seconds'}
        return out
    ts_mod.certain = certain

    def units(ai_, st, side, act):
        out = units0(ai_, st, side, act)
        stage('units', out)
        return out
    units_mod.pick = units
    league0 = wpb.LeagueSearcher.choose

    def league(self, st, side, kind, banned, flag=None, left_ms=None):
        last['stages'] = {}
        pick, info = league0(self, st, side, kind, banned, flag, left_ms)
        stage('pick', pick)
        last['stages']['how'] = info.get('how') if isinstance(info, dict) else None
        return pick, info
    wpb.LeagueSearcher.choose = league
    own_model.OwnModelSearch.choose = choose
    searcher = wpb.LeagueSearcher(300, None, 1, 1, 'mc200_fpwin_shwin')
    for path in transcripts:
        bot = wpb.AlphaWotrBot(searcher, names)
        before = len(cases)
        with open(path, encoding='utf-8') as f:
            for line in f:
                rec = json.loads(line)
                if rec.get('dir') != 'out':
                    continue
                try:
                    bot.handle(json.loads(rec['line']))
                except SystemExit:
                    break
        print('%s: %d cases' % (os.path.basename(path), len(cases) - before), file=sys.stderr, flush=True)
    with open(os.path.join(out, 'build_state.awb'), 'wb') as f:
        f.write(b'AWB5' + struct.pack('<I', len(cases)))
        for item in cases:
            b, setup = item[0], item[1]
            if setup and len(item) < 3:     # set up and searched but no answer recorded (an error in the decision)
                setup += struct.pack('<q', -1) + struct.pack('<I', 4) + b'none' + struct.pack('<QIQq', 0, 0, 0, -1)
                setup += struct.pack('<I', 2) + b'{}'
            f.write(b + struct.pack('<I', int(bool(setup))) + setup)
    print('wrote %d build_state cases (%d set up and searched)' % (len(cases), sum(1 for c in cases if c[1])), file=sys.stderr)


if __name__ == '__main__':
    main()
