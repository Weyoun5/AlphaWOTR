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

awb: b'AWB1', u32 count, per case: u32 json bytes, json, u32 side, rng (624 u32, u32 index, i32 has_gauss, u64 bits),
     u8 messenger_used, u8 fellowship_acted, u8 in_mordor, graph (awg record): the state
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
    ap.add_argument('transcripts', nargs='+')
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
        special_tiles=sorted(names.special_tile))
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
            cases.append(b)
        return st
    wpb.build_state = build_state
    from core import own_model
    choose0 = own_model.OwnModelSearch.choose

    def choose(self, st, side, *, num_simulations=300):
        return choose0(self, st, side, num_simulations=min(8, num_simulations))
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
        f.write(b'AWB1' + struct.pack('<I', len(cases)))
        for b in cases:
            f.write(b)
    print('wrote %d build_state cases' % len(cases), file=sys.stderr)


if __name__ == '__main__':
    main()
