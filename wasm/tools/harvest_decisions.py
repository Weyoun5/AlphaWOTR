"""Harvest real decisions from protocol transcripts: the states the protocol adapter hands its search.

    venv/bin/python -I wasm/tools/harvest_decisions.py --alpha ALPHAWOTR_CHECKOUT --out FILE.pkl
        [--every K] [--battles B] TRANSCRIPT.protocol.jsonl ...

A transcript is `cli referee`'s protocol log of one seat (lines {"dir": "out"|"in", "line": message}). The referee's
messages ("out") are handed to the adapter (protocol/wotr_protocol_bot.py AlphaWotrBot.handle, the league searcher at
strength 1) in order; its answers are dropped (the referee's next message is the record's). Each search the adapter
asks for is recorded instead of run in full (it is answered by an 8-simulation search, so the adapter's chains go on):
the state as handed (pickled), the side, the simulations it asked for. Every K-th search is kept, and every search
inside a battle (pending_combat) up to B per transcript. The file has record_searches.py's --positions shape
('handed', 'side', 'owner', 'menu' [] , 'turn', plus 'sims', 'battle', 'source').
"""
import argparse
import json
import os
import pickle
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--every', type=int, default=10)
    ap.add_argument('--battles', type=int, default=6)
    ap.add_argument('transcripts', nargs='+')
    a = ap.parse_args()
    alpha = os.path.abspath(a.alpha)
    out = os.path.abspath(a.out)
    transcripts = [os.path.abspath(t) for t in a.transcripts]
    sys.path[:0] = [os.path.join(alpha, 'protocol'), os.path.join(alpha, 'wotr_bridge')]
    import wotr_protocol_bot as wpb
    wpb.import_engine(os.path.join(alpha, 'src'), True)
    ai = wpb.ai
    names = wpb.Names()
    searcher = wpb.LeagueSearcher(300, None, 1, 1, 'mc200_fpwin_shwin')
    from core import own_model
    kept, state = [], {'n': 0, 'battles': 0, 'source': None}
    choose0 = own_model.OwnModelSearch.choose

    def choose(self, st, side, *, num_simulations=300):
        state['n'] += 1
        battle = bool(getattr(st, 'pending_combat', None))
        take = state['n'] % a.every == 0 or (battle and state['battles'] < a.battles)
        if take:
            state['battles'] += battle
            kept.append(dict(handed=pickle.dumps(st), side=side.name, owner=None, menu=[], turn=getattr(st, 'turn', None),
                             sims=int(num_simulations), battle=battle, source='%s#%d' % (state['source'], state['n'])))
        return choose0(self, st, side, num_simulations=min(8, num_simulations))
    own_model.OwnModelSearch.choose = choose
    for path in transcripts:
        bot = wpb.AlphaWotrBot(searcher, names)
        state.update(n=0, battles=0, source=os.path.basename(path))
        before = len(kept)
        with open(path, encoding='utf-8') as f:
            for line in f:
                rec = json.loads(line)
                if rec.get('dir') != 'out':
                    continue
                try:
                    bot.handle(json.loads(rec['line']))
                except SystemExit:
                    break
        print('%s: %d searches, %d kept (%d in battles)' % (
            os.path.basename(path), state['n'], len(kept) - before,
            sum(1 for k in kept[before:] if k['battle'])), file=sys.stderr, flush=True)
    with open(out, 'wb') as f:
        pickle.dump(kept, f)
    print('wrote %d decisions to %s' % (len(kept), out), file=sys.stderr)


if __name__ == '__main__':
    main()
