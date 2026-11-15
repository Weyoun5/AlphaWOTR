"""Recognize literal selected-die reroll/return followed by pool-token payment.

This supplies a candidate interval, not permission to use a Ring. The caller
must additionally prove live ownership, both full untouched turn-start pools,
the previous die face, legal result class, and the public Ring action.
"""
import re


def discover(batches, raw):
    import log_census as lc
    from location_codes_authoritative import AUTH_LOCATION_CODES as zones
    owner = lambda s: s.split('<~', 1)[0].lstrip('$')
    def quiet(s):
        return not s.strip() or bool(re.fullmatch(r'<[^<>~]+> .*', s) and '<~' not in s
            and not re.match(r'<(?:auto|game|client|host|server|replay|system)>', s, re.I))
    commands = [(i, s) for i, s in enumerate(raw) if not quiet(s)]
    records = sorted([(b, r) for b in batches for r in b['records']], key=lambda v: v[1]['line'])
    raw_kills = [(i, lc.KILL_RE.search(s)) for i, s in enumerate(raw)
                 if lc.KILL_RE.search(s) and set(lc.KILL_RE.search(s)[1].split()) & lc.DWARVEN_RING_PIDS]
    public_kills = [(b, r) for b, r in records if r.get('op') == 'kill'
                    and any(str(p).lower() == 'dwarven ring' for p in r.get('pieces', []))]
    if len(raw_kills) != len(public_kills):
        return {}
    result = {}
    for (ki, kill), (batch, payment) in zip(raw_kills, public_kills):
        ids = kill[1].split()
        origin = re.match(r'<~([\w.]+)~>', raw[ki][kill.end():])
        if (payment.get('pieces') != ['Dwarven Ring']
                or len(ids) != 1 or ids[0] not in lc.DWARVEN_RING_PIDS or origin is None
                or origin[1] not in ('Game.2042', 'Game.2043')
                or payment.get('zone') != zones[origin[1]] or payment.get('player') != owner(raw[ki])):
            continue
        pool = origin[1]
        side = 'FREE_PEOPLES' if pool == 'Game.2042' else 'SHADOW'
        k = next(n for n, (i, _) in enumerate(commands) if i == ki)
        if k < 3:
            continue
        # Single selected die, optional literal empty flip, one or more
        # same-ID rolls, exact return, exact payment; nothing else can hide.
        ri, return_line = commands[k - 1]
        returned = lc.MOVE_RE.search(return_line)
        if returned is None or len(returned[1].split()) != 1 or returned[3] != pool:
            continue
        pid, selected_zone = returned[1].strip(), returned[2]
        if selected_zone not in ('Game.2035', 'Game.2037' if side == 'FREE_PEOPLES' else 'Game.2036'):
            continue
        cursor = k - 2
        rolled = []
        while cursor >= 0 and lc.ROLL_RE.search(commands[cursor][1]):
            li, line = commands[cursor]
            roll = lc.ROLL_RE.search(line)
            if roll[1].split() != [pid] or len(roll[2].split()) != 1:
                break
            value = int(roll[2])
            if side == 'FREE_PEOPLES' and value in range(5):
                face = 'WCPMH'[value]
            elif side == 'SHADOW' and value in (*range(6), *range(10, 16)):
                face = 'MHACPE'[value % 10]
            else:
                break
            rolled.append((li, owner(line), face))
            cursor -= 1
        if not rolled or cursor < 0:
            continue
        rolled.reverse()
        flip = None
        if re.fullmatch(r'\$[^<]+<~Controls\.323~>' + re.escape(pid) + r'\s+(?:[A-F0-9]{6}|null)', commands[cursor][1]):
            flip = commands[cursor]
            cursor -= 1
        if cursor < 0:
            continue
        si, stage_line = commands[cursor]
        stage = lc.MOVE_RE.search(stage_line)
        actor = owner(stage_line)
        if (stage is None or stage[1].split() != [pid] or stage[2] != pool or stage[3] != selected_zone
                or any(a != actor for _, a, _ in rolled) or (flip and owner(flip[1]) != actor)):
            continue
        count = 3 + len(rolled) + bool(flip)
        public_k = next(n for n, (_, r) in enumerate(records) if r is payment)
        window = records[public_k - count + 1:public_k + 1] if public_k + 1 >= count else []
        if len(window) != count:
            continue
        selection = window[0][1]
        m = re.fullmatch(r'(Free|Shadow) Action Die \(([WCPMHAE])\)', str(selection.get('obj', '')))
        if (selection.get('op') != 'move' or m is None or
                ('FREE_PEOPLES' if m[1] == 'Free' else 'SHADOW') != side or
                selection.get('player') != actor or selection.get('from') != zones[pool]
                or selection.get('to') != zones[selected_zone]):
            continue
        at = 1
        if flip:
            f = window[at][1]
            if f.get('op') != 'flip' or f.get('obj') != 'nothing' or f.get('player') != actor or f.get('zone') != zones[selected_zone]:
                continue
            at += 1
        expected = [(a, face) for _, a, face in rolled]
        observed = [(r.get('player'), r.get('faces', [None])[0]) for _, r in window[at:at + len(rolled)]
                    if r.get('op') == 'roll_dice' and len(r.get('faces', [])) == 1]
        if observed != expected:
            continue
        ret = window[-2][1]
        label = ('Free' if side == 'FREE_PEOPLES' else 'Shadow') + ' Action Die (' + rolled[-1][2] + ')'
        if (ret.get('op') != 'move' or ret.get('obj') != label or ret.get('player') != owner(return_line)
                or ret.get('from') != zones[selected_zone] or ret.get('to') != zones[pool]):
            continue
        if any(int(b['turn']) != int(batch['turn']) for b, _ in window):
            continue
        proof = {'pid': pid, 'ring_id': ids[0], 'side': side, 'pool': pool, 'old_face': m[2],
                 'raw_stage_line': si + 1, 'raw_cleanup_line': ki + 1,
                 'raw_flip_line': flip[0] + 1 if flip else None,
                 'rendered_stage_line': selection['line'], 'rendered_cleanup_line': payment['line'],
                 'rendered_return_line': ret['line'], 'rendered_selection_lines': [selection['line'], ret['line']],
                 'rendered_flip_line': window[1][1]['line'] if flip else None}
        payment['_rx_unstaged_ring_payment'] = proof
        result[si + 1] = (batch, selection, payment, proof)
    return result
