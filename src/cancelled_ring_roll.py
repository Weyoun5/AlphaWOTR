"""Certify an undone ring attempt whose non-snapshot die roll was restored.

Neither Undo alone nor player chat restores a die face. This accepts only an
explicit same-ID restoration to its independently known preceding face, with
no intervening command except the single Undo of the ring selection.
"""
import re


def normalize(batches, rendered_path, original):
    import log_census as lc
    import client_snapshot_net as sn
    import rendered_replay as rr

    net, events = sn.net_raw(original, with_events=True)
    attempts = [i for i, line in enumerate(original)
                if not net[i] and (move := lc.MOVE_RE.search(line))
                and len(move[1].split()) == 1 and move[1].strip() in lc.DWARVEN_RING_PIDS
                and move[2] == 'Game.2042' and move[3] in ('Game.2035', 'Game.2037')]
    if not attempts:
        return lc.dedupe_echoes(net), []
    public, _ = rr._parse_all(rendered_path)
    sn.validate_rendered_pair(public, original)
    actor = lambda s: s.split('<~', 1)[0].lstrip('$')

    def quiet(s):
        return not s.strip() or bool(re.fullmatch(r'<[^<>~]+> .*', s) and
            '<~' not in s and not re.match(r'<(?:auto|game|client|host|server|replay|system)>', s, re.I))

    raw_rolls = [(i, lc.ROLL_RE.search(s)) for i, s in enumerate(original) if lc.ROLL_RE.search(s)]
    rendered_rolls = [r for r in public if r.get('op') == 'roll_dice']
    # No inferred raw/rendered ordinal when either stream is incomplete.
    if len(raw_rolls) != len(rendered_rolls):
        return lc.dedupe_echoes(net), []
    roll_pairs = {i: r for (i, _), r in zip(raw_rolls, rendered_rolls)}
    known = {}
    proofs = []
    removed_lines = set()
    for i, line in enumerate(original):
        move = lc.MOVE_RE.search(line)
        if move and move[1].split() in ([p] for p in lc.DWARVEN_RING_PIDS) and not net[i]:
            ring = move[1].strip()
            previous = i - 1
            while previous >= 0 and quiet(original[previous]):
                previous -= 1
            staged = lc.MOVE_RE.search(original[previous]) if previous >= 0 else None
            owner = actor(line)
            if move[2] != 'Game.2042' or move[3] not in ('Game.2035', 'Game.2037'):
                continue
            indices = []
            for j in range(i + 1, len(original)):
                if quiet(original[j]):
                    continue
                indices.append(j)
                if len(indices) == 3:
                    break
            if len(indices) != 3:
                continue
            ri, ui, ci = indices
            roll = lc.ROLL_RE.search(original[ri])
            change = lc.CHANGE_RE.search(original[ci])
            undo = [e for e in events if e['line'] == ui + 1]
            if (roll is None or change is None or len(undo) != 1 or
                    undo[0]['op'] != 'undo' or undo[0]['targets'] != [i + 1] or
                    len(roll[1].split()) != 1 or len(roll[2].split()) != 1):
                continue
            pid = roll[1].strip()
            old = known.get(pid)
            if (staged is None or staged[1].split() != [pid] or staged[2] != 'Game.2042' or
                    staged[3] != 'Game.2035' or actor(original[previous]) != owner):
                continue
            if (pid not in lc.BASE_FP_DIE_PIDS | lc.KNOWN_BONUS_FP or
                    old is None or change[1] != pid or change[2] != old or
                    actor(original[ri]) != owner or not original[ci].startswith('$' + owner + ' changes ') or
                    int(roll[2]) not in range(5)):
                continue
            new = 'WCPMH'[int(roll[2])]
            rendered_roll = roll_pairs[ri]
            if rendered_roll.get('player') != owner or rendered_roll.get('faces') != [new]:
                continue
            # Match the complete literal public window; a board/card gesture
            # cannot be hidden by accepting only the two face observations.
            rl = int(rendered_roll['line'])
            # _parse_all labels chat as 'other'; retain only explicit harmless
            # player chat before selecting the adjacent commands.
            def commands(records):
                return [r for r in records if not quiet(str(r.get('body', r.get('raw', ''))))]
            prior = commands([r for r in public if int(r['line']) < rl])
            tail = commands([r for r in public if int(r['line']) > rl])
            if len(prior) < 2 or len(tail) < 2:
                continue
            selection, token, undone, restored = prior[-2], prior[-1], tail[0], tail[1]
            if (selection.get('op') != 'move' or selection.get('obj') != 'Free Action Die (' + old + ')' or
                    selection.get('player') != owner or selection.get('from') != 'FP Dice' or
                    selection.get('to') != 'Current Action'):
                continue
            body = str(restored.get('raw', restored.get('body', '')))
            if (token.get('op') != 'move' or token.get('obj') != 'Dwarven Ring' or
                    token.get('player') != owner or token.get('from') != 'FP Dice' or
                    token.get('to') not in ('Current Action', 'Current FP Card') or
                    undone.get('op') != 'undo' or
                    not re.fullmatch(re.escape(owner) + r' changes Free Action Die \(' + new + r'\) to ' + old + r'\.?', body)):
                continue
            proof = {'ring_id': ring, 'die_id': pid, 'original_face': old, 'attempted_face': new,
                     'raw_ring_line': i + 1, 'raw_roll_line': ri + 1, 'raw_undo_line': ui + 1,
                     'raw_restore_line': ci + 1, 'rendered_roll_line': rl,
                     'rendered_restore_line': restored['line']}
            proofs.append(proof)
            removed_lines.update((rl, restored['line']))
            net[ri] = net[ci] = ''
        roll = lc.ROLL_RE.search(line)
        if roll:
            ids, numbers = roll[1].split(), roll[2].split()
            if len(ids) == len(numbers):
                for pid, value in zip(ids, numbers):
                    if pid in lc.BASE_FP_DIE_PIDS | lc.KNOWN_BONUS_FP and int(value) in range(5):
                        known[pid] = 'WCPMH'[int(value)]
        change = lc.CHANGE_RE.search(line)
        if change:
            known[change[1]] = change[2]
    if proofs:
        for batch in batches:
            for key in ('records', 'rng'):
                if key in batch:
                    batch[key] = [r for r in batch[key] if r.get('line') not in removed_lines]
        batches[0]['_rx_cancelled_ring_rolls'] = proofs
        import hashlib
        batches[0]['_rx_cancelled_ring_source_sha256'] = hashlib.sha256(
            '\n'.join(original).encode('utf8', errors='surrogateescape')).hexdigest()
    return lc.dedupe_echoes(net), proofs
